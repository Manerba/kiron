"""Per-operation end evidence for the loopback-only native service.

Receipts are process-local and bounded. Missing/evicted receipts, a backend
restart, or unsuccessful CUDA synchronization never imply termination.
"""
import asyncio
from collections import OrderedDict
from uuid import uuid4

from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse

from kiron_common.gpu_admission.native_contract import (
    COMPLETION_HEADER, OPERATION_HEADER, OVERLAY_HEADER, valid_operation_id,
)


class OperationLedger:
    def __init__(self, limit=4096):
        self.instance = uuid4().hex
        self.limit = limit
        self.records = OrderedDict()

    def begin(self, operation_id, overlay_token):
        if operation_id in self.records:
            return False
        if len(self.records) >= self.limit:
            finished = next((key for key, row in self.records.items()
                             if row["state"] == "terminated"), None)
            if finished is None:
                return False
            del self.records[finished]
        self.records[operation_id] = {
            "schema_version": 1, "backend_instance": self.instance,
            "operation_id": operation_id, "overlay_token": overlay_token,
            "state": "active",
        }
        return True

    def finish(self, operation_id, confirmed):
        self.records[operation_id]["state"] = "terminated" if confirmed else "unknown"

    def snapshot(self, operation_id):
        row = self.records.get(operation_id)
        return dict(row) if row is not None else None


def tracked_route_class(ledger, confirm_end):
    class TrackedRoute(APIRoute):
        def get_route_handler(self):
            endpoint = super().get_route_handler()

            async def handle(request):
                operation_id = request.headers.get(OPERATION_HEADER)
                if request.method != "POST" or operation_id is None:
                    return await endpoint(request)
                overlay = request.headers.get(OVERLAY_HEADER)
                if not valid_operation_id(operation_id) or (
                        overlay is not None and not valid_operation_id(overlay)):
                    return JSONResponse({"error": "invalid native operation identity"}, status_code=400)
                if not ledger.begin(operation_id, overlay):
                    # No completion receipt: a replay must not clear the original.
                    return JSONResponse({"error": "native operation identity unavailable"}, status_code=503)

                async def finish():
                    try:
                        confirmed = await confirm_end()
                    except Exception:
                        confirmed = False
                    ledger.finish(operation_id, confirmed is True)
                    return confirmed is True

                response = None
                try:
                    try:
                        response = await endpoint(request)
                    except RequestValidationError as exc:
                        # Build the standard validation response here so it can
                        # carry end evidence after the same cleanup as inference.
                        response = await request_validation_exception_handler(request, exc)
                finally:
                    # Cancellation must not detach the service's cleanup worker.
                    cleanup = asyncio.create_task(finish())
                    cancelled = False
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            cancelled = True
                            continue
                    if response is not None and cleanup.result():
                        response.headers[COMPLETION_HEADER] = operation_id
                    if cancelled:
                        raise asyncio.CancelledError
                return response

            return handle

    return TrackedRoute
