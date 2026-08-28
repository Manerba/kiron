"""kitt-worker service skeleton."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from types import SimpleNamespace
from typing import Any


logger = logging.getLogger(__name__)
_RUNTIME: SimpleNamespace | None = None

STARTUP_EXIT_OK = 0
STARTUP_EXIT_CONFIG_ERROR = 2
STARTUP_EXIT_RUNTIME_ERROR = 1


def _load_runtime_modules() -> SimpleNamespace:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn

    import artifact_staging
    import config
    import contract
    import executor
    import monitoring
    import queue_store
    import v1

    _RUNTIME = SimpleNamespace(
        FastAPI=FastAPI,
        JSONResponse=JSONResponse,
        artifact_staging=artifact_staging,
        config=config,
        contract=contract,
        executor=executor,
        monitoring=monitoring,
        queue_store=queue_store,
        uvicorn=uvicorn,
        v1=v1,
    )
    return _RUNTIME


def create_app(
    *,
    validate_runtime: bool = True,
    validate_auth: bool = True,
    load_config_kwargs: dict[str, Any] | None = None,
    load_auth_kwargs: dict[str, Any] | None = None,
    runtime: SimpleNamespace | None = None,
) -> Any:
    rt = _load_runtime_modules() if runtime is None else runtime
    artifact_staging = rt.artifact_staging
    config = rt.config
    contract = rt.contract
    executor = rt.executor
    monitoring = rt.monitoring
    queue_store = rt.queue_store
    v1 = rt.v1
    load_config_kwargs = {} if load_config_kwargs is None else dict(load_config_kwargs)
    load_auth_kwargs = {} if load_auth_kwargs is None else dict(load_auth_kwargs)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = config.load_config(
            validate_runtime=validate_runtime,
            **load_config_kwargs,
        )
        config.configure_logging(cfg)
        app.state.worker_config = cfg
        app.state.queue_store = None
        app.state.queue_unavailable = None
        app.state.artifact_staging_root = None
        app.state.artifact_staging_unavailable = None
        app.state.publish_verify_keyring = None
        app.state.publish_verify_keyring_unavailable = (
            cfg.publish_verify_keyring_config_error_code
        )
        app.state.monitoring_manager = monitoring.MonitoringManager(
            cfg=cfg,
            app_state=app.state,
        )
        credential_store = None
        if validate_auth and cfg.enable_v1 and cfg.auth_mode == "required":
            try:
                credential_store = config.load_auth_store(cfg, **load_auth_kwargs)
            except config.RuntimeConfigError:
                logger.warning("kitt-worker auth store not ready for /v1")
        app.state.credential_store = credential_store
        try:
            app.state.publish_verify_keyring = config.load_publish_verify_keyring(cfg)
            if app.state.publish_verify_keyring is not None:
                app.state.publish_verify_keyring_unavailable = None
        except config.RuntimeConfigError:
            app.state.publish_verify_keyring = None
            app.state.publish_verify_keyring_unavailable = "config_invalid"
            logger.warning("kitt-worker publish verify keyring unavailable")
        lease_sink = (
            app.state.monitoring_manager.record_stale_leases
            if app.state.monitoring_manager.enabled
            else None
        )
        try:
            store = queue_store.QueueStore.open_from_config(
                cfg,
                lease_expiry_sink=lease_sink,
            )
            store.expire_stale_leases()
            app.state.queue_store = store
        except queue_store.QueueUnavailable as exc:
            app.state.queue_unavailable = queue_store.QueueUnavailableState(
                exc.reason_code
            )
            logger.warning(
                "kitt-worker queue unavailable reason_code=%s",
                exc.reason_code,
            )
        try:
            app.state.artifact_staging_root = artifact_staging.validate_staging_root(cfg)
        except artifact_staging.ArtifactStagingError as exc:
            app.state.artifact_staging_unavailable = exc.reason_code
            logger.info(
                "kitt-worker artifact staging unavailable reason_code=%s",
                exc.reason_code,
            )
        logger.info(
            "kitt-worker skeleton ready on %s:%s with auth %s",
            cfg.bind_host,
            cfg.port,
            cfg.auth_mode,
        )
        executor_task = executor.start_executor_task(app.state)
        monitoring_task = monitoring.start_monitoring_task(app.state)
        try:
            yield
        finally:
            if monitoring_task is not None:
                await monitoring.stop_monitoring_task(app.state)
            if executor_task is not None:
                await executor.stop_executor_task(app.state)

    app = rt.FastAPI(
        title="Kiron kitt-worker",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    contract.install_exception_handlers(app)

    @app.get(config.DEFAULT_HEALTH_PATH)
    async def internal_health():
        return rt.JSONResponse(
            {
                "status": "ok",
                "service": "kitt-worker",
                "mode": "skeleton",
            }
        )

    app.include_router(v1.router)

    return app


def run(
    *,
    load_config_kwargs: dict[str, Any] | None = None,
    uvicorn_run=None,
    runtime_loader=_load_runtime_modules,
) -> int:
    load_config_kwargs = {} if load_config_kwargs is None else dict(load_config_kwargs)
    try:
        rt = runtime_loader()
    except Exception:
        _log_startup_failure("startup_failed")
        return STARTUP_EXIT_RUNTIME_ERROR

    try:
        cfg = rt.config.load_config(validate_runtime=True, **load_config_kwargs)
    except rt.config.ConfigError:
        _log_startup_failure("config_invalid")
        return STARTUP_EXIT_CONFIG_ERROR
    except Exception:
        _log_startup_failure("startup_failed")
        return STARTUP_EXIT_RUNTIME_ERROR

    rt.config.configure_logging(cfg)
    try:
        app = create_app(runtime=rt)
        selected_uvicorn_run = rt.uvicorn.run if uvicorn_run is None else uvicorn_run
        selected_uvicorn_run(
            app,
            host=cfg.bind_host,
            port=cfg.port,
            log_level=cfg.log_level.lower(),
            access_log=False,
        )
    except Exception:
        _log_startup_failure("startup_failed")
        return STARTUP_EXIT_RUNTIME_ERROR
    return STARTUP_EXIT_OK


def main() -> None:
    raise SystemExit(run())


def _log_startup_failure(reason_code: str) -> None:
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.error(
        "event=kitt_worker_startup_error reason_code=%s",
        reason_code,
        exc_info=False,
    )


if __name__ == "__main__":
    main()
