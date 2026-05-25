#!/bin/bash
# Docling-serve Container mit korrekter Konfiguration erstellen.
# Ausfuehren wenn der Container komplett neu erstellt werden muss
# (z.B. nach docker rm oder Image-Update).
#
# Der Container bekommt bewusst keine Docker-Auto-Restart-Policy.
# Starts laufen ueber den gegateten Proxy-/Dashboard-Pfad, damit Ollama-
# GPU-Requests waehrend Docling-Startup/Shutdown blockiert werden.
#
# Portbindung: Host-Port 5002 wird localhost-only gebunden, damit der
# docling-serve-HTTP-Port nicht extern erreichbar ist. Der Proxy auf
# Port 5001 (services/kiron-docling/proxy.py) ist der einzige legitime
# Vermittler nach aussen (Issue #260 — Proxy-Port 0.0.0.0 ist wontfix).

set -e

CONTAINER=docling-serve
IMAGE=quay.io/docling-project/docling-serve-cu130:v1.14.0

echo "Stoppe und entferne alten Container..."
docker stop -t 10 "$CONTAINER" 2>/dev/null || true
docker rm "$CONTAINER" 2>/dev/null || true

echo "Erstelle Container..."
docker create --name "$CONTAINER" \
  --gpus all \
  -p 127.0.0.1:5002:5001 \
  --restart=no \
  --memory=14g \
  --memory-swap=-1 \
  -e DOCLING_SERVE_OCR_BATCH_SIZE=16 \
  -e DOCLING_SERVE_LAYOUT_BATCH_SIZE=16 \
  -e DOCLING_SERVE_TABLE_BATCH_SIZE=16 \
  -e DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2 \
  -e DOCLING_SERVE_MAX_SYNC_WAIT=3600 \
  -e DOCLING_NUM_THREADS=4 \
  -e DOCLING_DEVICE=cuda \
  "$IMAGE"

echo "Container erstellt. Wird vom Proxy bei Bedarf gestartet."
echo "Normaler Startpfad: Dashboard /api/docling/start oder ein Docling-Proxy-Request."
echo "Direktes 'docker start $CONTAINER' ist nur ein Notfall-Ops-Eingriff mit VRAM-Risiko."
