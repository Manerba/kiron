#!/bin/bash
# Deployt Kiron aus dem Git-Repo (/opt/kiron) in die Produktiv-Umgebung (/usr/lib/kiron).
# Verwendung: bash /opt/kiron/scripts/deploy-local.sh [--restart]
set -Ee

# Argumentvalidierung: Typos wie --restartt wuerden sonst stumm uebersprungen,
# Deploy meldet Erfolg, Services bleiben aber auf altem Code.
if [ "$#" -gt 1 ]; then
    echo "FEHLER: Zu viele Argumente. Verwendung: bash $0 [--restart]" >&2
    exit 1
fi
case "${1:-}" in
    ""|--restart) ;;
    *)
        echo "FEHLER: Unbekanntes Argument '$1'. Verwendung: bash $0 [--restart]" >&2
        exit 1
        ;;
esac

SRC="/opt/kiron"
DST="/usr/lib/kiron"
REPORT_DIR="$SRC/data/ollama_compat_reports"
RUNTIME_HANDOFF="$DST/data/ollama_compat_runtime.json"
SKIP_GATE="${KIRON_SKIP_OLLAMA_COMPAT_GATE:-0}"

echo "Deploye Kiron nach ${DST}..."

# Preflight: Quellen pruefen, bevor irgendetwas veraendert wird.
# Vermeidet die haeufigste Ursache fuer einen mittendrin abgebrochenen Deploy.
for svc in kiron-proxy kiron-docling kiron-embeddings kiron-deberta; do
    if [ ! -d "$SRC/services/$svc" ]; then
        echo "FEHLER: Quelle $SRC/services/$svc fehlt. Deploy abgebrochen." >&2
        exit 1
    fi
done
if [ ! -f "$SRC/services/kiron-common/pyproject.toml" ] || [ ! -d "$SRC/services/kiron-common/kiron_common" ]; then
    echo "FEHLER: Quelle $SRC/services/kiron-common fehlt. Deploy abgebrochen." >&2
    exit 1
fi
for f in version.txt docker/docker-compose.yml data/db_config.json; do
    if [ ! -f "$SRC/$f" ]; then
        echo "FEHLER: Quelle $SRC/$f fehlt. Deploy abgebrochen." >&2
        exit 1
    fi
done
if ! ls "$SRC/systemd/"*.service >/dev/null 2>&1; then
    echo "FEHLER: keine systemd-Unit in $SRC/systemd/ gefunden. Deploy abgebrochen." >&2
    exit 1
fi

compose_ollama_image() {
    python3 - "$1" <<'PY'
import sys
path = sys.argv[1]
in_ollama = False
with open(path, encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("  ollama:"):
            in_ollama = True
            continue
        if in_ollama and line.startswith("  ") and not line.startswith("    "):
            break
        # Genau 4 Spaces vor 'image:' verlangen, sonst matched ein nested
        # Sub-Key wie services.ollama.build.image (6 Spaces) den Tag faelschlich.
        if in_ollama and line.startswith("    image:") and not line.startswith("     "):
            print(line.split(":", 1)[1].split("#", 1)[0].strip().strip("'\""))
            raise SystemExit(0)
sys.stderr.write(
    f"FEHLER: Compose-Service 'ollama' (mit 'image:') in {path} nicht gefunden. "
    f"Service-Name ist im Parser hartcodiert; bei Umbenennung Parser anpassen.\n"
)
raise SystemExit(1)
PY
}

docker_image_snapshot() {
    local image="$1"
    # Atomarer Snapshot: ImageID + alle RepoDigests in EINEM docker image inspect
    # Aufruf. Zwei separate Calls oeffnen ein Race-Window in dem ein paralleler
    # `docker pull` das Tag-zu-ID-Mapping verschieben kann -- Compat-Report waere
    # dann fuer Digest A gruen, Container liefe mit Image B (#851, #1002).
    # Ausgabeformat: Zeile 1 = .Id, Zeile 2..N = RepoDigests (unsortiert).
    docker image inspect "$image" \
        --format '{{println .Id}}{{range .RepoDigests}}{{println .}}{{end}}' 2>/dev/null
}

find_compat_report() {
    local image="$1"
    local digest="$2"
    python3 "$SRC/scripts/check-ollama-compat.py" \
        --find-report \
        --image "$image" \
        --digest "$digest" \
        --report-dir "$REPORT_DIR"
}

validator_command_hint() {
    printf '  %s --image %s\n' "$SRC/scripts/check-ollama-compat.py" "$NEW_OLLAMA_IMAGE"
}

write_runtime_handoff() {
    local digest="$1"
    local report_path="$2"
    local effective="$3"
    mkdir -p "$(dirname "$RUNTIME_HANDOFF")"
    python3 - "$RUNTIME_HANDOFF" "$digest" "$report_path" "$effective" <<'PY'
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "image_digest": sys.argv[2],
    "report_path": sys.argv[3],
    "num_gpu_zero_effective": sys.argv[4] == "true",
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
os.chmod(tmp, 0o644)
os.replace(tmp, path)
PY
}

check_common_venvs() {
    # Frueher --restart-Preflight. Importprobe ist hier sinnlos: kiron-common
    # ist editable aus $DST installiert, und rsync von kiron-common passiert
    # erst spaeter — die Pruefung wuerde den alten Code testen (#782).
    # Alle vier Services pruefen, da der spaetere systemctl-restart-Loop alle
    # vier abdeckt — fehlt ein venv, soll der Abbruch vor mutierenden Schritten
    # passieren statt erst nach rsync und Docker-Handoff.
    for svc in kiron-proxy kiron-docling kiron-embeddings kiron-deberta; do
        py="$DST/services/$svc/venv/bin/python"
        if [ ! -x "$py" ]; then
            echo "FEHLER: $py fehlt; --restart benoetigt vorhandene venvs aller Services." >&2
            echo "Fuehre die sichere Sequenz aus:" >&2
            echo "  bash /opt/kiron/scripts/deploy-local.sh" >&2
            echo "  bash /opt/kiron/scripts/setup-venvs.sh" >&2
            echo "  bash /opt/kiron/scripts/deploy-local.sh --restart" >&2
            return 1
        fi
    done
}

smoke_common_imports() {
    # MUSS nach rsync von kiron-common laufen, damit der aktuelle Code geprueft wird.
    # Iteriert alle vier Services analog zu check_common_venvs und Restart-Loop;
    # Services ohne kiron-common im venv (aktuell kiron-embeddings/-deberta, vgl.
    # setup-venvs.sh:140) werden uebersprungen. Sobald ein weiterer Service
    # kiron_common adoptiert, deckt der Smoke-Test ihn automatisch ab — ohne
    # dass die Service-Liste an zwei Stellen synchron gehalten werden muss.
    for svc in kiron-proxy kiron-docling kiron-embeddings kiron-deberta; do
        py="$DST/services/$svc/venv/bin/python"
        if ! "$py" -c 'import kiron_common' 2>/dev/null; then
            continue
        fi
        "$py" -c 'import kiron_common.ollama_compat' || {
            echo "FEHLER: kiron_common.ollama_compat Import in $svc venv fehlgeschlagen." >&2
            echo "Fuehre die sichere Sequenz aus:" >&2
            echo "  bash /opt/kiron/scripts/deploy-local.sh" >&2
            echo "  bash /opt/kiron/scripts/setup-venvs.sh" >&2
            echo "  bash /opt/kiron/scripts/deploy-local.sh --restart" >&2
            return 1
        }
    done
}

validate_ollama_runtime_contract() {
    local target_image_id="$1"
    if ! docker inspect kiron-ollama \
        | python3 "$SRC/scripts/validate-ollama-runtime-contract.py" \
            --target-image-id "$target_image_id" \
            --expected-image "$NEW_OLLAMA_IMAGE"; then
        echo "FEHLER: kiron-ollama Runtime-Contract passt nicht zur erwarteten Compose-Konfiguration." >&2
        echo "       Runtime-Handoff abgebrochen; Container per Compose neu erstellen:" >&2
        echo "         docker compose -f $DST/docker/docker-compose.yml up -d --force-recreate ollama" >&2
        return 1
    fi
}

OLD_OLLAMA_IMAGE=""
NEW_OLLAMA_IMAGE="$(compose_ollama_image "$SRC/docker/docker-compose.yml")"
OLLAMA_IMAGE_CHANGED=0
OLLAMA_SERVICE_CHANGED=0
TARGET_DIGEST=""
MATCHING_REPORT=""
MATCHING_REPORT_NUM_GPU="false"
if [ -f "$DST/docker/docker-compose.yml" ]; then
    OLD_OLLAMA_IMAGE="$(compose_ollama_image "$DST/docker/docker-compose.yml" || true)"
    if [ "$OLD_OLLAMA_IMAGE" != "$NEW_OLLAMA_IMAGE" ]; then
        OLLAMA_IMAGE_CHANGED=1
    elif ! cmp -s "$SRC/docker/docker-compose.yml" "$DST/docker/docker-compose.yml"; then
        OLLAMA_SERVICE_CHANGED=1
    fi
else
    OLLAMA_SERVICE_CHANGED=1
fi

if [ "${1:-}" = "--restart" ]; then
    check_common_venvs
fi

if [ "$OLLAMA_IMAGE_CHANGED" = "1" ]; then
    echo "Ollama-Image-Aenderung: ${OLD_OLLAMA_IMAGE:-<none>} -> $NEW_OLLAMA_IMAGE"
    docker pull "$NEW_OLLAMA_IMAGE"
fi

# ImageID + RepoDigest atomar in EINEM docker image inspect Aufruf einfrieren.
# Zwei separate Inspect-Calls wuerden ein Race-Window oeffnen, in dem ein
# paralleler `docker pull` das Tag-zu-ID-Mapping zwischen Compat-Lookup und
# Container-Start verschiebt -- Handoff persistiert dann image_digest A,
# waehrend Container mit Image B startet (#851, #1002).
TARGET_INSPECT="$(docker_image_snapshot "$NEW_OLLAMA_IMAGE")"
TARGET_IMAGE_ID="$(printf '%s\n' "$TARGET_INSPECT" | sed -n '1p')"
# Lexikografisch kleinster RepoDigest -- muss mit check-ollama-compat.py:_image_digest
# uebereinstimmen, sonst schlaegt der Report-Lookup bei mehreren Mirror-Tags fehl.
TARGET_DIGEST="$(printf '%s\n' "$TARGET_INSPECT" \
    | tail -n +2 \
    | sed '/^[[:space:]]*$/d' \
    | LC_ALL=C sort \
    | head -n1)"
if [ -z "$TARGET_DIGEST" ]; then
    echo "FEHLER: RepoDigest fuer $NEW_OLLAMA_IMAGE konnte nicht ermittelt werden." >&2
    echo "Fuehre zuerst den Ollama-Compat-Validator aus, damit das geplante Image gepullt und geprueft wird:" >&2
    validator_command_hint >&2
    exit 1
fi
if [ -z "$TARGET_IMAGE_ID" ]; then
    echo "FEHLER: ImageID fuer $NEW_OLLAMA_IMAGE konnte nicht ermittelt werden." >&2
    validator_command_hint >&2
    exit 1
fi
COMPAT_FIND_STDERR="$(mktemp)"
if report_info="$(find_compat_report "$NEW_OLLAMA_IMAGE" "$TARGET_DIGEST" 2>"$COMPAT_FIND_STDERR")"; then
    MATCHING_REPORT="$(printf '%s\n' "$report_info" | sed -n '1p')"
    MATCHING_REPORT_NUM_GPU="$(printf '%s\n' "$report_info" | sed -n '2p')"
    if [ "$MATCHING_REPORT_NUM_GPU" != "true" ]; then
        if [ "$SKIP_GATE" = "1" ]; then
            echo "WARNUNG: KIRON_SKIP_OLLAMA_COMPAT_GATE=1 gesetzt; Compat-Report $MATCHING_REPORT ist nicht num_gpu=0-sicher; Runtime-Handoff bleibt fail-closed." >&2
            MATCHING_REPORT_NUM_GPU="false"
        else
            echo "FEHLER: Compat-Report $MATCHING_REPORT ist gruen, aber capabilities.num_gpu_zero_chat_generate.data.num_gpu_zero_effective ist nicht true." >&2
            echo "Fuehre vor dem Deploy einen aktuellen Validator-Lauf aus:" >&2
            validator_command_hint >&2
            rm -f "$COMPAT_FIND_STDERR"
            exit 1
        fi
    fi
    echo "  Compat-Report akzeptiert: $MATCHING_REPORT"
else
    # Validator-Exit: 1 = kein Match, 2 = unerwarteter Crash. Alles andere
    # (z.B. 127 python3 missing, 126 permission denied, 139 segfault) ist
    # unerwartet und wird ebenfalls als Crash behandelt -- sonst sieht der
    # Operator die irrefuehrende "Kein Match"-Meldung, obwohl die wahre
    # Ursache anderswo liegt (#1056). SKIP_GATE bypass bleibt bewusst nur
    # fuer rc=1 (sauberer Validator-Lauf ohne Treffer) zugelassen.
    FIND_REPORT_RC=$?
    if [ "$FIND_REPORT_RC" != "1" ]; then
        echo "FEHLER: Ollama-Compat-Validator (--find-report) abgestuerzt (exit $FIND_REPORT_RC). stderr:" >&2
        sed 's/^/  /' "$COMPAT_FIND_STDERR" >&2
        echo "Fuehre vor dem Deploy den Validator fuer das aktuelle/geplante Ollama-Image aus:" >&2
        validator_command_hint >&2
        rm -f "$COMPAT_FIND_STDERR"
        exit 1
    elif [ "$SKIP_GATE" = "1" ]; then
        echo "WARNUNG: KIRON_SKIP_OLLAMA_COMPAT_GATE=1 gesetzt; fehlender gruener Report wird uebergangen." >&2
        echo "WARNUNG: Runtime-Handoff wird fail-closed geschrieben; num_gpu=0-CPU-Offload bleibt blockiert." >&2
        MATCHING_REPORT="override"
        MATCHING_REPORT_NUM_GPU="false"
    else
        echo "FEHLER: Kein passender gruener Ollama-Compat-Report fuer $NEW_OLLAMA_IMAGE / $TARGET_DIGEST." >&2
        echo "Fuehre vor dem Deploy den Validator fuer das aktuelle/geplante Ollama-Image aus:" >&2
        validator_command_hint >&2
        rm -f "$COMPAT_FIND_STDERR"
        exit 1
    fi
fi
rm -f "$COMPAT_FIND_STDERR"

# Bei Abbruch mitten im Deploy klare Meldung statt stillem set-e-Exit.
fail_partial() {
    echo "" >&2
    echo "FEHLER: Deploy bei einem Zwischenschritt abgebrochen." >&2
    echo "ACHTUNG: $DST kann in inkonsistentem Zustand sein (teils neu, teils alt)." >&2
    echo "         $DST/version.txt wurde bewusst NICHT aktualisiert und zeigt weiterhin" >&2
    echo "         die zuletzt vollstaendig deployte Version an." >&2
    echo "         Ursache beheben, dann diesen Deploy erneut ausfuehren." >&2
}
trap fail_partial ERR

abort_partial() {
    fail_partial
    exit 1
}

# Verzeichnisstruktur sicherstellen
mkdir -p "$DST"/{services/kiron-proxy,services/kiron-docling,services/kiron-embeddings,services/kiron-deberta,services/kiron-common,data,docker}

# Service-rsync gekapselt, damit alle Services identisch behandelt werden.
sync_service() {
    local name="$1"
    shift
    rsync -a --delete \
        --exclude='venv/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        "$@" \
        "$SRC/services/$name/" "$DST/services/$name/"
}

sync_service kiron-proxy      --exclude='requirements-lock.txt'
sync_service kiron-docling    --exclude='requirements-lock.txt'
sync_service kiron-embeddings --exclude='requirements-lock.txt'
sync_service kiron-deberta    --exclude='requirements-lock.txt'
rsync -a --delete \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    "$SRC/services/kiron-common/" "$DST/services/kiron-common/"

# Importprobe gegen den jetzt aktuellen Common-Code, bevor Docker-Compose,
# Runtime-Handoff und systemd-Aenderungen Side-Effects haben (#782).
if [ "${1:-}" = "--restart" ]; then
    smoke_common_imports
fi

# Docker-Compose
rsync -a --delete "$SRC/docker/" "$DST/docker/"

# External Docker-Volume fuer Ollama sicherstellen (#232)
if command -v docker >/dev/null 2>&1; then
    if ! docker volume inspect botmin_ollama >/dev/null 2>&1; then
        docker volume create botmin_ollama >/dev/null
        echo "  docker volume botmin_ollama: erstellt"
    fi
fi

# validate_ollama_runtime_contract verlangt unten einen laufenden kiron-ollama;
# ohne dieses Hochfahren bricht der Deploy bei gestopptem/fehlendem Container
# erst nach dem Service-rsync ab und hinterlaesst einen Halb-Deploy (#884).
OLLAMA_NEEDS_START=0
if [ "$(docker inspect kiron-ollama --format '{{.State.Running}}' 2>/dev/null)" != "true" ]; then
    OLLAMA_NEEDS_START=1
fi

if [ "$OLLAMA_IMAGE_CHANGED" = "1" ] || [ "$OLLAMA_SERVICE_CHANGED" = "1" ] || [ "$OLLAMA_NEEDS_START" = "1" ]; then
    echo "Aktualisiere Ollama-Compose-Service..."
    # Pre-Handoff fail-closed schreiben (#947), bevor compose up den Container
    # neu startet. Wenn validate_ollama_runtime_contract spaeter abortet, laeuft
    # der Container schon mit dem neuen Image — der alte Handoff wuerde sonst
    # ein veraltetes image_digest beschreiben und vram_lease.py num_gpu=0 fuer
    # ein potentiell unsicheres Image freigeben. Der zweite write_runtime_handoff
    # nach erfolgreichem validate ueberschreibt atomar mit dem finalen Wert.
    write_runtime_handoff "$TARGET_DIGEST" "${MATCHING_REPORT:-override}" "false"
    docker compose -f "$DST/docker/docker-compose.yml" up -d ollama
fi

# Vor dem Handoff IMMER pruefen, dass kiron-ollama tatsaechlich mit dem Image laeuft,
# fuer das der Compat-Report ermittelt wurde. TARGET_IMAGE_ID stammt aus dem Snapshot
# direkt nach dem RepoDigest-Lookup (#851) — eine spaetere Tag-Drift zwischen
# Compat-Lookup und Container-Start fuehrt damit zu einem ImageID-Mismatch in
# validate_ollama_runtime_contract statt zu einem inkonsistenten Handoff.
if ! validate_ollama_runtime_contract "$TARGET_IMAGE_ID"; then
    abort_partial
fi

write_runtime_handoff "$TARGET_DIGEST" "${MATCHING_REPORT:-override}" "$MATCHING_REPORT_NUM_GPU"
echo "  Ollama Runtime-Handoff aktualisiert: $RUNTIME_HANDOFF"

# Data: db_config.json nur kopieren wenn am Ziel noch nicht vorhanden
if [ ! -f "$DST/data/db_config.json" ]; then
    cp "$SRC/data/db_config.json" "$DST/data/db_config.json"
    echo "  db_config.json: initial kopiert"
else
    echo "  db_config.json: existiert bereits, uebersprungen"
fi

# systemd-Units installieren
cp "$SRC/systemd/"*.service /etc/systemd/system/
systemctl daemon-reload
echo "  systemd-Units aktualisiert"

# Optional: Services neustarten
# Sequentiell, damit die 4 Services nicht gleichzeitig um GPU/VRAM konkurrieren (#545).
# Proxy zuerst, damit das Dashboard/OpenAI-API so frueh wie moeglich wieder antwortet.
if [ "$1" = "--restart" ]; then
    echo "Starte Services neu..."
    # Per-Service-Fehler tolerieren statt fail-fast: sonst bleiben die uebrigen
    # Services mit altem in-memory-Code stehen, waehrend on-disk schon der neue
    # Code liegt — Halb-Deploy-Zustand (#936).
    failed_services=()
    for svc in kiron-proxy kiron-docling kiron-embeddings kiron-deberta; do
        if ! systemctl restart "$svc"; then
            failed_services+=("$svc")
            echo "  FEHLER: systemctl restart $svc fehlgeschlagen, fahre mit naechstem Service fort." >&2
        fi
    done
    if [ ${#failed_services[@]} -gt 0 ]; then
        echo "FEHLER: Restart fehlgeschlagen fuer: ${failed_services[*]}" >&2
        echo "       Logs pruefen via: journalctl -u <service> -n 50" >&2
        abort_partial
    fi
    echo "Services neugestartet."
fi

# version.txt als allerletzter Schritt: fungiert als Deploy-Complete-Marker.
# Bei Abbruch vorher bleibt die alte Version sichtbar und signalisiert Inkonsistenz.
cp "$SRC/version.txt" "$DST/version.txt"

trap - ERR

echo "Deploy abgeschlossen: $(cat "$DST/version.txt")"
