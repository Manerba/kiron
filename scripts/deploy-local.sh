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
KIRON_SERVICES=(kiron-proxy kiron-docling kiron-embeddings kiron-deberta kiron-prism kitt-worker)
COMMON_CONSUMERS=(kiron-proxy kiron-docling kiron-embeddings kiron-deberta kiron-prism kitt-worker)
PROXY_DATA_DIR="$DST/data/kiron-proxy"
SHARED_DATA_DIR="$DST/data/shared"
MODEL_REGISTRY_FILE="$SHARED_DATA_DIR/local-model-registry.json"
MODEL_REGISTRY_LOCK_FILE="$MODEL_REGISTRY_FILE.lock"

echo "Deploye Kiron nach ${DST}..."

service_group() {
    case "$1" in
        kiron-proxy) echo "kiron-proxy" ;;
        kiron-docling) echo "kiron-docling" ;;
        kiron-embeddings) echo "kiron-embeddings" ;;
        kiron-deberta) echo "kiron-deberta" ;;
        kiron-prism) echo "kiron-prism" ;;
        kitt-worker) echo "kitt-worker" ;;
        *) echo "FEHLER: unbekannter Service $1" >&2; return 1 ;;
    esac
}

apply_readonly_tree_permissions() {
    local path="$1"
    local group="$2"

    if [ ! -d "$path" ]; then
        echo "FEHLER: Rechte-Ziel fehlt oder ist kein Verzeichnis: $path" >&2
        return 1
    fi
    chown -hR root:"$group" "$path"
    find "$path" -type d -exec chmod 0750 {} +
    find "$path" -type f -perm /111 -exec chmod 0750 {} +
    find "$path" -type f ! -perm /111 -exec chmod 0640 {} +
}

apply_code_permissions() {
    local svc group

    apply_readonly_tree_permissions "$DST/services/kiron-common" kiron-common
    for svc in "${KIRON_SERVICES[@]}"; do
        group="$(service_group "$svc")"
        apply_readonly_tree_permissions "$DST/services/$svc" "$group"
    done
    # The proxy binds its Prism adapter revision to these controller sources.
    # Both services already share the read-only controller access group.
    chgrp kiron-prism-control "$DST/services/kiron-prism"
    local name
    for name in main.py composition.py controller.py process.py admission.py; do
        chgrp kiron-prism-control "$DST/services/kiron-prism/$name"
    done
}

check_local_model_registry_paths() {
    local file

    for file in "$MODEL_REGISTRY_FILE" "$MODEL_REGISTRY_LOCK_FILE"; do
        if [ -L "$file" ]; then
            echo "FEHLER: $file darf kein Symlink sein." >&2
            return 1
        fi
        if [ ! -e "$file" ]; then
            continue
        fi
        if [ ! -f "$file" ]; then
            echo "FEHLER: $file muss eine regulaere Datei sein." >&2
            return 1
        fi
        if [ "$(stat -c '%h' "$file")" != "1" ]; then
            echo "FEHLER: $file darf kein Hardlink sein." >&2
            return 1
        fi
    done
}

apply_local_model_registry_permissions() {
    local file

    check_local_model_registry_paths || return 1
    for file in "$MODEL_REGISTRY_FILE" "$MODEL_REGISTRY_LOCK_FILE"; do
        if [ ! -e "$file" ]; then
            continue
        fi
        chown kiron-proxy:kiron-common "$file"
        chmod 0640 "$file"
        if [ "$(stat -c '%U:%G %a' "$file")" != "kiron-proxy:kiron-common 640" ]; then
            echo "FEHLER: $file muss kiron-proxy:kiron-common 0640 sein." >&2
            return 1
        fi
    done
}

apply_data_permissions() {
    mkdir -p "$DST/data" "$PROXY_DATA_DIR" "$SHARED_DATA_DIR"

    chown root:root "$DST/data"
    chmod 0755 "$DST/data"

    chown kiron-proxy:kiron-proxy "$PROXY_DATA_DIR"
    chmod 0750 "$PROXY_DATA_DIR"

    chown kiron-proxy:kiron-common "$SHARED_DATA_DIR"
    chmod 2750 "$SHARED_DATA_DIR"
    apply_local_model_registry_permissions

    if [ -f "$DST/data/db_config.json" ]; then
        chown root:kiron-proxy "$DST/data/db_config.json"
        chmod 0640 "$DST/data/db_config.json"
    fi
    if [ -f "$DST/data/ollama_compat_runtime.json" ]; then
        chown root:root "$DST/data/ollama_compat_runtime.json"
        chmod 0644 "$DST/data/ollama_compat_runtime.json"
    fi
    if [ -f "$SHARED_DATA_DIR/runtime_config.json" ]; then
        chown kiron-proxy:kiron-config "$SHARED_DATA_DIR/runtime_config.json"
        chmod 0640 "$SHARED_DATA_DIR/runtime_config.json"
    fi
    for file in \
        "$PROXY_DATA_DIR"/metrics.db \
        "$PROXY_DATA_DIR"/metrics.db-wal \
        "$PROXY_DATA_DIR"/metrics.db-shm \
        "$PROXY_DATA_DIR"/maintenance_mode.json \
        "$PROXY_DATA_DIR"/selftest_results.json \
        "$PROXY_DATA_DIR"/registry_cache.json \
        "$PROXY_DATA_DIR"/benchmarks_cache.json; do
        if [ -e "$file" ]; then
            chown kiron-proxy:kiron-proxy "$file"
            chmod 0640 "$file"
        fi
    done

    if [ -d "$DST/data/kitt-worker" ]; then
        chown kitt-worker:kitt-worker "$DST/data/kitt-worker"
        chmod 0750 "$DST/data/kitt-worker"
    fi
    for path in "$DST/data/kitt-worker/staging" "$DST/data/kitt-worker/work"; do
        if [ -d "$path" ]; then
            chown kitt-worker:kitt-worker "$path"
            chmod 0750 "$path"
        fi
    done
}

check_catalog_consistency_after_restart() {
    local proxy_python="$DST/services/kiron-proxy/venv/bin/python"
    local checker="$DST/services/kiron-proxy/catalog_health.py"

    if [ ! -x "$proxy_python" ] || [ ! -f "$checker" ]; then
        echo "FEHLER: Catalog-Konsistenz-Checker oder Proxy-Python fehlt." >&2
        return 1
    fi
    echo "Pruefe Catalog-Digests der laufenden verwalteten Services..."
    runuser -u kiron-proxy -- env PYTHONNOUSERSITE=1 \
        "$proxy_python" "$checker" --wait-seconds 30
}

required_identity_groups() {
    case "$1" in
        kiron-proxy) echo "docker kiron-runtime kiron-common kiron-config kiron-prism-control" ;;
        kiron-docling) echo "docker kiron-runtime kiron-common" ;;
        kiron-embeddings) echo "kiron-models kiron-config kiron-common kiron-runtime video render" ;;
        kiron-deberta) echo "kiron-models kiron-common video render" ;;
        kiron-prism) echo "kiron-common kiron-config kiron-runtime kiron-prism-control video render" ;;
        kitt-worker) echo "kiron-runtime" ;;
        *) echo "FEHLER: unbekannter Service $1" >&2; return 1 ;;
    esac
}

check_exact_identity_groups() {
    local svc actual expected supplementary

    for svc in "${KIRON_SERVICES[@]}"; do
        supplementary="$(required_identity_groups "$svc")" || return 1
        actual="$(id -nG "$svc" | tr ' ' '\n' | LC_ALL=C sort | xargs)" \
            || return 1
        # shellcheck disable=SC2086 # fixed, space-separated group contract
        expected="$(printf '%s\n' "$svc" $supplementary | LC_ALL=C sort | xargs)"
        if [ "$actual" != "$expected" ]; then
            echo "FEHLER: $svc Gruppen falsch: ist '$actual', soll '$expected'." >&2
            echo "       bash scripts/install-system-configs.sh zuerst ausfuehren." >&2
            return 1
        fi
    done
}

validate_model_catalog_source() {
    echo "Validiere eingebauten Model-Catalog (offline)..."
    env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
        python3 "$SRC/scripts/validate-model-catalog.py" || {
            echo "FEHLER: Model-Catalog-/Projektionsvalidierung fehlgeschlagen; Deploy bleibt unveraendert." >&2
            return 1
        }
}

check_issue_859_deploy_prereqs() {
    local svc group

    for group in \
        kiron-proxy kiron-docling kiron-embeddings kiron-deberta kiron-prism kiron-prism-control \
        kiron-models kiron-runtime kiron-config kiron-common \
        docker video render kitt-worker; do
        if ! getent group "$group" >/dev/null 2>&1; then
            echo "FEHLER: Gruppe $group fehlt; install-system-configs.sh zuerst ausfuehren." >&2
            return 1
        fi
    done

    for svc in "${KIRON_SERVICES[@]}"; do
        if ! id -u "$svc" >/dev/null 2>&1; then
            echo "FEHLER: User $svc fehlt; install-system-configs.sh zuerst ausfuehren." >&2
            return 1
        fi
    done

    check_exact_identity_groups || return 1
    check_kiron_dashboard_env_contract || return 1
    check_local_model_registry_paths || return 1
}

# Preflight: Quellen pruefen, bevor irgendetwas veraendert wird.
# Vermeidet die haeufigste Ursache fuer einen mittendrin abgebrochenen Deploy.
for svc in "${KIRON_SERVICES[@]}"; do
    if [ ! -d "$SRC/services/$svc" ]; then
        echo "FEHLER: Quelle $SRC/services/$svc fehlt. Deploy abgebrochen." >&2
        exit 1
    fi
    if [ ! -f "$SRC/services/$svc/requirements.txt" ]; then
        echo "FEHLER: Quelle $SRC/services/$svc/requirements.txt fehlt. Deploy abgebrochen." >&2
        exit 1
    fi
done
if [ ! -f "$SRC/services/kiron-common/pyproject.toml" ] || [ ! -d "$SRC/services/kiron-common/kiron_common" ]; then
    echo "FEHLER: Quelle $SRC/services/kiron-common fehlt. Deploy abgebrochen." >&2
    exit 1
fi
if [ ! -f "$SRC/scripts/validate-model-catalog.py" ]; then
    echo "FEHLER: Model-Catalog-Validator fehlt: $SRC/scripts/validate-model-catalog.py" >&2
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

# Architektur-Gate vor docker pull, Verzeichnis-/Rechteaenderungen, rsync,
# Compose-/Handoff-/Unit-Aenderungen, Restarts und version.txt.
validate_model_catalog_source

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

install_compat_report() {
    if [ "$MATCHING_REPORT" = "override" ]; then
        printf '%s\n' override
        return
    fi
    # Freeze and revalidate exact bytes using the existing offline gate. Runtime
    # consumers never need access to the development repository.
    python3 - "$SRC/scripts/check-ollama-compat.py" "$MATCHING_REPORT" \
        "$NEW_OLLAMA_IMAGE" "$TARGET_DIGEST" "$MATCHING_REPORT_NUM_GPU" \
        "$DST/data/ollama_compat_reports" <<'PY'
import grp
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
import tempfile

validator_path, source, image, digest, expected_safe, destination = sys.argv[1:]
spec = importlib.util.spec_from_file_location("kiron_compat_gate", validator_path)
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)
descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
with os.fdopen(descriptor, "rb") as stream:
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
        raise SystemExit("invalid or oversized compat report")
    payload = stream.read(1024 * 1024 + 1)
    if len(payload) != info.st_size:
        raise SystemExit("compat report changed during read")
with tempfile.TemporaryDirectory(prefix="kiron-compat-gate-") as scratch:
    snapshot = Path(scratch) / "snapshot.json"
    snapshot.write_bytes(payload)
    match = validator.find_matching_report(Path(scratch), image, digest)
    if match is None or match[1] != (expected_safe == "true"):
        raise SystemExit("compat report differs from the accepted gate result")
checksum = hashlib.sha256(payload).hexdigest()
directory = Path(destination)
directory.mkdir(mode=0o750, parents=True, exist_ok=True)
directory_info = directory.lstat()
if not stat.S_ISDIR(directory_info.st_mode) or directory_info.st_uid != 0 or directory_info.st_mode & 0o022:
    raise SystemExit("unsafe runtime report directory")
gid = grp.getgrnam("kiron-common").gr_gid
os.chown(directory, 0, gid)
directory.chmod(0o750)
target = directory / ("sha256-" + checksum + ".json")
descriptor, temporary = tempfile.mkstemp(prefix=".report-", dir=directory)
try:
    with os.fdopen(descriptor, "wb") as stream:
        os.fchown(stream.fileno(), 0, gid)
        os.fchmod(stream.fileno(), 0o640)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
finally:
    Path(temporary).unlink(missing_ok=True)
print(target)
PY
}

write_runtime_handoff() {
    local digest="$1"
    local report_path="$2"
    local effective="$3"
    mkdir -p "$(dirname "$RUNTIME_HANDOFF")"
    python3 - "$RUNTIME_HANDOFF" "$digest" "$report_path" "$effective" <<'PY'
import hashlib, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
report_path = sys.argv[3]
payload = {
    "image_digest": sys.argv[2],
    "report_path": report_path,
    "report_sha256": (hashlib.sha256(pathlib.Path(report_path).read_bytes()).hexdigest()
                      if report_path != "override" else None),
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
    # Alle Services pruefen, da der spaetere systemctl-restart-Loop alle
    # abdeckt — fehlt ein venv, soll der Abbruch vor mutierenden Schritten
    # passieren statt erst nach rsync und Docker-Handoff.
    for svc in "${KIRON_SERVICES[@]}"; do
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

check_kitt_common_snapshot() {
    # Private package copy: reject stale code before a restart/deploy mutation.
    "$DST/services/kitt-worker/venv/bin/python" - "$1" <<'PY'
import hashlib
from pathlib import Path
import sys
import kiron_common
def inventory(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".json"}}
if inventory(Path(sys.argv[1])) != inventory(Path(kiron_common.__file__).parent):
    raise SystemExit("KITT common package differs: run scripts/setup-venvs.sh before restart")
PY
}

smoke_common_imports() {
    # MUSS nach rsync von kiron-common laufen, damit der aktuelle Code geprueft wird.
    # Fuer deklarierte Common-Consumer ist ein fehlender Import ein harter
    # Venv-Vertragsfehler; nur noch nicht migrierte Services stehen nicht in
    # COMMON_CONSUMERS.
    for svc in "${COMMON_CONSUMERS[@]}"; do
        py="$DST/services/$svc/venv/bin/python"
        if ! "$py" -c 'import kiron_common' 2>/dev/null; then
            echo "FEHLER: kiron_common fehlt im $svc venv." >&2
            echo "Fuehre die sichere Sequenz aus:" >&2
            echo "  bash /opt/kiron/scripts/deploy-local.sh" >&2
            echo "  bash /opt/kiron/scripts/setup-venvs.sh" >&2
            echo "  bash /opt/kiron/scripts/deploy-local.sh --restart" >&2
            return 1
        fi
        "$py" -c 'import kiron_common.catalog_consistency, kiron_common.model_state, kiron_common.ollama_compat' || {
            echo "FEHLER: aktuelle kiron_common Module im $svc venv nicht importierbar." >&2
            echo "Fuehre die sichere Sequenz aus:" >&2
            echo "  bash /opt/kiron/scripts/deploy-local.sh" >&2
            echo "  bash /opt/kiron/scripts/setup-venvs.sh" >&2
            echo "  bash /opt/kiron/scripts/deploy-local.sh --restart" >&2
            return 1
        }
        if [ "$svc" = "kitt-worker" ]; then
            check_kitt_common_snapshot "$DST/services/kiron-common/kiron_common" || return 1
        fi
        if [ "$svc" = "kiron-proxy" ] || \
           [ "$svc" = "kiron-embeddings" ] || \
           [ "$svc" = "kiron-deberta" ]; then
            "$py" -c 'from importlib import resources; from kiron_common.embedding_registry import MODEL_CATALOG; root = resources.files("kiron_common.model_catalog.manifests"); assert any(item.name.endswith(".model.json") for item in root.iterdir()); assert MODEL_CATALOG.groups' || {
                echo "FEHLER: Model-Catalog-Package-Daten im $svc venv fehlen." >&2
                return 1
            }
        fi
    done
}

check_prism_restart_prereqs() {
    local py="$DST/services/kiron-prism/venv/bin/python"
    [ -x "$py" ] || { echo "FEHLER: kiron-prism venv fehlt; setup-venvs.sh zuerst ausfuehren." >&2; return 1; }
    # Verify as the actual consumer; no sockets, subprocesses or model loads.
    runuser -u kiron-prism -- env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
        "$py" -c 'from pathlib import Path; from kiron_common.prism_runtime_policy import Policy; from kiron_common.local_model_registry import RuntimeModelRegistry; Policy.load(Path("/usr/lib/kiron/data/prism-runtime-policy.json")); RuntimeModelRegistry(readonly=True).list()' || {
        echo "FEHLER: Prism Policy/Binarybundle/Registry nicht startbereit; Deploy bleibt unveraendert." >&2
        return 1
    }
}

check_kitt_worker_restart_prereqs() {
    if ! getent group kitt-worker >/dev/null 2>&1; then
        echo "FEHLER: Gruppe kitt-worker fehlt; install-system-configs.sh zuerst ausfuehren." >&2
        return 1
    fi
    if ! id -u kitt-worker >/dev/null 2>&1; then
        echo "FEHLER: User kitt-worker fehlt; install-system-configs.sh zuerst ausfuehren." >&2
        return 1
    fi
    for path in "$DST/data/kitt-worker" "/run/kiron/kitt-worker"; do
        if [ ! -d "$path" ]; then
            echo "FEHLER: Runtime-Pfad $path fehlt; install-system-configs.sh zuerst ausfuehren." >&2
            return 1
        fi
        if [ "$(stat -c '%U:%G %a' "$path")" != "kitt-worker:kitt-worker 750" ]; then
            echo "FEHLER: Runtime-Pfad $path muss kitt-worker:kitt-worker 0750 sein." >&2
            return 1
        fi
    done
}

kitt_worker_effective_env_value() {
    local key="$1"
    local env_line part
    env_line="$(systemctl show kitt-worker.service -p Environment --value 2>/dev/null || true)"
    for part in $env_line; do
        case "$part" in
            "$key="*)
                printf '%s\n' "${part#*=}"
                return 0
                ;;
        esac
    done
    return 1
}

check_not_symlink() {
    local path="$1"
    if [ -L "$path" ]; then
        echo "FEHLER: $path darf kein Symlink sein." >&2
        return 1
    fi
}

check_kiron_dashboard_env_contract() {
    local dashboard_env="/etc/kiron/dashboard.env"

    check_not_symlink /etc/kiron || return 1
    check_not_symlink "$dashboard_env" || return 1
    if [ ! -e "$dashboard_env" ]; then
        return 0
    fi
    [ -d /etc/kiron ] \
        || { echo "FEHLER: /etc/kiron fehlt." >&2; return 1; }
    [ "$(stat -c '%U:%G %a' /etc/kiron)" = "root:root 755" ] \
        || { echo "FEHLER: /etc/kiron muss root:root 0755 sein." >&2; return 1; }
    [ -f "$dashboard_env" ] \
        || { echo "FEHLER: /etc/kiron/dashboard.env muss eine regulaere Datei sein." >&2; return 1; }
    [ "$(stat -c '%U:%G %a' "$dashboard_env")" = "root:root 600" ] \
        || { echo "FEHLER: /etc/kiron/dashboard.env muss root:root 0600 sein." >&2; return 1; }
}

validate_kitt_worker_credentials_path() {
    local credentials_file="$1"
    if ! python3 - "$credentials_file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
credential_dir = Path("/etc/kiron/kitt-worker")
if not path.is_absolute():
    raise SystemExit(1)
if path.parent != credential_dir:
    raise SystemExit(1)

resolved = path.resolve(strict=False)
root = credential_dir.resolve(strict=False)
if resolved != root and not resolved.is_relative_to(root):
    raise SystemExit(1)
PY
    then
        echo "FEHLER: KITT_WORKER_CREDENTIALS_FILE muss direkt unter /etc/kiron/kitt-worker liegen." >&2
        return 1
    fi
}

check_kitt_worker_credentials_file() {
    local credentials_file="$1"
    validate_kitt_worker_credentials_path "$credentials_file" || return 1
    check_not_symlink /etc/kiron || return 1
    check_not_symlink /etc/kiron/kitt-worker || return 1
    check_not_symlink "$credentials_file" || return 1
    [ -d /etc/kiron ] || { echo "FEHLER: /etc/kiron fehlt." >&2; return 1; }
    [ -d /etc/kiron/kitt-worker ] || { echo "FEHLER: /etc/kiron/kitt-worker fehlt." >&2; return 1; }
    [ -f "$credentials_file" ] || { echo "FEHLER: Credential-Datei fehlt." >&2; return 1; }
    [ "$(stat -c '%U:%G %a' /etc/kiron)" = "root:root 755" ] \
        || { echo "FEHLER: /etc/kiron muss root:root 0755 sein." >&2; return 1; }
    [ "$(stat -c '%U:%G %a' /etc/kiron/kitt-worker)" = "root:kitt-worker 750" ] \
        || { echo "FEHLER: /etc/kiron/kitt-worker muss root:kitt-worker 0750 sein." >&2; return 1; }
    [ "$(stat -c '%U:%G %a' "$credentials_file")" = "root:kitt-worker 640" ] \
        || { echo "FEHLER: Credential-Datei muss root:kitt-worker 0640 sein." >&2; return 1; }
}

check_kitt_worker_credentials_store_loadable() {
    local credentials_file="$1"
    local py="$DST/services/kitt-worker/venv/bin/python"
    if ! command -v runuser >/dev/null 2>&1; then
        echo "FEHLER: runuser fehlt; CredentialStore-Preflight muss als kitt-worker laufen." >&2
        return 1
    fi
    if [ ! -x "$py" ]; then
        echo "FEHLER: $py fehlt; CredentialStore-Preflight benoetigt vorhandene kitt-worker venv." >&2
        return 1
    fi
    if ! (cd "$DST/services/kitt-worker" && runuser -u kitt-worker -- "$py" - "$credentials_file" <<'PY'
from pathlib import Path
import sys

import auth

auth.load_credentials_file(Path(sys.argv[1]))
PY
    ); then
        echo "FEHLER: kitt-worker CredentialStore ist im Auth-required-Normalbetrieb nicht ladbar." >&2
        return 1
    fi
}

check_kitt_worker_no_dispatch_activation() {
    local scan_paths
    scan_paths=("$@")
    if [ "${#scan_paths[@]}" -eq 0 ]; then
        scan_paths=(
            "$DST/services/kitt-worker"
            "/etc/systemd/system/kitt-worker.service"
            "/etc/systemd/system/kitt-worker.service.d"
        )
    fi
    if ! python3 - "${scan_paths[@]}" <<'PY'
from pathlib import Path
import re
import sys

pattern = re.compile(
    r"""(?ix)
    (?<![A-Za-z0-9_])
    ["']?(?:[A-Za-z0-9_]*_)?allow_dispatch["']?
    \s*(?:=|:)\s*
    ["']?(?:true|1|yes|on)["']?
    (?![A-Za-z0-9_])
    """
)
suffixes = {".py", ".json", ".yaml", ".yml", ".service", ".sh", ".conf"}


def should_scan(path: Path) -> bool:
    return path.name == ".env" or path.suffix in suffixes


def iter_files(path: Path):
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file() and should_scan(child):
                yield child
    elif path.is_file():
        yield path


for raw in sys.argv[1:]:
    for path in iter_files(Path(raw)):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            print(f"FEHLER: Dispatch-Gate konnte {path} nicht lesen.", file=sys.stderr)
            sys.exit(1)
        if pattern.search(text):
            print(f"FEHLER: kitt-worker darf allow_dispatch nicht truthy setzen: {path}", file=sys.stderr)
            sys.exit(1)
PY
    then
        return 1
    fi
}

check_kitt_worker_auth_restart_preflight() {
    local enable_v1 auth_mode credentials_file
    check_kitt_worker_no_dispatch_activation || return 1
    enable_v1="$(kitt_worker_effective_env_value KITT_WORKER_ENABLE_V1 || true)"
    if [ -z "$enable_v1" ]; then
        enable_v1="true"
    fi
    case "$enable_v1" in
        true|1|yes|on) ;;
        false|0|no|off)
            echo "  kitt-worker /v1 disabled: Rollback-Modus, CredentialStore wird nicht verlangt"
            return 0
            ;;
        *)
            echo "FEHLER: KITT_WORKER_ENABLE_V1 ist ungueltig: $enable_v1" >&2
            return 1
            ;;
    esac
    auth_mode="$(kitt_worker_effective_env_value KITT_WORKER_AUTH_MODE || true)"
    if [ -z "$auth_mode" ]; then
        auth_mode="required"
    fi
    case "$auth_mode" in
        required)
            credentials_file="$(kitt_worker_effective_env_value KITT_WORKER_CREDENTIALS_FILE || true)"
            if [ -z "$credentials_file" ]; then
                echo "FEHLER: KITT_WORKER_CREDENTIALS_FILE fehlt in der effektiven systemd-Umgebung." >&2
                return 1
            fi
            check_kitt_worker_credentials_file "$credentials_file" || return 1
            check_kitt_worker_credentials_store_loadable "$credentials_file"
            ;;
        disabled)
            echo "  kitt-worker Auth disabled: /v1 Rollback-Modus, CredentialStore wird nicht verlangt"
            ;;
        *)
            echo "FEHLER: KITT_WORKER_AUTH_MODE ist ungueltig: $auth_mode" >&2
            return 1
            ;;
    esac
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

check_kitt_worker_no_dispatch_activation "$SRC/services/kitt-worker" "$SRC/scripts" "$SRC/systemd"
check_issue_859_deploy_prereqs
python3 "$SRC/scripts/check-ollama-admission.py"

if [ "${1:-}" = "--restart" ]; then
    check_common_venvs
    check_kitt_common_snapshot "$SRC/services/kiron-common/kiron_common"
    check_kitt_worker_restart_prereqs
    check_prism_restart_prereqs
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
mkdir -p "$DST/services/kiron-common" "$DST/data" "$DST/docker"
for svc in "${KIRON_SERVICES[@]}"; do
    mkdir -p "$DST/services/$svc"
done

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

for svc in "${KIRON_SERVICES[@]}"; do
    sync_service "$svc" --exclude='requirements-lock.txt'
done
rsync -a --delete \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    "$SRC/services/kiron-common/" "$DST/services/kiron-common/"
apply_code_permissions

check_kitt_worker_no_dispatch_activation "$DST/services/kitt-worker"

# Importprobe gegen den jetzt aktuellen Common-Code, bevor Docker-Compose,
# Runtime-Handoff und systemd-Aenderungen Side-Effects haben (#782).
if [ "${1:-}" = "--restart" ]; then
    smoke_common_imports
fi

# Docker-Compose
MATCHING_REPORT="$(install_compat_report)"
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
apply_data_permissions

# systemd-Units installieren
cp "$SRC/systemd/"*.service /etc/systemd/system/
check_kitt_worker_no_dispatch_activation \
    "$DST/services/kitt-worker" \
    "/etc/systemd/system/kitt-worker.service" \
    "/etc/systemd/system/kitt-worker.service.d"
systemctl daemon-reload
echo "  systemd-Units aktualisiert"
if [ "$1" = "--restart" ]; then
    check_kitt_worker_auth_restart_preflight
fi

# Optional: Services neustarten
# Sequentiell, damit Services nicht gleichzeitig um GPU/VRAM konkurrieren (#545).
# Proxy zuerst, damit das Dashboard/OpenAI-API so frueh wie moeglich wieder antwortet.
if [ "$1" = "--restart" ]; then
    echo "Starte Services neu..."
    # Per-Service-Fehler tolerieren statt fail-fast: sonst bleiben die uebrigen
    # Services mit altem in-memory-Code stehen, waehrend on-disk schon der neue
    # Code liegt — Halb-Deploy-Zustand (#936).
    failed_services=()
    for svc in "${KIRON_SERVICES[@]}"; do
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
    if ! check_catalog_consistency_after_restart; then
        echo "FEHLER: Catalog-Digests nach Restart inkonsistent; Deploy-Complete-Marker bleibt unveraendert." >&2
        abort_partial
    fi
    echo "Services neugestartet."
fi

# version.txt als allerletzter Schritt: fungiert als Deploy-Complete-Marker.
# Bei Abbruch vorher bleibt die alte Version sichtbar und signalisiert Inkonsistenz.
cp "$SRC/version.txt" "$DST/version.txt"

trap - ERR

echo "Deploy abgeschlossen: $(cat "$DST/version.txt")"
