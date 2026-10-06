#!/bin/bash
# Erstellt Python-venvs in der Produktiv-Umgebung (/usr/lib/kiron).
# Requirements kommen aus dem Git-Repo (/opt/kiron).
#
# #459: Stoppt laufende Services vor venv-Rebuild, startet sie am Ende wieder.
# Trap sorgt dafuer, dass Services auch bei Script-Fehler oder Ctrl+C wieder
# gestartet werden (cleanup-Garantie).
# #886: venvs werden in $venv_dir.new gebaut und erst nach erfolgreichem Build
# + Smoke-Tests atomar via mv in $venv_dir geswappt. Vorher wurde --clear
# direkt auf den Produktionspfad ausgefuehrt; ein pip-Fehler hinterliess
# halbe venvs und Services blieben unten. venv-Symlinks (bin/python ->
# python3 -> /usr/bin/python3) sind pfad-unabhaengig, daher ist mv sicher.
set -e

SRC="/opt/kiron"
DST="/usr/lib/kiron"
SERVICES=(kiron-proxy kiron-docling kiron-embeddings kiron-deberta kiron-prism kitt-worker)
COMMON_SRC="$DST/services/kiron-common"
HF_HOME="/var/cache/kiron/huggingface"
HF_HUB_CACHE="$HF_HOME/hub"
ST_HOME="$HF_HOME/sentence-transformers"
XDG_CACHE_HOME="/var/cache/kiron"
REGISTRY_CLI_ENTRYPOINT_SOURCE="$SRC/scripts/kiron-model-registry-entrypoint.sh"

# Optional, closed per-service package pins for an operator-prepared wheelhouse.
# Validate every file before even observing/stopping services. pip inherits the
# selected constraint only inside that service's build subshell below.
validate_venv_constraints() {
    if [ -z "${KIRON_VENV_CONSTRAINTS_DIR:-}" ]; then
        return 0
    fi
    python3 - "$KIRON_VENV_CONSTRAINTS_DIR" "${SERVICES[@]}" <<'PY'
from pathlib import Path
import re
import stat
import sys

root = Path(sys.argv[1])
if not root.is_absolute() or root.resolve() != root or not root.is_dir():
    raise SystemExit("FEHLER: KIRON_VENV_CONSTRAINTS_DIR muss ein absolutes Verzeichnis ohne Symlinks sein")
expected = {name + ".txt" for name in sys.argv[2:]}
if {path.name for path in root.iterdir()} != expected:
    raise SystemExit("FEHLER: Constraints-Verzeichnis muss genau die sechs Dienstdateien enthalten")
version = r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?"
pattern = re.compile(r"([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)==(" + version + r")")
for name in sorted(expected):
    path = root / name
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
        raise SystemExit("FEHLER: ungueltige Constraints-Datei: " + name)
    pins = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise SystemExit("FEHLER: Constraints erlauben nur package==version: " + name)
        package = re.sub(r"[-_.]+", "-", match[1]).lower()
        if package in pins:
            raise SystemExit("FEHLER: doppelter Constraints-Paketname: " + name)
        pins.add(package)
    if not {"pip", "setuptools", "wheel"} <= pins:
        raise SystemExit("FEHLER: Constraints muessen pip, setuptools und wheel pinnen: " + name)
PY
}

validate_venv_constraints

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

service_import_module() {
    case "$1" in
        kiron-proxy) echo "main" ;;
        kiron-docling) echo "proxy" ;;
        kiron-embeddings) echo "main" ;;
        kiron-deberta) echo "main" ;;
        kiron-prism) echo "main" ;;
        *) echo "FEHLER: kein Import-Smoke fuer $1 definiert" >&2; return 1 ;;
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

apply_all_service_tree_permissions() {
    local svc group

    apply_readonly_tree_permissions "$COMMON_SRC" kiron-common
    for svc in "${SERVICES[@]}"; do
        group="$(service_group "$svc")"
        apply_readonly_tree_permissions "$DST/services/$svc" "$group"
    done
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

    for svc in "${SERVICES[@]}"; do
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
            echo "FEHLER: Model-Catalog-/Projektionsvalidierung fehlgeschlagen; Services und venvs bleiben unveraendert." >&2
            return 1
        }
}

check_issue_859_prereqs() {
    local svc group path

    if ! command -v runuser >/dev/null 2>&1; then
        echo "FEHLER: runuser fehlt; Import-Smokes muessen als Service-User laufen." >&2
        return 1
    fi

    for group in \
        kiron-proxy kiron-docling kiron-embeddings kiron-deberta kiron-prism kiron-prism-control \
        kiron-models kiron-runtime kiron-config kiron-common docker video render; do
        if ! getent group "$group" >/dev/null 2>&1; then
            echo "FEHLER: Gruppe $group fehlt; install-system-configs.sh zuerst ausfuehren." >&2
            return 1
        fi
    done

    for svc in "${SERVICES[@]}"; do
        if ! id -u "$svc" >/dev/null 2>&1; then
            echo "FEHLER: User $svc fehlt; install-system-configs.sh zuerst ausfuehren." >&2
            return 1
        fi
    done

    check_exact_identity_groups || return 1

    for path in \
        "/run/kiron/vram" \
        "/run/kiron/prism" \
        "$DST/data/kiron-proxy" \
        "$DST/data/shared" \
        "$HF_HOME" \
        "$HF_HUB_CACHE" \
        "$ST_HOME" \
        "$HF_HOME/modules" \
        "$HF_HOME/xet"; do
        if [ ! -d "$path" ]; then
            echo "FEHLER: Runtime-/Cache-Pfad $path fehlt; install-system-configs.sh zuerst ausfuehren." >&2
            return 1
        fi
    done
}

smoke_service_venv_new() {
    local svc="$1"
    local user="$1"
    local module smoke_code venv_new

    module="$(service_import_module "$svc")"
    case "$svc" in
        kiron-proxy)
            smoke_code='from importlib import resources; from kiron_common.catalog_consistency import check_catalog_digests; from kiron_common.embedding_registry import MODEL_CATALOG, MODEL_STATE_VIEW; manifest_root = resources.files("kiron_common.model_catalog.manifests"); assert any(item.name.endswith(".model.json") for item in manifest_root.iterdir()); assert MODEL_STATE_VIEW.catalog is MODEL_CATALOG; import routing_catalog; assert routing_catalog.PROXY_ROUTING_VIEW.catalog is MODEL_CATALOG; assert routing_catalog.PROXY_ROUTING_VIEW.routes; import app, catalog_health, metrics, main; assert app.MODEL_STATE_VIEW.catalog is MODEL_CATALOG; assert metrics.PROXY_ROUTING_VIEW.catalog is MODEL_CATALOG; assert catalog_health.PROXY_ROUTING_VIEW.catalog is MODEL_CATALOG; assert check_catalog_digests(MODEL_CATALOG.catalog_digest, {"proxy": MODEL_CATALOG.catalog_digest}, required_services=("proxy",)).consistent'
            ;;
        kiron-embeddings)
            smoke_code='import json; from importlib import resources; import kiron_common; from kiron_common.embedding_registry import MODEL_CATALOG, MODEL_STATE_VIEW; manifest_root = resources.files("kiron_common.model_catalog.manifests"); assert any(item.name.endswith(".model.json") for item in manifest_root.iterdir()); assert MODEL_CATALOG.groups; assert MODEL_STATE_VIEW.catalog is MODEL_CATALOG; import main; health = json.loads(main.health().body); assert main.EMBEDDING_SERVICE_VIEW.catalog_digest == MODEL_CATALOG.catalog_digest == health["catalog_digest"]; assert health["model_states"]; assert all(type(row["installed"]) is bool for row in health["model_states"])'
            ;;
        kiron-deberta)
            smoke_code='import json; from importlib import resources; import kiron_common; from kiron_common.embedding_registry import MODEL_CATALOG, MODEL_STATE_VIEW; manifest_root = resources.files("kiron_common.model_catalog.manifests"); assert any(item.name.endswith(".model.json") for item in manifest_root.iterdir()); assert MODEL_CATALOG.groups; assert MODEL_STATE_VIEW.catalog is MODEL_CATALOG; import main; health = json.loads(main.health().body); assert main.SHARED_MODEL_CATALOG is MODEL_CATALOG; assert main.DEBERTA_CATALOG_VIEW.models; assert main.DEBERTA_CATALOG_VIEW.catalog_digest == MODEL_CATALOG.catalog_digest == health["catalog_digest"]; assert health["model_states"]; assert all(type(row["installed"]) is bool for row in health["model_states"])'
            ;;
        kiron-prism)
            smoke_code='import main, controller, composition; from kiron_common.prism_runtime_policy import Policy; from kiron_common.local_model_registry import RuntimeModelRegistry; from kiron_common.gpu_admission import AdmissionStore; assert callable(main.create_app)'
            ;;
        *)
            smoke_code="import ${module}"
            ;;
    esac
    venv_new="$DST/services/$svc/venv.new"
    if [ ! -x "$venv_new/bin/python" ]; then
        echo "FEHLER: $svc venv.new fehlt oder enthaelt keinen ausfuehrbaren Python: $venv_new/bin/python" >&2
        return 1
    fi
    echo "Smoke $svc-venv als $user..."
    runuser -u "$user" -- test -x "$venv_new/bin/python" || {
        echo "FEHLER: $svc venv-Python ist fuer $user nicht ausfuehrbar." >&2
        return 1
    }
    runuser -u "$user" -- test ! -w "$DST/services/$svc" || {
        echo "FEHLER: $user darf das Service-Verzeichnis nicht beschreiben: $DST/services/$svc" >&2
        return 1
    }
    runuser -u "$user" -- test ! -w "$venv_new" || {
        echo "FEHLER: $user darf die venv nicht beschreiben: $venv_new" >&2
        return 1
    }
    (
        cd "$DST/services/$svc" && \
        runuser -u "$user" -- env \
            KIRON_RUNTIME_DIR=/run/kiron/vram \
            HF_HOME="$HF_HOME" \
            HF_HUB_CACHE="$HF_HUB_CACHE" \
            SENTENCE_TRANSFORMERS_HOME="$ST_HOME" \
            XDG_CACHE_HOME="$XDG_CACHE_HOME" \
            "$venv_new/bin/python" -c "$smoke_code"
    ) || {
        echo "FEHLER: Import-Smoke fuer $svc als $user fehlgeschlagen." >&2
        return 1
    }
}

if [ ! -f "$COMMON_SRC/pyproject.toml" ] || [ ! -d "$COMMON_SRC/kiron_common" ]; then
    echo "FEHLER: kiron-common Quelle fehlt unter $COMMON_SRC." >&2
    echo "       Erst scripts/deploy-local.sh ohne --restart ausfuehren." >&2
    exit 1
fi

# #584: Preflight aller requirements.txt bevor Services gestoppt werden.
# Sonst wuerde ein fehlendes File silent geskippt und am Ende SETUP_OK=1 trotzdem
# laufende Services mit altem/fehlendem venv neu starten (ImportError beim Start).
for svc in "${SERVICES[@]}"; do
    req="$SRC/services/$svc/requirements.txt"
    if [ ! -f "$req" ]; then
        echo "FEHLER: requirements.txt fehlt fuer $svc: $req" >&2
        exit 1
    fi
done

if [ ! -f "$SRC/scripts/validate-model-catalog.py" ]; then
    echo "FEHLER: Model-Catalog-Validator fehlt: $SRC/scripts/validate-model-catalog.py" >&2
    exit 1
fi

if [ ! -f "$REGISTRY_CLI_ENTRYPOINT_SOURCE" ] || \
   [ -L "$REGISTRY_CLI_ENTRYPOINT_SOURCE" ]; then
    echo "FEHLER: relocatable Registry-CLI-Entrypoint fehlt oder ist ein Symlink: $REGISTRY_CLI_ENTRYPOINT_SOURCE" >&2
    exit 1
fi

# Architektur-Gate vor systemctl stop, venv-Build, Rechte- oder Swap-Aenderungen.
validate_model_catalog_source

check_kitt_worker_prereqs() {
    if ! command -v runuser >/dev/null 2>&1; then
        echo "FEHLER: runuser fehlt; kitt-worker Smoke muss als Service-User laufen." >&2
        return 1
    fi
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

check_kitt_worker_prereqs

check_issue_859_prereqs

# Merke welche Services aktiv waren, damit wir nur die restarten.
WAS_ACTIVE=()
for svc in "${SERVICES[@]}"; do
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        WAS_ACTIVE+=("$svc")
    fi
done

# #886: venv-Setup-Erfolg tracken. Build laeuft jetzt in $venv_dir.new; erst
# beim atomaren Swap wird die Produktion beruehrt. Bricht ein Build vorher ab,
# sind die alten venvs noch intakt und Services duerfen wieder starten.
SETUP_OK=0

# #583/#886: Stop-Phase-Fehler von Swap-Phase-Fehler trennen. VENV_TOUCHED=1
# erst kurz vor dem Swap — bis dahin sind die Produktions-venvs unangetastet
# und der Trap darf gestoppte Services wieder hochfahren.
STOPPED=()
VENV_TOUCHED=0

restart_services() {
    # #643: Startfehler propagieren. Bei urspruenglichem Erfolg (rc=0) muss ein
    # fehlgeschlagener systemctl start als exit 1 sichtbar werden, damit
    # Automation einen down gebliebenen Service nicht als Erfolg wertet. Fehler
    # aus der venv-Phase behalten ihren Nonzero-Status.
    local rc=$?
    local restart_failed=0
    if [ "${#WAS_ACTIVE[@]}" -eq 0 ]; then
        return
    fi
    if [ "$VENV_TOUCHED" -eq 0 ]; then
        if [ "${#STOPPED[@]}" -gt 0 ]; then
            # #642: Nur die urspruenglich aktiven Services wieder starten — STOPPED
            # kann jetzt auch Auto-Restart-Kandidaten enthalten, die vorher inaktiv
            # waren und inaktiv bleiben sollen.
            echo "Stop-Phase abgebrochen — starte urspruenglich aktive Services wieder: ${WAS_ACTIVE[*]}"
            for svc in "${WAS_ACTIVE[@]}"; do
                systemctl start "$svc" || { echo "  WARNUNG: $svc start fehlgeschlagen"; restart_failed=1; }
            done
        fi
    elif [ "$SETUP_OK" -eq 0 ]; then
        echo "FEHLER: venv-Swap unterbrochen — Services NICHT neu gestartet."
        echo "  Produktions-venvs koennten inkonsistent sein. Ursache pruefen,"
        echo "  '$0' erneut ausfuehren (rebuilds .new und swappt)."
        echo "  Betroffen: ${WAS_ACTIVE[*]}"
    else
        echo "Starte Services wieder: ${WAS_ACTIVE[*]}"
        for svc in "${WAS_ACTIVE[@]}"; do
            systemctl start "$svc" || { echo "  WARNUNG: $svc start fehlgeschlagen"; restart_failed=1; }
        done
    fi
    if [ "$restart_failed" -eq 1 ] && [ "$rc" -eq 0 ]; then
        exit 1
    fi
}
trap restart_services EXIT

# #642/#886: Alle bekannten Services stoppen — auch die, die gerade nicht
# active sind. kiron-deberta/-embeddings haben Restart=always, kiron-proxy/
# -docling on-failure; ein Service in failed/RestartSec ist nicht in
# WAS_ACTIVE (is-active liefert exit 0 nur fuer active/reloading), wuerde aber
# waehrend des Swap (mv $venv_dir → $venv_dir.old) vom systemd-Auto-Restart-
# Job wieder hochgefahren und entweder den umbenannten Pfad lesen oder Files
# aus der gerade verschwundenen venv geoeffnet halten. Stop cancelt pending
# Auto-Restart-Jobs zuverlaessig.
echo "Stoppe alle Kiron-Services..."
for svc in "${SERVICES[@]}"; do
    if ! systemctl cat "$svc" >/dev/null 2>&1; then
        # Unit nicht installiert — kann nicht auto-restarten, skip.
        continue
    fi
    if ! systemctl stop "$svc"; then
        echo "  FEHLER: $svc stop fehlgeschlagen — Abbruch (venv-Swap bei laufendem Service unsafe)."
        exit 1
    fi
    STOPPED+=("$svc")
done
# #494/#642: Verifizieren dass kein bekannter Service mehr active, activating,
# reloading oder deactivating ist (alle vier koennten venv-Files lesen).
for svc in "${SERVICES[@]}"; do
    if ! systemctl cat "$svc" >/dev/null 2>&1; then
        continue
    fi
    state=$(systemctl is-active "$svc" 2>/dev/null || true)
    case "$state" in
        active|activating|reloading|deactivating)
            echo "  FEHLER: $svc ist trotz stop noch $state — Abbruch."
            exit 1
            ;;
    esac
done

# #886 Phase 1: neue venvs in $venv_dir.new bauen — Produktions-venvs bleiben
# unberuehrt. Ein pip-Fehler fuehrt jetzt zum frueh-Exit, der Trap darf die
# Services wieder starten (VENV_TOUCHED=0), weil $venv_dir noch intakt ist.
for svc in "${SERVICES[@]}"; do
  (
    if [ -n "${KIRON_VENV_CONSTRAINTS_DIR:-}" ]; then
        export PIP_CONSTRAINT="$KIRON_VENV_CONSTRAINTS_DIR/$svc.txt"
        export PIP_BUILD_CONSTRAINT="$PIP_CONSTRAINT"
    fi
    req="$SRC/services/$svc/requirements.txt"
    venv_new="$DST/services/$svc/venv.new"
    if [ -f "$req" ]; then
        echo "Baue $svc-venv in $venv_new..."
        rm -rf "$venv_new"
        python3 -m venv "$venv_new"
        "$venv_new/bin/pip" install --upgrade pip -q
        "$venv_new/bin/pip" install -r "$req"
        if [ "$svc" = "kiron-proxy" ] || \
           [ "$svc" = "kiron-docling" ] || \
           [ "$svc" = "kiron-embeddings" ] || \
           [ "$svc" = "kiron-deberta" ] || \
           [ "$svc" = "kiron-prism" ]; then
            "$venv_new/bin/pip" install -e "$COMMON_SRC"
            "$venv_new/bin/python" -c 'import kiron_common'
            # pip erzeugt absolute Shebangs auf venv.new. Der gemeinsame
            # Root-CLI muss den atomaren Verzeichnis-Swap hingegen ueberleben.
            install -m 0750 -o root -g root "$REGISTRY_CLI_ENTRYPOINT_SOURCE" "$venv_new/bin/kiron-model-registry"
        fi
        if [ "$svc" = "kitt-worker" ]; then
            # Private installed copy: KITT needs no access to the common source tree.
            "$venv_new/bin/pip" install "$COMMON_SRC"
        fi
        echo "  $svc: build ok"
    fi
  )
done

# #886 Phase 2: Rechte-Zielmodell und Smoke-Tests gegen die neuen venvs vor dem Swap.
apply_all_service_tree_permissions
for svc in kiron-proxy kiron-docling kiron-embeddings kiron-deberta kiron-prism; do
    smoke_service_venv_new "$svc" || exit 1
done

venv_new="$DST/services/kitt-worker/venv.new"
if [ ! -x "$venv_new/bin/python" ]; then
    echo "FEHLER: kitt-worker venv.new fehlt oder enthaelt keinen ausfuehrbaren Python: $venv_new/bin/python" >&2
    exit 1
fi
for required in main.py config.py auth.py contract.py job_stubs.py queue_store.py artifact_staging.py capabilities.py gpu_policy.py runners.py sft_command.py sft_trainer.py executor.py monitoring.py v1.py healthcheck.py; do
    if [ ! -f "$DST/services/kitt-worker/$required" ]; then
        echo "FEHLER: kitt-worker Service-Datei fehlt vor Pre-Swap-Smoke: $DST/services/kitt-worker/$required" >&2
        exit 1
    fi
done
echo "Smoke kitt-worker-venv als kitt-worker..."
runuser -u kitt-worker -- test -x "$DST/services/kitt-worker/venv.new/bin/python" || {
    echo "FEHLER: kitt-worker venv-Python ist fuer kitt-worker nicht ausfuehrbar." >&2
    exit 1
}
(cd "$DST/services/kitt-worker" && runuser -u kitt-worker -- "$DST/services/kitt-worker/venv.new/bin/python" -c 'import auth, config, contract, job_stubs, queue_store, artifact_staging, capabilities, gpu_policy, runners, sft_command, sft_trainer, executor, monitoring, main, v1, healthcheck') || {
    echo "FEHLER: Import-Smoke fuer kitt-worker fehlgeschlagen." >&2
    exit 1
}
runuser -u kitt-worker -- test ! -w "$DST/services/kitt-worker" || {
    echo "FEHLER: kitt-worker darf das Service-Verzeichnis nicht beschreiben." >&2
    exit 1
}
runuser -u kitt-worker -- test ! -w "$DST/services/kitt-worker/venv.new" || {
    echo "FEHLER: kitt-worker darf die venv nicht beschreiben." >&2
    exit 1
}
runuser -u kitt-worker -- test -w "$DST/data/kitt-worker" || {
    echo "FEHLER: kitt-worker kann persistenten Runtime-Pfad nicht beschreiben." >&2
    exit 1
}
runuser -u kitt-worker -- test -w "/run/kiron/kitt-worker" || {
    echo "FEHLER: kitt-worker kann kurzlebigen Runtime-Pfad nicht beschreiben." >&2
    exit 1
}
runuser -u kitt-worker -- test -w "/run/kiron/vram" || {
    echo "FEHLER: kitt-worker kann den gemeinsamen Admission-Pfad nicht beschreiben." >&2
    exit 1
}
for required_dir in \
    "$DST/data/kitt-worker/staging" \
    "$DST/data/kitt-worker/work"; do
    if [ ! -d "$required_dir" ]; then
        echo "FEHLER: kitt-worker Runtime-Pfad fehlt: $required_dir" >&2
        exit 1
    fi
    runuser -u kitt-worker -- test -w "$required_dir" || {
        echo "FEHLER: kitt-worker kann Runtime-Pfad nicht beschreiben: $required_dir" >&2
        exit 1
    }
done
for queue_file in "$DST/data/kitt-worker"/queue*; do
    if [ ! -e "$queue_file" ]; then
        continue
    fi
    case "$(basename "$queue_file")" in
        queue.sqlite3|queue.sqlite3-wal|queue.sqlite3-shm)
            ;;
        *)
            echo "FEHLER: unerlaubte kitt-worker Queue-Datei: $queue_file" >&2
            exit 1
            ;;
    esac
done

# #886 Phase 3: atomarer Swap. Ab hier ist Produktion betroffen
# (VENV_TOUCHED=1 → Trap startet bei Fehler keine Services mit halben venvs).
# mv ist auf gleichem Filesystem atomar pro Verzeichnis. venv-Symlinks sind
# pfad-unabhaengig (bin/python -> python3 -> /usr/bin/python3); pip-Shebangs
# in $venv/bin/* zeigen nach dem mv ins Leere. Services nutzen bin/python
# direkt; der produktive kiron-model-registry-Entrypoint wird deshalb oben
# bewusst durch einen relativen Wrapper ersetzt.
echo "Swappe venvs in Produktionspfad..."
VENV_TOUCHED=1
for svc in "${SERVICES[@]}"; do
    venv_dir="$DST/services/$svc/venv"
    venv_new="$venv_dir.new"
    venv_old="$venv_dir.old"
    if [ -d "$venv_new" ]; then
        rm -rf "$venv_old"
        if [ -d "$venv_dir" ]; then
            mv "$venv_dir" "$venv_old"
        fi
        mv "$venv_new" "$venv_dir"
        rm -rf "$venv_old"
        echo "  $svc: aktiv"
    fi
done
apply_all_service_tree_permissions

SETUP_OK=1
echo "Alle venvs erstellt."
