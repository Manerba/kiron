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
SERVICES=(kiron-proxy kiron-docling kiron-embeddings kiron-deberta)
COMMON_SRC="$DST/services/kiron-common"

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
    req="$SRC/services/$svc/requirements.txt"
    venv_new="$DST/services/$svc/venv.new"
    if [ -f "$req" ]; then
        echo "Baue $svc-venv in $venv_new..."
        rm -rf "$venv_new"
        python3 -m venv "$venv_new"
        "$venv_new/bin/pip" install --upgrade pip -q
        "$venv_new/bin/pip" install -r "$req"
        if [ "$svc" = "kiron-proxy" ] || [ "$svc" = "kiron-docling" ]; then
            "$venv_new/bin/pip" install -e "$COMMON_SRC"
            "$venv_new/bin/python" -c 'import kiron_common.ollama_compat'
        fi
        echo "  $svc: build ok"
    fi
done

# #886 Phase 2: Smoke-Tests gegen die neuen venvs vor dem Swap.
for svc in kiron-embeddings kiron-deberta; do
    venv_new="$DST/services/$svc/venv.new"
    if [ -x "$venv_new/bin/python" ] && [ -f "$DST/services/$svc/main.py" ]; then
        (cd "$DST/services/$svc" && "$venv_new/bin/python" -c 'import main') || {
            echo "FEHLER: Import-Smoke fuer $svc fehlgeschlagen." >&2
            exit 1
        }
    fi
done

# #886 Phase 3: atomarer Swap. Ab hier ist Produktion betroffen
# (VENV_TOUCHED=1 → Trap startet bei Fehler keine Services mit halben venvs).
# mv ist auf gleichem Filesystem atomar pro Verzeichnis. venv-Symlinks sind
# pfad-unabhaengig (bin/python -> python3 -> /usr/bin/python3); pip-Shebangs
# in $venv/bin/* zeigen nach dem mv ins Leere, das ist ok — Services nutzen
# bin/python direkt, pip wird erst im naechsten setup-venvs-Lauf gebraucht.
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

SETUP_OK=1
echo "Alle venvs erstellt."
