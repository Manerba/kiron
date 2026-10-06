#!/bin/bash
set -e

echo "Installiere System-Konfigurationen..."

SERVICE_UNITS=(
    kiron-proxy.service
    kiron-docling.service
    kiron-embeddings.service
    kiron-deberta.service
    kiron-prism.service
    kitt-worker.service
)

fail() {
    echo "FEHLER: $*" >&2
    exit 1
}

sorted_words() {
    printf '%s\n' "$@" | LC_ALL=C sort | xargs
}

ensure_group() {
    local group="$1"
    if ! getent group "$group" >/dev/null 2>&1; then
        groupadd --system "$group"
        echo "  Gruppe $group angelegt"
    fi
}

lock_service_password() {
    local user="$1"
    if command -v passwd >/dev/null 2>&1; then
        passwd -l "$user" >/dev/null 2>&1 || true
    fi
}

ensure_kiron_service_identity() {
    local user="$1"
    shift
    local primary="$user"
    local group csv

    ensure_group "$primary"
    for group in "$@"; do
        ensure_group "$group"
    done

    if ! id -u "$user" >/dev/null 2>&1; then
        useradd \
            --system \
            --gid "$primary" \
            --home-dir /nonexistent \
            --no-create-home \
            --shell /usr/sbin/nologin \
            "$user"
        echo "  User $user angelegt"
    else
        usermod \
            --gid "$primary" \
            --home /nonexistent \
            --shell /usr/sbin/nologin \
            "$user"
    fi

    csv="$(IFS=,; echo "$*")"
    usermod -G "$csv" "$user"
    lock_service_password "$user"
}

verify_service_identity() {
    local user="$1"
    shift
    local passwd_entry group_entry user_gid group_gid home shell groups expected password_state

    passwd_entry="$(getent passwd "$user")" || fail "User $user fehlt"
    group_entry="$(getent group "$user")" || fail "Primaergruppe $user fehlt"
    user_gid="$(printf '%s' "$passwd_entry" | cut -d: -f4)"
    group_gid="$(printf '%s' "$group_entry" | cut -d: -f3)"
    home="$(printf '%s' "$passwd_entry" | cut -d: -f6)"
    shell="$(printf '%s' "$passwd_entry" | cut -d: -f7)"
    [ "$user_gid" = "$group_gid" ] || fail "$user muss Primaergruppe $user nutzen"
    [ "$home" = "/nonexistent" ] || fail "$user darf keinen breiten Home-Schreibbereich haben"
    case "$shell" in
        /usr/sbin/nologin|/sbin/nologin|/bin/false) ;;
        *) fail "$user muss eine Non-Login-Shell nutzen" ;;
    esac

    groups="$(id -nG "$user" | tr ' ' '\n' | LC_ALL=C sort | xargs)"
    expected="$(sorted_words "$user" "$@")"
    [ "$groups" = "$expected" ] || fail "$user Gruppen falsch: ist '$groups', soll '$expected'"

    if command -v passwd >/dev/null 2>&1; then
        password_state="$(passwd -S "$user" 2>/dev/null | awk '{print $2}')"
        case "$password_state" in
            L|LK|NP) ;;
            *) fail "$user Passwort muss gesperrt oder ungesetzt sein" ;;
        esac
    fi
}

ensure_kiron_identities() {
    ensure_kiron_service_identity kiron-proxy docker kiron-runtime kiron-common kiron-config kiron-prism-control
    ensure_kiron_service_identity kiron-docling docker kiron-runtime kiron-common
    ensure_kiron_service_identity kiron-embeddings kiron-models kiron-config kiron-common kiron-runtime video render
    ensure_kiron_service_identity kiron-deberta kiron-models kiron-common video render

    ensure_kiron_service_identity kiron-prism kiron-common kiron-config kiron-runtime kiron-prism-control video render

    verify_service_identity kiron-proxy docker kiron-runtime kiron-common kiron-config kiron-prism-control
    verify_service_identity kiron-docling docker kiron-runtime kiron-common
    verify_service_identity kiron-embeddings kiron-models kiron-config kiron-common kiron-runtime video render
    verify_service_identity kiron-deberta kiron-models kiron-common video render
    verify_service_identity kiron-prism kiron-common kiron-config kiron-runtime kiron-prism-control video render
    echo "  KIron Service-User und Gruppen installiert"
}

ensure_kitt_worker_identity() {
    if ! getent group kitt-worker >/dev/null 2>&1; then
        groupadd --system kitt-worker
        echo "  Gruppe kitt-worker angelegt"
    fi
    if ! id -u kitt-worker >/dev/null 2>&1; then
        useradd \
            --system \
            --gid kitt-worker \
            --home-dir /nonexistent \
            --no-create-home \
            --shell /usr/sbin/nologin \
            kitt-worker
        echo "  User kitt-worker angelegt"
    fi
    ensure_group kiron-runtime
    usermod -G kiron-runtime kitt-worker
    lock_service_password kitt-worker
}

verify_kitt_worker_identity() {
    local passwd_entry group_entry user_gid group_gid home shell groups password_state
    passwd_entry="$(getent passwd kitt-worker)" || fail "User kitt-worker fehlt"
    group_entry="$(getent group kitt-worker)" || fail "Gruppe kitt-worker fehlt"
    user_gid="$(printf '%s' "$passwd_entry" | cut -d: -f4)"
    group_gid="$(printf '%s' "$group_entry" | cut -d: -f3)"
    home="$(printf '%s' "$passwd_entry" | cut -d: -f6)"
    shell="$(printf '%s' "$passwd_entry" | cut -d: -f7)"
    [ "$user_gid" = "$group_gid" ] || fail "kitt-worker muss Primaergruppe kitt-worker nutzen"
    [ "$home" = "/nonexistent" ] || fail "kitt-worker darf keinen breiten Home-Schreibbereich haben"
    case "$shell" in
        /usr/sbin/nologin|/sbin/nologin|/bin/false) ;;
        *) fail "kitt-worker muss eine Non-Login-Shell nutzen" ;;
    esac
    groups="$(id -nG kitt-worker | tr ' ' '\n' | LC_ALL=C sort | xargs)"
    [ "$groups" = "kiron-runtime kitt-worker" ] || fail "kitt-worker darf nur kiron-runtime als Zusatzgruppe haben: $groups"
    if command -v passwd >/dev/null 2>&1; then
        password_state="$(passwd -S kitt-worker 2>/dev/null | awk '{print $2}')"
        case "$password_state" in
            L|LK|NP) ;;
            *) fail "kitt-worker Passwort muss gesperrt oder ungesetzt sein" ;;
        esac
    fi
}

verify_path_stat() {
    local path="$1"
    local expected="$2"
    [ -e "$path" ] || fail "Pfad $path fehlt"
    [ "$(stat -c '%U:%G %a' "$path")" = "$expected" ] \
        || fail "$path muss $expected sein"
}

verify_kitt_worker_runtime_dir() {
    verify_path_stat "$1" "kitt-worker:kitt-worker 750"
}

verify_not_symlink() {
    local path="$1"
    [ ! -L "$path" ] || fail "$path darf kein Symlink sein"
}

verify_optional_regular_file() {
    local path="$1"

    verify_not_symlink "$path"
    if [ ! -e "$path" ]; then
        return 0
    fi
    [ -f "$path" ] || fail "$path muss eine regulaere Datei sein"
    [ "$(stat -c '%h' "$path")" = "1" ] \
        || fail "$path darf kein Hardlink sein"
}

verify_optional_path_stat() {
    local path="$1"
    local expected="$2"

    verify_optional_regular_file "$path"
    if [ -e "$path" ]; then
        verify_path_stat "$path" "$expected"
    fi
}

ensure_kitt_worker_secret_dir() {
    verify_not_symlink /etc/kiron
    mkdir -p /etc/kiron
    chown root:root /etc/kiron
    chmod 0755 /etc/kiron
    [ "$(stat -c '%U:%G %a' /etc/kiron)" = "root:root 755" ] \
        || fail "/etc/kiron muss root:root 0755 sein"

    verify_not_symlink /etc/kiron/kitt-worker
    mkdir -p /etc/kiron/kitt-worker
    chown root:kitt-worker /etc/kiron/kitt-worker
    chmod 0750 /etc/kiron/kitt-worker
    [ "$(stat -c '%U:%G %a' /etc/kiron/kitt-worker)" = "root:kitt-worker 750" ] \
        || fail "/etc/kiron/kitt-worker muss root:kitt-worker 0750 sein"

    verify_not_symlink /etc/kiron/kitt-worker/credentials.json
    if [ -e /etc/kiron/kitt-worker/credentials.json ]; then
        [ -f /etc/kiron/kitt-worker/credentials.json ] \
            || fail "/etc/kiron/kitt-worker/credentials.json muss eine regulaere Datei sein"
        [ "$(stat -c '%U:%G %a' /etc/kiron/kitt-worker/credentials.json)" = "root:kitt-worker 640" ] \
            || fail "/etc/kiron/kitt-worker/credentials.json muss root:kitt-worker 0640 sein"
    fi
}

ensure_kiron_dashboard_env_contract() {
    verify_not_symlink /etc/kiron/dashboard.env
    if [ ! -e /etc/kiron/dashboard.env ]; then
        return 0
    fi
    [ -f /etc/kiron/dashboard.env ] \
        || fail "/etc/kiron/dashboard.env muss eine regulaere Datei sein"
    [ "$(stat -c '%U:%G %a' /etc/kiron/dashboard.env)" = "root:root 600" ] \
        || fail "/etc/kiron/dashboard.env muss root:root 0600 sein"
}

validate_service_unit_sources() {
    local unit

    for unit in "${SERVICE_UNITS[@]}"; do
        [ -f "/opt/kiron/systemd/$unit" ] \
            || fail "Pflicht-Serviceunit fehlt: /opt/kiron/systemd/$unit"
    done
}

install_service_unit_sources() {
    local unit

    for unit in "${SERVICE_UNITS[@]}"; do
        install -o root -g root -m 0644 \
            "/opt/kiron/systemd/$unit" "/etc/systemd/system/$unit"
        cmp -s "/opt/kiron/systemd/$unit" "/etc/systemd/system/$unit" \
            || fail "Serviceunit wurde nicht byteidentisch installiert: $unit"
    done
    systemctl daemon-reload
    echo "  KIron Serviceunits installiert und systemd neu geladen (ohne Restart)"
}

install_wrapper_sources() {
    local src_dir="/opt/kiron/system/sbin"
    local helper target

    install -d -o root -g root -m 0755 /usr/local/sbin
    for helper in kiron-service-control kiron-maintenance-firewall; do
        [ -f "$src_dir/$helper" ] || fail "Pflicht-Wrapper fehlt: $src_dir/$helper"
        target="/usr/local/sbin/$helper"
        install -o root -g root -m 0755 "$src_dir/$helper" "$target"
        [ "$(stat -c '%U:%G %a' "$target")" = "root:root 755" ] \
            || fail "$target muss root:root 0755 sein"
    done
    echo "  /usr/local/sbin KIron-Wrapper installiert"
}

install_sudoers_sources() {
    local src_dir="/opt/kiron/system/sudoers.d"
    local name src target

    command -v visudo >/dev/null 2>&1 || fail "visudo fehlt; sudoers Gate kann nicht validieren"
    install -d -o root -g root -m 0750 /etc/sudoers.d
    for name in kiron-proxy-service-control kiron-proxy-firewall; do
        src="$src_dir/$name"
        target="/etc/sudoers.d/$name"
        [ -f "$src" ] || fail "Pflicht-sudoers fehlt: $src"
        visudo -cf "$src" >/dev/null
        install -o root -g root -m 0440 "$src" "$target"
        [ "$(stat -c '%U:%G %a' "$target")" = "root:root 440" ] \
            || fail "$target muss root:root 0440 sein"
    done
    visudo -c >/dev/null
    echo "  sudoers Drop-ins installiert und mit visudo -c validiert"
}

verify_kiron_runtime_paths() {
    verify_path_stat /run/kiron "root:root 755"
    verify_path_stat /run/kiron/vram "root:kiron-runtime 2770"
    verify_path_stat /run/kiron/prism "kiron-prism:kiron-prism-control 2750"
    verify_path_stat /usr/lib/kiron/data/gguf-models "root:kiron-common 2750"
    verify_optional_path_stat /usr/lib/kiron/data/prism-runtime-policy.json "root:kiron-config 640"
    verify_path_stat /run/xtables.lock "root:root 600"
    verify_path_stat /usr/lib/kiron/data "root:root 755"
    verify_path_stat /usr/lib/kiron/data/kiron-proxy "kiron-proxy:kiron-proxy 750"
    verify_path_stat /usr/lib/kiron/data/shared "kiron-proxy:kiron-common 2750"
    verify_path_stat /usr/lib/kiron/data/local-models "root:kiron-common 2750"
    verify_optional_path_stat /usr/lib/kiron/data/shared/local-model-registry.json "kiron-proxy:kiron-common 640"
    verify_optional_path_stat /usr/lib/kiron/data/shared/local-model-registry.json.lock "kiron-proxy:kiron-common 640"
    verify_path_stat /var/cache/kiron "root:root 755"
    verify_path_stat /var/cache/kiron/huggingface "root:kiron-models 2770"
    verify_path_stat /var/cache/kiron/huggingface/hub "root:kiron-models 2770"
    verify_path_stat /var/cache/kiron/huggingface/modules "root:kiron-models 2770"
    verify_path_stat /var/cache/kiron/huggingface/sentence-transformers "root:kiron-models 2770"
    verify_path_stat /var/cache/kiron/huggingface/xet "root:kiron-models 2770"
}

initialize_local_model_registry() {
    # Create only missing v2 storage. Preserve existing contents, including a
    # corrupt/unsupported registry which must fail visibly in the common decoder.
    python3 - <<'PY'
import fcntl
import grp
import os
import pwd
import stat

uid = pwd.getpwnam("kiron-proxy").pw_uid
gid = grp.getgrnam("kiron-common").gr_gid
directory = os.open("/usr/lib/kiron/data/shared", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
name = "local-model-registry.json"

def open_file(name, *, exclusive):
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    if exclusive:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(name, flags, 0o640, dir_fd=directory)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor)
        raise RuntimeError("registry files must be regular single-link files")
    if exclusive:
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, 0o640)
    elif (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (uid, gid, 0o640):
        os.close(descriptor)
        raise RuntimeError("unexpected registry file ownership or mode")
    return descriptor

try:
    try:
        lock = open_file(name + ".lock", exclusive=True)
    except FileExistsError:
        lock = open_file(name + ".lock", exclusive=False)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            descriptor = open_file(name, exclusive=True)
        except FileExistsError:
            descriptor = open_file(name, exclusive=False)
            os.close(descriptor)
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(b'{"version":2,"entries":[]}\n')
                stream.flush()
                os.fsync(stream.fileno())
        os.fsync(directory)
    finally:
        os.close(lock)
finally:
    os.close(directory)
PY
}

validate_service_unit_sources
verify_optional_regular_file /usr/lib/kiron/data/prism-runtime-policy.json
verify_optional_regular_file /usr/lib/kiron/data/shared/local-model-registry.json
verify_optional_regular_file /usr/lib/kiron/data/shared/local-model-registry.json.lock
ensure_kiron_identities
ensure_kitt_worker_identity
verify_kitt_worker_identity
ensure_kitt_worker_secret_dir
ensure_kiron_dashboard_env_contract

install_wrapper_sources
install_sudoers_sources
install_service_unit_sources

# Runtime-Verzeichnisse fuer KIron und kitt-worker. User/Group muessen vorher existieren.
tmpfiles_src="/opt/kiron/system/tmpfiles.d/kiron-runtime.conf"
[ -f "$tmpfiles_src" ] || fail "Pflicht-tmpfiles fehlt: $tmpfiles_src"
mkdir -p /etc/tmpfiles.d
cp "$tmpfiles_src" /etc/tmpfiles.d/kiron-runtime.conf
systemd-tmpfiles --create /etc/tmpfiles.d/kiron-runtime.conf
initialize_local_model_registry
verify_kiron_runtime_paths
verify_kitt_worker_runtime_dir /usr/lib/kiron/data/kitt-worker
verify_kitt_worker_runtime_dir /run/kiron/kitt-worker
echo "  /run/kiron, /usr/lib/kiron/data und HF-Cache tmpfiles.d installiert"
echo "  /etc/kiron/kitt-worker Secret-Verzeichnis installiert (ohne Credential-Datei)"

# NVIDIA unattended-upgrades Blacklist
if [ -f /opt/kiron/system/apt/50unattended-upgrades ]; then
    cp /opt/kiron/system/apt/50unattended-upgrades /etc/apt/apt.conf.d/50unattended-upgrades
    echo "  NVIDIA apt-Blacklist installiert"
fi

# nvidia-persistenced Drop-in
if [ -f /opt/kiron/system/systemd/nvidia-persistenced-enable.conf ]; then
    mkdir -p /etc/systemd/system/nvidia-persistenced.service.d/
    cp /opt/kiron/system/systemd/nvidia-persistenced-enable.conf \
       /etc/systemd/system/nvidia-persistenced.service.d/enable.conf
    systemctl daemon-reload
    if systemctl list-unit-files | grep -q '^nvidia-persistenced\.service'; then
        systemctl enable --now nvidia-persistenced.service
        echo "  nvidia-persistenced Drop-in installiert und aktiviert"
    else
        echo "  nvidia-persistenced Drop-in installiert, aber Unit nicht vorhanden (Paket fehlt) - enable uebersprungen"
    fi
fi

echo "Fertig."
