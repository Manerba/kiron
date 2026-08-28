#!/bin/sh
set -eu

entrypoint_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec "$entrypoint_dir/python" -P -m kiron_common.local_model_registry.cli "$@"
