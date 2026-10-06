# Prism controller

The controller owns one unprivileged native child and one model slot. It does
not admit GPU work independently, register models, fetch artifacts, execute tools,
or expose an inference API. The proxy owns request admission and the public API.

`main.create_app(policy, resolver, admission, ...)` is injectable for offline
tests. Production uses the fixed `composition.build_controller(policy)` and
the same Catalog/local registry and `kiron_common.gpu_admission` as the proxy.
Imports start no process and perform no discovery.

## Control wire

HTTP is served only on `/run/kiron/prism/control.sock`; the native backend binds
to `127.0.0.1:11442` by default. There is no TCP management listener.

- `GET /health`: current boot/spawn generation, state, deployment ID,
  configuration fingerprint, resolver revision, backend alias and error code.
- `POST /load` and `POST /unload`: exactly `deployment_id`, `snapshot_revision`,
  `expected_generation: {boot_id, process_id}` and `operation_id`. No paths,
  environment, profiles or argv are accepted from the caller. Body limit: 4096
  bytes; body deadline: five seconds. Unknown fields are rejected.

The first health observation has a fresh random boot ID and null process ID.
Every attempted spawn receives a fresh random process token, not an OS PID.
Generation and snapshot mismatches, conflicting operations and occupied ports
fail before spawn. Same-operation retries join the bounded operation; a reused
operation ID with different arguments conflicts. Completed retries return current
observations, never stale historical `loaded` states. Different loads do not evict.

`loaded` requires the live owned process, its listening socket, `/health` success
and the exact per-spawn alias in `/v1/models`. Health loss becomes `unknown`.
A confirmed crash triggers process-group cleanup before shared reservation release.
An unconfirmed process or admission cleanup stays `unknown`. A new controller boot
does not clear the previous controller's shared reservations automatically.

Current slot observations expose only `active_requests`, `slot_task_id` and
`slots_observed_at`. Missing/malformed/unreachable observations remain unknown.
The pinned source enables `/slots` by default (`common/common.h:659`); its handler
uses a high-priority queued snapshot (`server-context.cpp:4601ff`). Therefore an
idle slot alone does **not** prove that a disconnected, as-yet unassigned request
has ended. Request cleanup must also establish task/start identity or retain its
unknown ticket. No request cancellation implicitly unloads the model.

## Shared boundaries

All callbacks are asynchronous and fail closed:

| Boundary | Contract |
| --- | --- |
| `resolver.snapshot()` | Immutable snapshot with `revision` and `resolve_deployment(id)`. Checked again after hashing, before admission/spawn. |
| `admission.validate(op, deployment, generation, action)` | Validate the proxy-owned reservation and current generation for `load`/`unload`. |
| `admission.loaded(op, deployment, new_generation)` | Bind the confirmed resident to its fresh spawn token. |
| `admission.heartbeat(op, deployment, generation)` | Renew only after actual owned-process readiness succeeds. |
| `admission.drain(op, deployment, generation, deadline_monotonic)` | Stop new work and confirm existing request work ended within the deadline. |
| `admission.terminated(op, deployment, generation)` | Called only after confirmed process-group end; also handles failed loads before `loaded`. |

The singleton `controller.lock` only prevents duplicate control servers. It is
not a GPU lock. GPU state and process-wide locking remain in `/run/kiron/vram`.
Disconnecting the HTTP caller does not orphan its already accepted operation.
Shutdown performs bounded drain and TERM/KILL/reap; WNOWAIT reserves a dead
leader's PID until final group cleanup. The unit uses `KillMode=mixed` so TERM
reaches the controller first and the stop deadline still kills its entire cgroup.

## Root-owned policy and installation inputs

The sole parser is `kiron_common.prism_runtime_policy.Policy`. Its
`resource_profiles()` and `registration_profiles()` expose the same measured
profiles to the resolver and local registration validator.

The policy path is `/usr/lib/kiron/data/prism-runtime-policy.json`, root-owned,
not group/other writable, readable by `kiron-config`. Schema version is 1:

- `runtime_root`: production bundle below `/usr/lib/kiron/runtimes/prism/`.
- `binary`: binary path relative to that bundle; `library_dirs`: relative library
  directories. All files and directory ancestors must be root-owned and immutable.
- `bundle_manifest`: complete relative-file map with `{sha256: ...}` for regular
  files and `{link: ...}` for contained packaged library symlinks. Extra files,
  changed bytes and escaping links fail validation.
- `artifact_roots`: immutable roots beneath `/usr/lib/kiron/data/gguf-models/`.
- `profiles`: mapping from profile ID to `gpu_layers`, `threads`, `context`,
  `batch`, `model_sha256`, `architecture`, optional `projector_sha256`,
  `gpu_memory_bytes`, `host_memory_bytes`, `memory_headroom_bytes`.
- Optional `port` and positive bounded `startup_timeout`, `health_timeout`,
  `drain_timeout`, `term_timeout`, `kill_timeout`; defaults 11442/180/2/30/10/5.

Profile `prism-bonsai27b-cuda40-c1024-v1` uses the measured GPU40/4-thread,
context1024/batch128/ubatch128/one-slot/CPU-projector configuration. Root integration
supplies the verified artifact hashes and conservative reservations of 4608 MiB
GPU, 8192 MiB host RAM and 512 MiB GPU headroom. These are explicit profile inputs,
not a service-side model registry. A deployment must match every profile field.

Artifact hashing retains verified read-only descriptors through spawn; argv uses
`/proc/self/fd/...`. The common bounded GGUF parser checks architecture and optional
projector dimensions on those same held files. Symlinks, FIFOs, mutable files,
hardlinks, escapes and changed file identities are rejected.

The deployed binary and CUDA libraries must form a separate, verified production
bundle. No production policy may point at test-venvs or a temporary build report.
Packaging and validating ELF dependencies/RPATHs belongs to deployment integration.

Required identity: primary `kiron-prism`, supplementary `kiron-common`,
`kiron-config`, `kiron-runtime`, `kiron-prism-control`, `video`, `render`.
`/run/kiron/prism` must be `kiron-prism:kiron-prism-control`, mode2750; the socket
is0660. Proxy has control-group access and reads the root-owned shared policy.
Writable unit paths are limited to
`/run/kiron/prism` and the existing `/run/kiron/vram` state area. No user, group,
unit or production configuration is created by importing or testing this service.

## Offline checks

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=services/kiron-common:services/kiron-prism \
  /usr/lib/kiron/test-venvs/local-inference/bin/python -m unittest discover \
  -s services/kiron-prism -p 'test_*.py' -v
```

Fixtures use tiny GGUF metadata files and fake child/resolver/admission objects.
The process-group test starts only a tiny Python fork fixture. The socket-rights
test drops to `nobody` for a temporary Unix socket; it never starts uvicorn, a
model or a unit. Hardware inference and real unit hardening remain separate gates.

## Isolated controller smoke harness

`scripts/prism/package-runtime.py` produces a test candidate under
`/usr/lib/kiron/test-runtimes/prism/bundles/prism-9a9394a-sm86-v1/`:
`runtime/`, `bundle-manifest.json`, `provenance.json`, `verification.json`.
It copies pinned source-build/CUDA libraries and licenses, removes absolute Prism
RUNPATHs using pinned patchelf, checks every ELF dependency and runs only `--version`
as `nobody`. Production installation remains separate.

The explicitly invoked harness needs `httpx`, `starlette`, `uvicorn` in the test
venv. Prepare hashes the existing model/projector without copying them, validates
the bundle, and writes an immutable registry and plan into a new report directory:

```bash
/usr/lib/kiron/test-venvs/kitt-worker/bin/python scripts/prism/smoke-controller.py \
  prepare --report-dir /usr/lib/kiron/test-runtimes/prism/reports/controller-runtime-v1
# Separate, explicitly authorized live step, after reviewing the prepared plan:
/usr/lib/kiron/test-venvs/kitt-worker/bin/python scripts/prism/smoke-controller.py \
  run --report-dir /usr/lib/kiron/test-runtimes/prism/reports/controller-runtime-v1
```

Run drops to `nobody:kiron-common` with no supplementary groups, starts the actual
workspace controller on the report's `uds/control.sock` and backend port18089,
then uses the real readonly registry resolver, ControllerAdmission, PrismProvider
and RuntimeService for load/health/text/unload. `registry/`, `admission/`, `uds/`
and `results/` are isolated; production policies, registries, markers and units
are never used. This is explicit Policy-object injection as in tests; the
production Policy.load path restrictions remain unchanged.

The isolated admission store does not see or prevent concurrent production GPU
work. Immediately before `run`, the operator must separately inspect production
service PIDs/start timestamps, Ollama residency, existing `/run/kiron/vram` markers
(read only), GPU processes/activity, available RAM and port18089. The operator
continues observing production during the run and can create the report-root
`STOP` file. These external checks are a precondition for this lifecycle probe;
its memory thresholds do not replace production admission or prove isolation of
the physical GPU.

The harness stops on report-root `STOP`, SIGTERM, 600-second workflow deadline,
GPU usage above11GiB or available RAM below2GiB, then drains its own controller.
These are monitoring/abort thresholds, not hard cgroup limits. Metrics and process
identity go to `results/metrics.jsonl`, final status to `results/result.json`.
The harness cannot be rerun into an existing report. Its text capability is a
private probe authorization based on #286; it publishes no production evidence.
The 600 seconds bound the workflow, not total process lifetime: setup and bounded
cleanup waits add time, and Python can still await resistant tasks/executor work
when exiting. The live runner must remain externally supervised; overdue cleanup
is recorded as failed and never counted as confirmed process termination.
Passing this smoke does not establish systemd hardening or deployment readiness.
