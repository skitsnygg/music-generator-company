# MGC CLI Contract (CI-facing)

This document defines the behavior that CI (and humans piping output) relies on.
If you change behavior here, update CI + tests in the same commit.

## Global flags

- `--db PATH`
  - Can appear:
    - before the top-level command: `mgc --db ... rebuild ls`
    - after the top-level command (but before subcommands): `mgc rebuild --db ... ls`
    - after a subcommand where that subcommand defines it (e.g. run subcommands): `mgc run status --db ...`
  - CI commonly uses `mgc rebuild ls --db ... --json`.

- `--json`
  - When supported by a command, it MUST cause stdout to emit exactly one JSON object.

## Output rule: "Exactly one JSON object"
For commands that support JSON mode:
- stdout MUST contain exactly one JSON object and a trailing newline.
- No logs, banners, or warnings may appear on stdout in JSON mode.
- Logs may appear on stderr.

## Exit codes
Unless otherwise stated:
- `0` success
- `1` not found / empty result (non-fatal informational)
- `2` fatal / validation error / "fail gate"

### Commands

#### `mgc status`
- `--json` supported.
- Exit:
  - `0` always when it runs successfully (it is an informational snapshot).

#### `mgc rebuild ls`
- `--json` supported.
- Exit:
  - `0` on success

#### `mgc rebuild playlists|tracks`
- `--json` optional (if implemented), but MUST NOT print extra stdout junk in JSON mode.
- Exit:
  - `0` on success
  - `2` if determinism-check fails (when enabled)

#### `mgc rebuild verify playlists|tracks`
- `--json` supported.
- Exit:
  - `0` on success
  - `2` on diff / strict mismatch failure

#### `mgc run diff`
- `--json` supported.
- Exit:
  - `0` no blocking changes
  - `2` blocking changes when `--fail-on-changes` is set
  - `0` and `found=false` payload when there are insufficient manifests / missing paths

#### `mgc run status`
- `--json` supported.
- Exit:
  - `0` when found=true and no requested failure condition
  - `1` when found=false
  - `2` when `--fail-on-error` and any stage has status=error

## JSON schema stability
- Field order is not guaranteed, but keys and semantics should remain stable.
- New fields may be added; existing fields should not be removed without updating CI.
