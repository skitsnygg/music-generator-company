# Music Generator Company  
## v1

This document defines the **stable v1 release** for Music Generator Company (MGC) drop artifacts.

Anything conforming to this contract is considered **reviewable, portable, and reproducible**.

Once published, this contract is frozen for v1.

---

## Scope

This contract applies to artifacts produced by:

- `mgc run drop`
- `mgc submission build`
- `mgc submission latest`

and verified by:

- `mgc submission verify`

---

## Determinism Guarantees

For deterministic runs (`--deterministic` or `MGC_DETERMINISTIC=1`):

- All timestamps are derived from a fixed run timestamp
- All IDs are derived from stable UUIDv5 inputs
- ZIP archives are written with:
  - stable file ordering
  - fixed timestamps
  - normalized permissions
- Re-running the same command with the same inputs **must produce byte-identical output**

Non-deterministic runs may vary in timestamps and IDs but must still conform to layout and validation rules.

---

## Submission Directory Layout

Each drop produces a submission directory at:

```
data/submissions/<drop_id>/
```

Required files:

```
data/submissions/<drop_id>/
├── submission.zip
└── submission.json
```

---

## submission.json (Required)

A self-describing pointer file used by CI and humans.

### Schema

```json
{
  "schema": "mgc.submission.v1",
  "drop_id": "<uuid>",
  "run_id": "<uuid>",
  "deterministic": true,
  "ts": "2020-01-01T00:00:00Z",
  "submission_zip": "submission.zip"
}
```

### Rules

- `submission_zip` is **relative** to the submission directory
- `ts` must be the run timestamp (not wall clock)
- Keys must be written using stable JSON ordering

---

## submission.zip (Required)

A portable, review-ready archive.

### ZIP Root Layout

```
submission/
├── README.md
└── drop_bundle/
    ├── playlist.json
    ├── daily_evidence.json
    └── tracks/
        └── <track_id>.<ext>
```

---

## README.md

Human-readable summary generated at build time.

Must include:
- drop_id
- run_id
- track_id
- context
- provider
- deterministic flag
- instructions for review

Timestamps inside the README must use the run timestamp for deterministic runs.

---

## drop_bundle Contents

### Required Files

| File | Purpose |
|------|---------|
| playlist.json | Points to bundled audio under tracks/ |
| daily_evidence.json | Provenance, metadata, sha256 hashes |
| tracks/<track_id>.<ext> | Audio asset |

### Rules

- All paths inside playlist.json must be **relative**
- All hashes in evidence must match the bundled files
- No external references are allowed

---

## Validation

The authoritative validator is:

```bash
mgc submission verify submission.zip
```

Validation performs:

1. ZIP layout inspection
2. Bundle extraction
3. Bundle validation
4. Hash verification

Exit codes:
- `0` → valid submission
- `2` → invalid submission

---

## CI Requirements

CI **must** assert:

- submission.zip exists
- submission.json exists
- submission verify passes
- deterministic runs produce identical outputs

Any CI failure indicates a **contract violation**.

---

## Versioning

- This document defines **v1**
- Breaking changes require a new contract version:
  - mgc.submission.v2
  - submission_contract_v2.md
- v1 behavior must remain supported indefinitely

---

## Non-Goals

This contract does **not** specify:

- Audio quality or artistic merit
- Publishing destinations
- Provider internals
- UI or web presentation

Those concerns are intentionally decoupled.

---

## Summary

If an artifact satisfies this contract:

- It is portable
- It is reviewable
- It is reproducible

Anything else is a bug.
