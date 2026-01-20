CREATE TABLE IF NOT EXISTS billing_token_revocations (
    token_sha256 TEXT PRIMARY KEY,
    revoked_ts   TEXT NOT NULL,
    reason       TEXT,
    meta_json    TEXT
);

CREATE INDEX IF NOT EXISTS idx_billing_token_revocations_revoked_ts
ON billing_token_revocations(revoked_ts);
