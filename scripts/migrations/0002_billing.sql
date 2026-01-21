--0002_billing.sql
-- Billing users, tokens, and entitlements

CREATE TABLE IF NOT EXISTS billing_users (
    user_id TEXT PRIMARY KEY,
    email TEXT,
    created_ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS billing_tokens (
    token_sha256 TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_ts TEXT NOT NULL,
    label TEXT,
    FOREIGN KEY (user_id) REFERENCES billing_users(user_id)
);

CREATE TABLE IF NOT EXISTS billing_entitlements (
    entitlement_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    tier TEXT NOT NULL,           -- free | supporter | pro
    starts_ts TEXT NOT NULL,
    ends_ts TEXT,                 -- NULL = no expiry
    source TEXT NOT NULL,          -- manual | stripe | promo
    meta_json TEXT,
    FOREIGN KEY (user_id) REFERENCES billing_users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_billing_tokens_user
    ON billing_tokens(user_id);

CREATE INDEX IF NOT EXISTS idx_billing_entitlements_user
    ON billing_entitlements(user_id);

CREATE INDEX IF NOT EXISTS idx_billing_entitlements_active
    ON billing_entitlements(user_id, starts_ts, ends_ts);
