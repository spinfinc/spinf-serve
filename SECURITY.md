# Security

Report a vulnerability to **security@spinf.com**, not in a public issue. Say which repository and version, what the
problem is and how to reproduce it; we answer within a few days and credit you in the fix unless you prefer not to.

A self-hosted spinf-serve without `--token` is open to whoever can reach its port: put it behind your own network
boundary or start it with a token. The hosted API's keys are the account owner's responsibility; a leaked key is revoked
in the console.
