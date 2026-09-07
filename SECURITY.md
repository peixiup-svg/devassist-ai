# Security policy

DevAssist is a diagnostic assistant, not a code-execution sandbox.

- It never executes user-submitted commands or code.
- Retrieved text is treated as untrusted evidence and cannot add tools.
- Citations may only reference URLs already present in the indexed corpus.
- Credential-like strings are redacted before feedback is persisted.
- The reindex endpoint is disabled unless `DEVASSIST_ADMIN_TOKEN` is configured.
- Do not submit private source code, production logs, access tokens, or personal data to a public deployment.

For a real deployment, add authentication, tenant-level document authorization,
durable rate limiting, encrypted storage, dependency scanning, and an incident
response process.

