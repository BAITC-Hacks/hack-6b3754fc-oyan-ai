# Dependency approval

Prefer the language standard library and already-approved stack. A dependency must remove more five-hour risk than it introduces.

## Required checks

- Exact capability needed; reject convenience-only additions with trivial local implementations.
- Official documentation and canonical package registry/source repository.
- License exists, is compatible, and will be disclosed.
- Stable maintained release, recent relevant activity, and no archived/deprecated status.
- Evidence of real adoption appropriate to the ecosystem; do not use stars alone.
- Known security advisories, suspicious ownership changes, typosquatting risk, and opaque binaries/install scripts.
- Transitive dependency count, native build requirements, platform compatibility, cold install time, and network dependence.
- API clarity, testability, and a simpler fallback.

## Immediate rejection signals

- no clear license or canonical maintainer;
- archived, abandoned, or replacement recommended by its maintainer;
- package/repository name does not match the official source;
- copied application or substantial ready-made solution rather than a library;
- requires personal credentials that judges cannot use;
- core demo fails completely when the service is unavailable;
- install or build cannot be reproduced on the event machine.

Pre-1.0 status or a small community is a warning, not an automatic rejection, when the organizer mandates the technology. In that case, isolate it behind a small adapter and provide a fallback.

## Decision record

```text
Candidate and version:
Capability needed:
Official docs/repository:
License:
Maintenance/adoption evidence:
Security/operational risks:
Fallback:
Decision: approve | approve with fallback | reject
```

Pin approved versions in the language's lock file. Never install directly from an arbitrary Git URL or paste substantial code without source and license disclosure.
