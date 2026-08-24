# Bootstrap Task: Fill Project Development Guidelines

## Purpose

Populate `.trellis/spec/` with the repository's actual development conventions
so future Trellis implementation and review sessions use project-specific,
source-backed guidance instead of generic defaults.

## Scope

This bootstrap covers the Python backend in `sglang_omni/`, the router in
`sglang_omni_router/`, and their tests and configuration. General thinking
guides under `.trellis/spec/guides/` remain unchanged unless the repository
clearly contradicts them.

## Requirements

- Inspect existing convention files, source code, tests, and tool configuration.
- Ground each important rule in real repository paths and established patterns.
- Describe current practice rather than aspirational architecture.
- Keep the guidance in English and remove all scaffold placeholders.
- Maintain an index whose links match the files that actually exist.
- Do not change product code or unrelated documentation during bootstrap.

## Backend Deliverables

| File | Content |
|---|---|
| `.trellis/spec/backend/index.md` | Scope and links to the backend guides |
| `.trellis/spec/backend/directory-structure.md` | Package boundaries and placement of routes, services, models, transports, and tests |
| `.trellis/spec/backend/error-handling.md` | Domain errors, HTTP/WebSocket mapping, cancellation, cleanup, and tests |
| `.trellis/spec/backend/logging-guidelines.md` | Logger ownership, levels, context, safety, and volume |
| `.trellis/spec/backend/quality-guidelines.md` | Formatting, typing, async lifecycle, tests, and review checks |

No database guideline is required because the repository currently has no
database or persistence package. The backend index must record that conclusion;
add database guidance only if a real persistence layer is introduced.

## Acceptance Criteria

- [x] Backend guidelines are populated with source-backed conventions.
- [x] The guides contain concrete repository examples.
- [x] The backend index links to every backend guide and has no stale link to a
  database guide.
- [x] No placeholders or scaffold-only instructions remain in the deliverables.
- [x] Changes stay within `.trellis/spec/backend/` and this task artifact.

## Completion

After review passes, finish and archive this task with the project Trellis task
commands.
