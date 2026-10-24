# Validation

Validated with Python 3.12 and Node 24 in the build workspace.

- 71 automated backend, API integration and diagnostic tests pass; see test files for cases.
- `node --check frontend/app.js` passes.
- `bash -n run.sh` passes.
- HTML IDs and JavaScript element references checked for consistency.
- Both bundled person/dog story fixtures validate against the current schemas.
- Compiled example prompts produced for both requested TTS models with stable two-speaker mappings.
- Gemini and Fish request formats checked against official model, structured-output, Drama 3 and TTS documentation.

Tests mock Gemini and Fish. Coverage includes image preparation, source extraction without creative traits, separate text-only story direction, restoration of observed species/traits, rejection of invented speakers, explicit model validation, prompt-preview sanitation, native one-request scenes with ordered voice arrays, per-line rendering, WAV assembly, pauses, timing, and safe error handling. Native scene responses return no fabricated line timestamps.

Not verified here: live Gemini inference, real Fish synthesis, account/model access, voice-library match quality, voice quality or end-to-end latency. No user API keys were supplied.

Browser rendering was not exercised. No browser binary was available and the previous browser download failed in this environment. Frontend syntax and API contracts were checked; visual layout, camera picker and audio behavior need a local browser rehearsal.

The included person/dog JSON files are manually authored examples. They are not represented as results from a real Gemini run. Their compiled prompt text has no assigned real voice IDs and is not an audio artifact.

Starlette TestClient emits a dependency deprecation warning; tests pass. It does not affect the application's runtime HTTP client.

The generic Gemini 400 rejection was reproduced with mocked HTTP responses and the bounded compatibility path was verified. The live cause was not established. `diagnose.py --story` performs explicit live text, image-extraction and writing checks on the user’s account; no such check was run in this build workspace.
