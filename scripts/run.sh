#!/usr/bin/env bash
# Pull secrets from macOS Keychain into the process env, then run the app.
# Expects each key stored via: security add-generic-password -a "$USER" -s <name> -w "<value>"
set -euo pipefail

export GROQ_API_KEY="$(security find-generic-password -a "$USER" -s groq-api-key -w)"
export OPENROUTER_API_KEY="$(security find-generic-password -a "$USER" -s openrouter-api-key -w)"
export ELEVEN_API_KEY="$(security find-generic-password -a "$USER" -s eleven-api-key -w)"

exec uv run voice-agent "$@"
