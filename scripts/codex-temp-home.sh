#!/usr/bin/env bash

set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/codex-temp-home.sh [--keep-home] [--home-base DIR] [codex args...]

Options:
  --keep-home        Keep the temporary CODEX_HOME directory after codex exits.
  --home-base DIR    Parent directory for the temporary CODEX_HOME. Default: /tmp
  -h, --help         Show this help message.

Examples:
  bash scripts/codex-temp-home.sh
  bash scripts/codex-temp-home.sh --keep-home
  bash scripts/codex-temp-home.sh -- model list
EOF
}

keep_home=0
home_base="/tmp"
codex_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-home)
      keep_home=1
      shift
      ;;
    --home-base)
      if [[ $# -lt 2 ]]; then
        echo "error: --home-base requires a directory argument" >&2
        exit 1
      fi
      home_base="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      codex_args=("$@")
      break
      ;;
    *)
      codex_args+=("$1")
      shift
      ;;
  esac
done

if ! command -v codex >/dev/null 2>&1; then
  echo "error: codex command not found in PATH" >&2
  exit 127
fi

mkdir -p "$home_base"
tmp_home="$(mktemp -d "${home_base%/}/codex-${USER:-user}.XXXXXX")"
export CODEX_HOME="$tmp_home"

cleanup() {
  if [[ $keep_home -eq 1 ]]; then
    echo "Kept CODEX_HOME: $CODEX_HOME"
    return
  fi

  rm -rf "$CODEX_HOME"
}

trap cleanup EXIT INT TERM

echo "Using temporary CODEX_HOME: $CODEX_HOME"
echo "This directory will be removed when codex exits."

if [[ ${#codex_args[@]} -gt 0 ]]; then
  codex "${codex_args[@]}"
else
  codex
fi
