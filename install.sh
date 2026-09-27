#!/usr/bin/env sh
# Installs `podcode` for the current user. No sudo, pip, or system files.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10 or newer is required." >&2
  exit 1
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required.")
PY

USER_BIN=${RUNPOD_MODEL_BIN_DIR:-"$HOME/.local/bin"}
mkdir -p "$USER_BIN"

echo "Installing podcode to $USER_BIN..."
cp "$SCRIPT_DIR/runpod_model.py" "$USER_BIN/podcode"
chmod 755 "$USER_BIN/podcode"

if ! command -v opencode >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1 || ! command -v bash >/dev/null 2>&1; then
    echo "OpenCode installation requires curl and bash." >&2
    exit 1
  fi
  echo "OpenCode was not found; installing it..."
  INSTALLER=$(mktemp "${TMPDIR:-/tmp}/opencode-install.XXXXXX")
  trap 'rm -f "$INSTALLER"' EXIT HUP INT TERM
  curl -fsSL https://opencode.ai/v2/install -o "$INSTALLER"
  bash "$INSTALLER" --no-modify-path
  rm -f "$INSTALLER"
  trap - EXIT HUP INT TERM
  if [ ! -x "$HOME/.opencode/bin/opencode" ]; then
    echo "OpenCode installation completed but its binary was not found." >&2
    exit 1
  fi
  ln -sf "$HOME/.opencode/bin/opencode" "$USER_BIN/opencode"
fi
case ":${PATH}:" in
  *":${USER_BIN}:"*) ;;
  *)
    echo
    echo "Installed, but ${USER_BIN} is not on PATH. Add this to ~/.zshrc:"
    echo "  export PATH=\"${USER_BIN}:\$PATH\""
    ;;
esac

echo "Done. Open a new shell (if needed), then run: podcode up qwen3-coder-next"
