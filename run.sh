#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

find_python() {
  for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

install_python() {
  echo "Python 3.10+ was not found. Attempting installation..."
  if command -v brew >/dev/null 2>&1; then
    brew install python@3.12
  elif command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
    $SUDO apt-get update
    $SUDO apt-get install -y python3 python3-venv python3-pip
  elif command -v dnf >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
    $SUDO dnf install -y python3 python3-pip
  elif command -v pacman >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
    $SUDO pacman -S --noconfirm python python-pip
  elif command -v zypper >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
    $SUDO zypper --non-interactive install python3 python3-pip
  elif command -v apk >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
    $SUDO apk add python3 py3-pip
  else
    return 1
  fi
}

PYTHON="$(find_python || true)"
if [ -z "$PYTHON" ]; then
  install_python || true
  PYTHON="$(find_python || true)"
fi
if [ -z "$PYTHON" ]; then
  echo "Python 3.10 or newer could not be installed automatically." >&2
  exit 2
fi

exec "$PYTHON" bootstrap.py "$@"
