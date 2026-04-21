#!/usr/bin/env bash
# Source this file (do NOT execute it) to get a shell set up for building and
# running the terrain_toolkit_ros package:
#
#   source <repo>/ros/dev-shell.sh
#
# It activates the uv-created .venv at the repo root, sources ROS 2 Kilted,
# and — if the repo sits inside a colcon workspace (<ws>/src/terrain_toolkit) —
# sources the workspace install overlay as well.

_dev_shell_path="${BASH_SOURCE[0]:-$0}"
_dev_shell_dir="$(cd "$(dirname "$_dev_shell_path")" && pwd)"
_dev_shell_repo="$(cd "$_dev_shell_dir/.." && pwd)"

if [[ -f "$_dev_shell_repo/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$_dev_shell_repo/.venv/bin/activate"
else
    echo "[dev-shell] no .venv at $_dev_shell_repo/.venv" >&2
    echo "[dev-shell] create one with: cd $_dev_shell_repo && uv venv --system-site-packages && uv pip install -e ." >&2
fi

if [[ -f /opt/ros/kilted/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/kilted/setup.bash
else
    echo "[dev-shell] /opt/ros/kilted/setup.bash not found — skipping ROS source" >&2
fi

# If the repo is <ws>/src/terrain_toolkit, source <ws>/install/setup.bash when present.
_dev_shell_ws="$(cd "$_dev_shell_repo/../.." 2>/dev/null && pwd)"
if [[ -n "$_dev_shell_ws" && -f "$_dev_shell_ws/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "$_dev_shell_ws/install/setup.bash"
    echo "[dev-shell] sourced workspace overlay at $_dev_shell_ws/install"
fi

unset _dev_shell_path _dev_shell_dir _dev_shell_repo _dev_shell_ws
