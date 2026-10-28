#!/bin/bash
# Prepare an Advanced RAG workspace: venv, Jupyter kernel, bootcamp keys, data,
# and a background Jupyter server.
#
# The Coder template runs `bash scripts/setup.sh` from the repository root on
# every workspace start, so every step is idempotent and a second run changes
# nothing that the first run set up. It also works on your own machine.
#
# Usage: bash scripts/setup.sh [--fetch <spec>] [--no-fetch] [--no-onboard] [--no-jupyter]
#
#   --fetch <spec>  What to prefetch, passed to aieng.advanced_rag.corpora.download:
#                   "core" (default), RI numbers such as "core 3", or "none".
#                   ADVANCED_RAG_FETCH sets the same thing. Any value but
#                   "none" also installs every published reference-artifact
#                   manifest.
#   --no-fetch      Same as --fetch none.
#   --no-onboard    Do not run the platform's onboard command.
#   --no-jupyter    Do not start Jupyter. ADVANCED_RAG_START_JUPYTER=0 does the same.
#
# ADVANCED_RAG_BOOTCAMP_NAME, or the BOOTCAMP_NAME the Coder template sets, overrides
# the bootcamp name passed to onboard.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

bootcamp_name="${ADVANCED_RAG_BOOTCAMP_NAME:-${BOOTCAMP_NAME:-advanced-rag-bootcamp}}"
fetch_spec="${ADVANCED_RAG_FETCH:-core}"
run_onboard=1
start_jupyter="${ADVANCED_RAG_START_JUPYTER:-1}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --fetch) fetch_spec="${2:?--fetch needs a value}"; shift 2 ;;
        --no-fetch) fetch_spec="none"; shift ;;
        --no-onboard) run_onboard=0; shift ;;
        --no-jupyter) start_jupyter=0; shift ;;
        *) echo "setup.sh: unknown option $1" >&2; exit 2 ;;
    esac
done

log() { echo "[setup] $*"; }

# A VIRTUAL_ENV exported by another project makes uv warn and can point it at
# the wrong environment. This script only ever uses the repository's .venv.
unset VIRTUAL_ENV

# litellm loads any .env it finds above its install directory at import unless
# LITELLM_MODE is set. Jupyter inherits this, so kernels never pick up a .env
# from a parent or home directory.
export LITELLM_MODE=PRODUCTION

# 1. Dependencies. --frozen installs exactly what uv.lock pins and never
# re-resolves, so a start cannot pick up a new release. --all-packages with
# --all-extras installs every extra the reference implementations import.
log "Syncing the environment from uv.lock"
uv sync --frozen --all-packages --all-extras --dev

python=".venv/bin/python"

# 2. Jupyter kernel. Reinstalling overwrites the same kernelspec.
"$python" -m ipykernel install --user --name=advanced-rag --display-name "Advanced RAG Bootcamp" >/dev/null
log "Jupyter kernel advanced-rag installed"

# 3. Bootcamp keys. onboard prints export lines on stdout and its progress on
# stderr, runs the onboarding gate once, and records the result. On a later
# start it finds the participant already onboarded and only prints the keys.
# The keys stay in this process and the Jupyter server it starts, and are never
# written to disk.
github_user="${GITHUB_USER:-${GH_USER:-}}"
if [ "$run_onboard" = 1 ] && [ -x .venv/bin/onboard ] && [ -n "$github_user" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    log "Loading bootcamp keys with onboard"
    # onboard's progress, and any traceback, go to a log file. Facilitators and
    # anyone not yet provisioned have no secret, which is expected, so that case
    # gets one plain line.
    onboard_log="${XDG_CACHE_HOME:-$HOME/.cache}/aieng-advanced-rag/onboard.log"
    mkdir -p "$(dirname "$onboard_log")"
    onboard_status=0
    exports="$(.venv/bin/onboard --bootcamp-name "$bootcamp_name" --test-script tests/test_integration.py --test-marker integration_test 2>"$onboard_log")" || onboard_status=$?
    eval "$exports"
    if [ -z "${OPENAI_API_KEY:-}" ] && grep -q "Failed to fetch environment variables" "$onboard_log"; then
        log "No proxy key for this account; notebooks run from cached results."
    elif [ "$onboard_status" -ne 0 ]; then
        log "WARNING: onboard exited with $onboard_status; details in $onboard_log. Run implementations/getting_started/00_environment_check.ipynb, or see docs/setup.md."
    fi
elif [ "$run_onboard" = 1 ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    log "Skipping onboard: it needs GITHUB_USER (or GH_USER) and .venv/bin/onboard. Keys can come from .env instead; see docs/setup.md."
fi

# New terminals load the keys the same way. ~/.profile covers the login shells
# that VS Code Server and Jupyter kernels start, and ~/.zshrc covers zsh. The
# block holds a command, not a secret. It runs the gate, never --skip-test,
# because onboard 0.7.x marks a participant onboarded under --skip-test without
# testing anything. Once the gate has passed, onboard only prints the keys.
if [ "$run_onboard" = 1 ] && [ -n "$github_user" ]; then
    for rc_file in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        if grep -q '# >>> advanced-rag setup >>>' "$rc_file" 2>/dev/null; then
            continue
        fi
        cat >>"$rc_file" <<EOF
# >>> advanced-rag setup >>>
# Added by $repo_root/scripts/setup.sh. Loads the bootcamp keys and the repo venv.
export LITELLM_MODE=PRODUCTION
if [ -z "\${OPENAI_API_KEY:-}" ] && [ -x "$repo_root/.venv/bin/onboard" ]; then
    eval "\$(cd "$repo_root" && .venv/bin/onboard --bootcamp-name "$bootcamp_name" --test-script tests/test_integration.py --test-marker integration_test 2>/dev/null)"
fi
if [ -f "$repo_root/.venv/bin/activate" ] && [ -z "\${VIRTUAL_ENV:-}" ]; then
    . "$repo_root/.venv/bin/activate"
fi
# <<< advanced-rag setup <<<
EOF
        log "Added the key reload block to $rc_file"
    done
fi

# 4. Data. Each file is checked against its pinned SHA-256, so a warm cache
# costs a second. A failed download must not stop the workspace from starting.
if [ "$fetch_spec" != "none" ]; then
    fetch_args=()
    for part in $fetch_spec; do
        if [ "$part" = "core" ]; then fetch_args+=("--core"); else fetch_args+=("$part"); fi
    done
    log "Fetching data: $fetch_spec"
    "$python" -m aieng.advanced_rag.corpora.download "${fetch_args[@]}" ||
        log "WARNING: some data did not fetch. Re-run: python -m aieng.advanced_rag.corpora.download ${fetch_args[*]}"

    # Reference artifacts: every implementations/**/reference-artifacts.json.
    # Artifacts already installed with their pinned digest are skipped, so a
    # warm start downloads nothing, and a failure only warns.
    if compgen -G "implementations/*/reference-artifacts.json" >/dev/null ||
        compgen -G "implementations/*/*/reference-artifacts.json" >/dev/null; then
        log "Installing reference artifacts"
        "$python" scripts/fetch_reference_artifacts.py ||
            log "WARNING: some reference artifacts did not install. Re-run: python scripts/fetch_reference_artifacts.py"
    fi
fi

# 5. Jupyter, last and in the background, so this script exits and the
# workspace reports ready. A second start finds the server and leaves it.
if [ "$start_jupyter" = 1 ]; then
    if "$python" -m jupyter server list 2>/dev/null | grep -q ':8888/'; then
        log "Jupyter is already running on port 8888"
    else
        jupyter_log="${XDG_CACHE_HOME:-$HOME/.cache}/aieng-advanced-rag/jupyter.log"
        mkdir -p "$(dirname "$jupyter_log")"
        nohup "$python" -m jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.token='' \
            >"$jupyter_log" 2>&1 </dev/null &
        log "Jupyter Lab started on port 8888, logging to $jupyter_log"
    fi
fi

log "Done. Next: run implementations/getting_started/00_environment_check.ipynb"
