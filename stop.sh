#!/usr/bin/env bash
# Stop the Step-3.7-Flash llama-server.
set -euo pipefail
pkill -f "llama-server.*Step-3.7-flash" && echo "stopped" || echo "no matching server process"
