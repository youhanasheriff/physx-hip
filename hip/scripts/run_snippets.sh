#!/bin/bash
# Runs every PhysX snippet binary in a bin directory headlessly and reports pass/fail.
# The snippets must have been built with -DPX_SNIPPETS_HEADLESS=ON, otherwise they open
# a window and never exit.
#
# usage: run_snippets.sh <bin-dir> [log-dir] [per-snippet-timeout-seconds]
# A snippet passes when it exits 0 within the timeout and reports no PhysX error.
# Snippets that cannot run unattended are skipped and listed with the reason.
set -u
BIN_DIR=${1:?usage: run_snippets.sh <bin-dir> [log-dir] [timeout]}
LOG_DIR=${2:-$BIN_DIR/snippet-logs}
TIMEOUT=${3:-300}
mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)
PHYSX_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../physx" && pwd)
pass=0; fail=0; skip=0; failed=()
cd "$BIN_DIR" || exit 2

# Arguments some snippets require, and snippets that need something we do not provide.
snippet_args() {
	case "$1" in
		SnippetVehicle*) echo "--vehicleDataPath=$PHYSX_ROOT/snippets/media/vehicledata" ;;
		SnippetOmniPvdMesh_64|SnippetOmniPvdMultiGeom_64) echo "--omnipvdfile=$LOG_DIR/$1.ovd" ;;
	esac
}
skip_reason() {
	case "$1" in
		SnippetOmniPvd_64|SnippetOmniPvdNet_64) echo "needs an OmniPVD reader listening on a socket" ;;
		SnippetLoadCollection_64) echo "needs serialized collection files as input" ;;
		SnippetProfilerConverter_64) echo "needs a profiler capture file as input" ;;
	esac
}
for exe in Snippet*_64; do
	[ -x "$exe" ] || continue
	reason=$(skip_reason "$exe")
	if [ -n "$reason" ]; then
		skip=$((skip+1)); printf 'SKIP %-48s %s\n' "$exe" "$reason"; continue
	fi
	log="$LOG_DIR/$exe.log"
	start=$(date +%s.%N)
	# shellcheck disable=SC2046
	timeout "$TIMEOUT" "./$exe" $(snippet_args "$exe") > "$log" 2>&1 < /dev/null
	rc=$?
	secs=$(printf '%.1f' "$(echo "$(date +%s.%N) - $start" | bc)")
	# PhysX reports problems through the error callback, which the snippets print.
	errs=$(grep -ciE "PhysX (error|Internal error)|eINTERNAL_ERROR|eINVALID_|eABORT|Assertion|assert failed|Segmentation" "$log")
	if [ "$rc" -eq 0 ] && [ "$errs" -eq 0 ]; then
		pass=$((pass+1)); printf 'PASS %-48s %6ss\n' "$exe" "$secs"
	else
		fail=$((fail+1)); failed+=("$exe"); printf 'FAIL %-48s %6ss rc=%s errlines=%s\n' "$exe" "$secs" "$rc" "$errs"
	fi
done
echo "---"
echo "passed=$pass failed=$fail skipped=$skip"
[ "$fail" -eq 0 ] || { printf 'failed: %s\n' "${failed[@]}"; exit 1; }
