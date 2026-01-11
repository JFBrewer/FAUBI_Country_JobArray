#!/usr/bin/env bash
#SBATCH -c 24
#SBATCH -N 1
#SBATCH --mem=45g
#SBATCH --tmp=40g
#SBATCH -t 3-0:00                                                                                                                                                                         
#SBATCH -p msismall,ag2tb
# job-name and export vars (RUN_DIR, START, END, YEAR_MONTH) are set by submit_ensemble.sh
# NOTE: --output/--error are set by submit_ensemble.sh so logs land in ensemble/logs/

set -euo pipefail

: "${RUN_DIR:?RUN_DIR not set}"
: "${START:?START not set (YYYYMMDD)}"
: "${END:?END not set (YYYYMMDD)}"
: "${YEAR_MONTH:?YEAR_MONTH not set (YYYYMM)}"

echo "== $(date) =="
echo "Host: $(hostname)"
echo "RUN_DIR:     $RUN_DIR"
echo "START → END: $START → $END"
echo "YEAR_MONTH:  $YEAR_MONTH"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo

cd "$RUN_DIR"

# --- source your environment (temporarily disable nounset) ---
if [[ -f ./gcc.env ]]; then
  set +u
  # shellcheck disable=SC1091
  source ./gcc.env
  set -u
else
  echo "ERROR: gcc.env not found in $RUN_DIR" >&2
  exit 1
fi

CFG="geoschem_config.yml"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: $CFG not found in $RUN_DIR" >&2
  exit 1
fi

# Update start/end dates in geoschem_config.yml
tmpcfg="${CFG}.tmp.$$"
awk -v s="$START" -v e="$END" '
  BEGIN{FS=OFS=""}
  /^[[:space:]]*start_date:[[:space:]]*\[/ { sub(/\[.*\]/,"[" s ", 000000]"); print; next }
  /^[[:space:]]*end_date:[[:space:]]*\[/   { sub(/\[.*\]/,"[" e ", 000000]"); print; next }
  { print }
' "$CFG" > "$tmpcfg" && mv "$tmpcfg" "$CFG"

echo "Updated dates in $CFG:"
grep -E '^\s*(start_date|end_date)' "$CFG" || true
echo

# -------- Run the model --------
if [[ -x ./gcclassic ]]; then
  ./gcclassic
elif [[ -x ./runGCClassic.sh ]]; then
  ./runGCClassic.sh
elif command -v gcclassic >/dev/null 2>&1; then
  gcclassic
else
  echo "ERROR: Could not find gcclassic/runGCClassic.sh in $RUN_DIR or PATH" >&2
  exit 1
fi

# -------- Mark success for resume capability --------
done_dir="${RUN_DIR}/.month_done"
mkdir -p "$done_dir"
touch "${done_dir}/${YEAR_MONTH}.done"

echo
echo "== Completed at $(date) =="
