#!/usr/bin/env bash
# make_country_ensemble.sh
#
# Reads Country_Run_Manifest.csv (columns: RUNID,LIVESTOCK,MASK,COUNTRY)
# For each row, creates: ensemble/RUNID_LIVESTOCK_COUNTRY
# Then applies:
#   1) In HEMCO_Config.rc: replace token 1011 -> MASK ONLY on FAUBI lines
#      matching that LIVESTOCK code (e.g., FAUBI_*_Bf when LIVESTOCK=Bf).
#   2) In HEMCO_Config.rc: change any line that begins with
#        "0 FAUBI_*_{LIVESTOCK} ..."
#      to
#        "#0 FAUBI_*_{LIVESTOCK} ..."

set -euo pipefail

# ---------------- Config ----------------
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="${BASE}/template"
ENSEMBLE_DIR="${BASE}/ensemble"
MANIFEST_CSV="${BASE}/Country_Run_Manifest.csv"

HEMCO_RC_REL="HEMCO_Config.rc"

copy_template() {
  local src="$1" dst="$2"
  mkdir -p "$dst"
  cp -a "${src}/." "$dst"
}

trim_field() {
  # Trim whitespace and surrounding double-quotes; strip CR.
  local x="$1"
  x="${x//$'\r'/}"
  x="${x%\"}"; x="${x#\"}"
  printf "%s" "$x" | xargs
}

# ---------------- Edits ----------------
replace_mask_only_for_livestock_lines() {
  # Replace token 1011 -> MASK only for FAUBI lines matching this LIVESTOCK code
  # e.g. LIVESTOCK=Bf matches token2 like FAUBI_*_Bf
  local hemco_rc="$1"
  local livestock="$2"
  local mask="$3"

  if [[ ! -f "$hemco_rc" ]]; then
    echo "ERROR: Expected file for HEMCO rc, got: $hemco_rc" >&2
    exit 1
  fi
  if [[ ! "$mask" =~ ^[0-9]{4}$ ]]; then
    echo "ERROR: MASK must be a 4-digit code; got '$mask'." >&2
    exit 1
  fi

  local tmp="${hemco_rc}.tmp"

  # Strip CRLF if present; only edit matching livestock FAUBI lines
  tr -d '\r' < "$hemco_rc" | \
    awk -v lv="$livestock" -v m="$mask" '
      {
        line=$0
        n=split($0, f, /[ \t]+/)
        # token2 like FAUBI_..._<lv>
        if (n>=2 && f[2] ~ /^FAUBI_/ && f[2] ~ ("_" lv "$")) {
          gsub(/\<1011\>/, m, line)  # token-only replacement
        }
        print line
      }
    ' > "$tmp"

  mv "$tmp" "$hemco_rc"
}

comment_faubi_lines_for_livestock() {
  # In HEMCO_Config.rc: any line starting with "0 FAUBI_*_{livestock} ..."
  # becomes "#0 FAUBI_*_{livestock} ..."
  local hemco_rc="$1"
  local livestock="$2"

  if [[ ! -f "$hemco_rc" ]]; then
    echo "ERROR: Expected file for HEMCO rc, got: $hemco_rc" >&2
    exit 1
  fi

  local tmp="${hemco_rc}.tmp"

  tr -d '\r' < "$hemco_rc" | \
    awk -v lv="$livestock" '
      BEGIN { changed=0 }
      {
        line=$0
        n=split($0, f, /[ \t]+/)
        # token 1 == 0, token 2 begins with FAUBI_, and token 2 ends with _<lv>
        if (n>=2 && f[1]=="0" && f[2] ~ /^FAUBI_/ && f[2] ~ ("_" lv "$")) {
          sub(/^[[:space:]]*0/, "#0", line)
          changed++
        }
        print line
      }
      END {
        if (changed < 1) exit 42
      }
    ' > "$tmp" || {
      rc=$?
      rm -f "$tmp"
      if [[ $rc -eq 42 ]]; then
        echo "ERROR: No FAUBI lines matched for LIVESTOCK='${livestock}' in ${hemco_rc}" >&2
        echo "HINT: Expected lines like: 0 FAUBI_*_${livestock} ..." >&2
        echo "      Try: grep -n \"FAUBI_\" \"${hemco_rc}\" | head" >&2
      else
        echo "ERROR: awk rewrite failed for ${hemco_rc}" >&2
      fi
      exit 1
    }

  mv "$tmp" "$hemco_rc"
}

# ---------------- Preflight ----------------
if [[ ! -d "$TEMPLATE_DIR" ]]; then
  echo "ERROR: Template dir not found: $TEMPLATE_DIR" >&2
  exit 1
fi

if [[ ! -f "${TEMPLATE_DIR}/${HEMCO_RC_REL}" ]]; then
  echo "ERROR: Missing ${HEMCO_RC_REL} in template: ${TEMPLATE_DIR}/${HEMCO_RC_REL}" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST_CSV" ]]; then
  echo "ERROR: Missing manifest CSV: $MANIFEST_CSV" >&2
  exit 1
fi

mkdir -p "$ENSEMBLE_DIR"

echo ">> BASE:         $BASE"
echo ">> TEMPLATE_DIR: $TEMPLATE_DIR"
echo ">> ENSEMBLE_DIR: $ENSEMBLE_DIR"
echo ">> MANIFEST_CSV: $MANIFEST_CSV"
echo ">> HEMCO_RC_REL: $HEMCO_RC_REL"
echo

# ---------------- Build from CSV ----------------
# Skip header row; simple CSV parsing (no embedded commas in fields).
rownum=0
tail -n +2 "$MANIFEST_CSV" | while IFS=, read -r RUNID LIVESTOCK MASK COUNTRY REST; do
  [[ -z "${RUNID//[[:space:]]/}" ]] && continue

  RUNID="$(trim_field "${RUNID:-}")"
  LIVESTOCK="$(trim_field "${LIVESTOCK:-}")"
  MASK="$(trim_field "${MASK:-}")"
  COUNTRY="$(trim_field "${COUNTRY:-}")"

  if [[ -z "$RUNID" || -z "$LIVESTOCK" || -z "$MASK" || -z "$COUNTRY" ]]; then
    echo "ERROR: Missing required field(s) in CSV row: RUNID='$RUNID' LIVESTOCK='$LIVESTOCK' MASK='$MASK' COUNTRY='$COUNTRY'" >&2
    exit 1
  fi

  # Ensure MASK is exactly 4 digits (zero-pad if numeric)
  if [[ "$MASK" =~ ^[0-9]+$ ]]; then
    MASK="$(printf "%04d" "$MASK")"
  fi
  if [[ ! "$MASK" =~ ^[0-9]{4}$ ]]; then
    echo "ERROR: MASK must be a 4-digit code (got '$MASK') for RUNID='$RUNID'." >&2
    exit 1
  fi

  rownum=$((rownum + 1))
  dir_name="${RUNID}_${LIVESTOCK}_${COUNTRY}"
  run_dir="${ENSEMBLE_DIR}/${dir_name}"

  echo ">>> [Row ${rownum}] Creating ${run_dir}"
  [[ -d "$run_dir" ]] && rm -rf "$run_dir"
  copy_template "$TEMPLATE_DIR" "$run_dir"

  hemco_rc="${run_dir}/${HEMCO_RC_REL}"

  # Ensure HEMCO_Config.rc is a file
  if [[ -d "$hemco_rc" ]]; then
    echo "ERROR: hemco_rc points to a directory, not a file: $hemco_rc" >&2
    echo "DEBUG: run_dir=$run_dir" >&2
    echo "DEBUG: HEMCO_RC_REL='$HEMCO_RC_REL'" >&2
    echo "DEBUG: Contents of run_dir:" >&2
    ls -la "$run_dir" >&2
    exit 1
  fi
  if [[ ! -f "$hemco_rc" ]]; then
    echo "ERROR: ${HEMCO_RC_REL} not found at: $hemco_rc" >&2
    echo "DEBUG: Contents of run_dir:" >&2
    ls -la "$run_dir" >&2
    exit 1
  fi

  # (1) Replace 1011 -> MASK ONLY for matching FAUBI_*_<LIVESTOCK> lines
  replace_mask_only_for_livestock_lines "$hemco_rc" "$LIVESTOCK" "$MASK"

  # (2) Comment out FAUBI lines for this LIVESTOCK (0 -> #0)
  #comment_faubi_lines_for_livestock "$hemco_rc" "$LIVESTOCK"

  # Quick verification: ensure the targeted livestock FAUBI lines contain MASK (optional)
  # grep -n "FAUBI_.*_${LIVESTOCK}.*${MASK}" "$hemco_rc" >/dev/null || {
  #   echo "WARNING: Did not find MASK=${MASK} on FAUBI_*_${LIVESTOCK} lines in $hemco_rc" >&2
  # }

  echo "    - Updated MASK=${MASK} on FAUBI_*_${LIVESTOCK} lines only"
  echo "    - Commented FAUBI_*_${LIVESTOCK} lines in ${HEMCO_RC_REL}"
done

echo
echo "SUCCESS: Created country ensemble directories under: ${ENSEMBLE_DIR}"
