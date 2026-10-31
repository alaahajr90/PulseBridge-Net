#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/hajr/mcd_rppg/code_HR_RR_model/PulseBridge_Net"
ORIG="$ROOT/Stage_2_rPPG_FineTuning"
DST="$ROOT/Stage_2_rPPG_FineTuning_v4_MorphologyPreserving"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$ORIG" ]]; then
  echo "ERROR: original Stage2 folder not found: $ORIG"
  exit 2
fi
if [[ -e "$DST" ]]; then
  echo "ERROR: destination already exists: $DST"
  echo "Refusing to overwrite."
  exit 3
fi

mkdir -p "$DST"
cp -a "$SRC"/*.py "$SRC"/*.sh "$SRC"/README.md "$SRC"/PACKAGE_MANIFEST.json "$DST"/
chmod +x "$DST"/*.sh
mkdir -p "$DST/results"

echo "Installed: $DST"
echo "Original preserved: $ORIG"
echo "No file inside original Stage2 was modified."
echo "NEXT:"
echo "  cd \"$DST\""
echo "  python -u inspect_stage2.py"
