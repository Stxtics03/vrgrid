#!/usr/bin/env bash
# One command per demo scene. Run from anywhere; it finds the repo itself.
#
#   ./scripts/demo.sh check              preflight -- venv, data, rerun, Patchwork++
#   ./scripts/demo.sh bake               pre-render every scene to demo/*.rrd (do this
#                                        BEFORE you present; ~3 min, ~1 GB)
#   ./scripts/demo.sh <scene>            play a scene -- baked recording if one exists,
#                                        otherwise computes it live
#   ./scripts/demo.sh <scene> --live     force the live pipeline even if baked
#   ./scripts/demo.sh numbers            print the report tables to the terminal
#   ./scripts/demo.sh list               list the scenes
#
# Scenes, in the order docs/demo-runbook.md presents them:
#   foveation     seq 00, 0-160    rings + blind cone + the map filling in
#   ghosts-off    seq 00, 0-60     ghost removal DISABLED -- trails stay in the cells
#   ghosts-on     seq 00, 0-60     ghost removal ON -- the same frames, trails gone
#   traffic       seq 07, 650-700  dense moving traffic, colour by motion
#   reflectivity  seq 00, 4420-60  lane paint, colour by reflectivity
#   dense3d       seq 00, 0-60     our grid beside a uniform 5 cm dense voxel grid
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="$REPO/.venv/bin/python"
BAKE_DIR="$REPO/demo"

# The loader wants the directory holding poses/ and sequences/. In this clone
# that is data/dataset, NOT data -- pointing it one level up is the single most
# likely way to lose the demo, so it is resolved here and never typed by hand.
if [[ -z "${VRGRID_DATA_ROOT:-}" ]]; then
  if   [[ -d "$REPO/data/dataset/poses" ]]; then export VRGRID_DATA_ROOT="$REPO/data/dataset"
  elif [[ -d "$REPO/data/poses"         ]]; then export VRGRID_DATA_ROOT="$REPO/data"
  fi
fi

die() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
say() { printf '\033[36m%s\033[0m\n' "$*"; }

# scene -> the vrgrid.dash arguments that draw it
scene_args() {
  case "$1" in
    foveation)    echo "--seq 00 --frames 160 --color-by class" ;;
    ghosts-off)   echo "--seq 00 --frames 60 --color-by motion --show-ghosts" ;;
    ghosts-on)    echo "--seq 00 --frames 60 --color-by motion" ;;
    traffic)      echo "--seq 07 --start-frame 650 --frames 50 --color-by motion" ;;
    reflectivity) echo "--seq 00 --start-frame 4420 --frames 40 --color-by reflectivity" ;;
    dense3d)      echo "DENSE3D" ;;
    *)            return 1 ;;
  esac
}
SCENES="foveation ghosts-off ghosts-on traffic reflectivity dense3d"

run_scene() {   # $1 scene, $2 = "--save <path>" or "" for a spawned viewer
  local args; args="$(scene_args "$1")" || die "unknown scene: $1  (try: $SCENES)"
  if [[ "$args" == "DENSE3D" ]]; then
    local mode="--spawn"; [[ -n "$2" ]] && mode="$2"
    # shellcheck disable=SC2086
    "$PY" -m vrgrid.dash.dense3d_comparison --seq 00 --frames 60 $mode
  else
    # shellcheck disable=SC2086
    "$PY" -m vrgrid.dash $args $2
  fi
}

cmd="${1:-help}"; shift || true

case "$cmd" in

check)
  say "== preflight =="
  [[ -x "$PY" ]] || die "no venv at $REPO/.venv -- python3 -m venv .venv && .venv/bin/pip install -e '.[dev,dash,perception]'"
  echo "python        $("$PY" --version 2>&1)"
  "$PY" -c 'import vrgrid' 2>/dev/null && echo "vrgrid        importable" || die "vrgrid not installed -- .venv/bin/pip install -e '.[dev]'"
  "$PY" -c 'import rerun,sys;print("rerun         "+rerun.__version__)' 2>/dev/null || die "rerun missing -- .venv/bin/pip install rerun-sdk"
  "$PY" -c 'from vrgrid.perception import ground;print("patchwork++   "+("yes" if ground._HAVE_PATCHWORKPP else "NO -- ground falls back to the semantic proxy"))'
  echo "data root     ${VRGRID_DATA_ROOT:-UNSET}"
  [[ -n "${VRGRID_DATA_ROOT:-}" && -f "$VRGRID_DATA_ROOT/poses/00.txt" ]] \
    || die "poses not found under VRGRID_DATA_ROOT -- expected \$VRGRID_DATA_ROOT/poses/00.txt"
  echo "display       ${DISPLAY:-none}${WAYLAND_DISPLAY:+ (wayland $WAYLAND_DISPLAY)}"
  if [[ -d "$BAKE_DIR" ]]; then
    echo "baked scenes  $(ls -1 "$BAKE_DIR"/*.rrd 2>/dev/null | wc -l) in demo/"
  else
    echo "baked scenes  none -- run './scripts/demo.sh bake' before you present"
  fi
  say "== the one-frame smoke test =="
  "$PY" -m vrgrid.run --seq 00 --frames 2 2>&1 | grep -v PatchWork | tail -4
  say "OK -- the pipeline runs on real data."
  ;;

bake)
  mkdir -p "$BAKE_DIR"
  targets="${1:-$SCENES}"
  for s in $targets; do
    say "baking $s ..."
    run_scene "$s" "--save $BAKE_DIR/$s.rrd" 2>&1 | grep -v PatchWork | tail -2
  done
  say "baked into $BAKE_DIR:"; ls -lh "$BAKE_DIR"/*.rrd | awk '{print "  "$9"  "$5}'
  echo "Play one with: ./scripts/demo.sh <scene>   (or: rerun demo/<scene>.rrd)"
  ;;

numbers)
  say "== memory =="        ; "$PY" scripts/memory_table.py
  say "== ring schedule ==" ; "$PY" scripts/sampling_table.py 2>&1 | tail -40
  echo
  echo "Latency (takes ~2 min on real data):  .venv/bin/python scripts/timing_table.py --seq 08"
  echo "Ghost figure:                         .venv/bin/python scripts/ghost_removal_figure.py --seq 08"
  ;;

list) echo "$SCENES" | tr ' ' '\n' ;;

help|-h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//' ;;

*)
  scene_args "$cmd" >/dev/null || die "unknown command or scene: $cmd  (try: check | bake | numbers | list | $SCENES)"
  live=0; [[ "${1:-}" == "--live" ]] && live=1
  baked="$BAKE_DIR/$cmd.rrd"
  if [[ $live -eq 0 && -f "$baked" ]]; then
    say "playing baked recording $baked  (scrubbable, no compute)"
    if   [[ -x "$REPO/.venv/bin/rerun" ]]; then "$REPO/.venv/bin/rerun" "$baked"
    elif command -v rerun >/dev/null 2>&1;  then rerun "$baked"
    else die "no rerun viewer found -- pip install rerun-sdk, or open $baked on a machine that has it"; fi
  else
    say "computing $cmd live ..."
    run_scene "$cmd" ""
  fi
  ;;
esac
