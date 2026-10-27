#!/bin/bash
# usage: run_unit.sh <unit_dir> <deck_script_basename (no .py)>      e.g.  MIN=40 bash run_unit.sh ./kp18 kp18_deck
# Builds the zh and en decks of one unit (fit -> audit -> preview -> bottoms) inside <unit_dir>/W_zh and W_en. Set MIN to the lecture minutes.
# Resolve the course before changing into an external unit directory.
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 1
DEFAULT_REPO=$(cd "$TOOL_DIR/../../../.." && pwd) || exit 1
COURSE_REPO=$(cd "${COURSE_REPO:-$DEFAULT_REPO}" && pwd) || exit 1
export COURSE_REPO
SK="$COURSE_REPO/.agents/skills"
export COURSE_SKILLS="$SK"
export SKILLS_DIR="$SK"  # compatibility with existing unit scripts
KP=$1; DK=$2; cd "$KP" || exit 1
for f in *_text.py $DK.py; do python3 -c "import ast,sys;ast.parse(open('$f').read())" || { echo "SYNTAX ERROR in $f"; exit 1; }; done
export KATEX_DIR="${KATEX_DIR:-$SK/slides-to-html/node_modules/katex/dist}"
T=$SK/view-from-text/tools
for L in zh en; do
  mkdir -p W_$L; [ -f W_$L/figs.json ] || echo '{}' > W_$L/figs.json
  cp "$DK.py" *_text.py deckcommon.py W_$L/ || exit 1
  if [ -f course_context.py ]; then cp course_context.py W_$L/ || exit 1; fi
  (cd W_$L && rm -f need.json && KP_LANG=$L DECK_WORK=$PWD python3 -W ignore "$T/deckkit/fit.py" . $DK.py 2>&1 | tail -3)
  TF=terms.json; [ $L = en ] && TF=terms_en.json
  echo "#### $L"
  python3 -W ignore "$T/audit_deck.py" W_$L/draft.jbd.json --monolingual --speaker-notes --level undergrad --narrative --terms $TF --minutes ${MIN:-40} --kps 1 2>&1 | grep -E "FAIL|WARN" | cut -c1-200
  rm -rf W_$L/prev; python3 "$T/preview_deck.py" W_$L/draft.jbd.json --out W_$L/prev --sheets 2>&1 | grep overflow
  python3 - $L <<'PY'
import json,sys
d=json.load(open(f'W_{sys.argv[1]}/draft.jbd.json'))
st=[s for s in d['steps'] if s['target']['sceneId']!='scene-root']
r=[]
for i,s in enumerate(st,1):
    x0=s['target']['point']['x']-640
    b=max((n['transform']['y']+n['height']) for n in d['nodes'].values() if n['type'] in('text','shape','image') and x0-5<=n['transform']['x']<x0+1290 and n['sceneId']==s['target']['sceneId'])
    r.append(f"{i}:{round(b)}")
print('bottoms', ' '.join(r))
PY
done
echo "screenshots: W_zh/prev/sheet_*.png  W_en/prev/sheet_*.png   (overflow.json lists boxes whose text does not fit)"
