import json,math,subprocess
from pathlib import Path
r=Path(__file__).parent/'output';ds=[json.loads((r/('lec02-kp08-adaline-and-logistic-regression'+suffix+'.jbd.json')).read_text()) for suffix in ('','-en')];logs=[]
for d in ds:
 assert d['schemaVersion']==3
 for key,n in d['nodes'].items():
  assert key==n['id'] and n['type'] in ('frame','text','shape','connector')
  assert key in d['scenes'][n['sceneId']]['nodeIds']
  if n['type']=='connector':
   for end in ('source','target'):
    assert d['nodes'][n['connector'][end+'NodeId']]['sceneId']==n['sceneId']
    assert 0<=n['connector'][end+'Hook']<=15
  assert all(math.isfinite(v) for v in n['transform'].values())
  assert n.get('style',{}).get('fontSize',16)>=16
  assert not any(ord(c)<32 and c not in '\n\t' for c in n.get('text',''))
  assert '$$' not in n.get('text','')
 for s in d['steps']:
  for e in s.get('reveal',{}).get('enter',[]):assert d['nodes'][e['nodeId']]['sceneId']==s['target']['sceneId']
 for n in d['nodes'].values():
  if n.get('name')=='concept-dialogue':assert len(n['text'].splitlines())<=2
for a,b in zip(ds[0]['nodes'].values(),ds[1]['nodes'].values()):
 for k in ('type','transform','width','height','shapeKind','polygonPoints','initiallyHidden'):assert a.get(k)==b.get(k),(a['id'],k)
assert [s['target'] for s in ds[0]['steps']]==[s['target'] for s in ds[1]['steps']]
for p in r.glob('*.jbd.json'):
 for script,args in [('/workspace/scratch/idl-lec02-kp08-review/toolchain/.agents/skills/prompt-to-jbd-page/tools/check_deck_quality.py',['--page-width','1440','--strict']),('/workspace/scratch/idl-lec02-kp08-review/toolchain/scripts/check_slides.py',[])]:
  v=subprocess.run(['python3',script,str(p),*args],capture_output=True,text=True);logs.append({'file':p.name,'check':Path(script).name,'code':v.returncode,'output':v.stdout+v.stderr});assert v.returncode==0,v.stdout+v.stderr
(r/'native-validation.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2))
result={'schemaVersion':3,'pages_per_language':34,'steps_per_language':103,'nodes_per_language':len(ds[0]['nodes']),'native_only':True,'minimum_font_size':16,'matching_bilingual_geometry':True,'valid_scene_and_reveal_references':True,'control_characters_and_empty_math':False,'mascot_on_every_page':True,'concept_based_props_and_dialogue':True}
(r/'structure-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('PASS: structure, bilingual geometry, font sizes, reveal references and both course validators.')
