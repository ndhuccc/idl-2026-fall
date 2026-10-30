from pathlib import Path
import json
from playwright.sync_api import sync_playwright
R=Path(__file__).parent;O=R/'output';results=[]
with sync_playwright() as pw:
 b=pw.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 for lang in ['zh','en']:
  for w,h in [(1280,720),(1920,1080)]:
   p=b.new_page(viewport={'width':w,'height':h});errors=[];requests=[]
   p.on('pageerror',lambda e:errors.append(str(e)));p.on('request',lambda r:requests.append(r.url) if r.url.startswith(('http:','https:')) else None)
   stem='lec02-kp07-loss-functions'+('-en' if lang=='en' else '');d=json.loads((O/(stem+'.jbd.json')).read_text());p.set_content((O/(stem+'.html')).read_text(),wait_until='load')
   m=next(v for v in json.loads((O/(lang+'-pages.json')).read_text()) if v['page']==6)
   for i in range(m['first_step'],m['final_step']+1):
    p.evaluate('(i)=>go(i,true)',i);ids=[e['nodeId'] for e in d['steps'][i]['reveal']['enter']]
    bad=p.evaluate("ids=>ids.filter(id=>!document.querySelector('.layer.on [data-n=\"'+id+'\"]')||getComputedStyle(document.querySelector('.layer.on [data-n=\"'+id+'\"]')).opacity!='1')",ids);assert not bad,bad
   assert p.locator('.katex-error').count()==0
   assert not errors and not requests
   target=next(n for n in d['nodes'].values() if n['id'].startswith(lang+'-p06-') and 'h_1(1)=' in n.get('text',''))
   assert p.locator('.layer.on [data-n="'+target['id']+'"]').count()==1
   results.append(dict(language=lang,viewport=[w,h],page=6,steps=3,reveal_nodes='all visible',math_errors=0,page_errors=errors,external_requests=requests));p.close()
 b.close()
(O/'最終算式播放驗證.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print('PASS: final Huber substitution in 4 viewports/languages; 12 reveal steps.')
