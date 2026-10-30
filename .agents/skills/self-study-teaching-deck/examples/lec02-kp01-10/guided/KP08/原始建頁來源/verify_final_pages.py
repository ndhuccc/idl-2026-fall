from pathlib import Path
import json,io
from PIL import Image
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).parent;OUT=ROOT/'output'
results=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
    for lang in ('zh','en'):
        for ww,hh in ((1280,720),(1920,1080)):
            page=browser.new_page(viewport={'width':ww,'height':hh})
            errors=[];requests=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('request',lambda r:requests.append(r.url) if r.url.startswith(('http:','https:')) else None)
            stem='lec02-kp08-adaline-and-logistic-regression'+('-en' if lang=='en' else '')
            page.set_content((OUT/(stem+'.html')).read_text(),wait_until='load');page.wait_for_timeout(100)
            doc=json.loads((OUT/(stem+'.jbd.json')).read_text())
            wanted={j for pg in json.loads((OUT/(lang+'-pages.json')).read_text()) if pg['page'] in [2,20,27,28,34] for j in range(pg['first_step'],pg['final_step']+1)}
            for i,step in enumerate(doc['steps']):
                if i not in wanted:continue
                page.evaluate('(i)=>go(i,true)',i)
                if i>0:
                    ids=[x['nodeId'] for x in step['reveal']['enter']]
                    invisible=page.evaluate('''ids=>{const es=[...document.querySelectorAll('.layer.on [data-n]')];return ids.filter(id=>{const found=es.filter(e=>e.dataset.n==id);return !found.length||found.some(e=>getComputedStyle(e).opacity!='1')})}''',ids)
                    exited=step['reveal'].get('exitNodeIds',[])
                    visible_exits=page.evaluate('''ids=>[...document.querySelectorAll('.layer.on [data-n]')].filter(e=>ids.includes(e.dataset.n)&&getComputedStyle(e).opacity!='0').map(e=>e.dataset.n)''',exited)
                    assert not visible_exits,(lang,ww,i,visible_exits)
                    assert not invisible,(lang,ww,i,invisible)
            katex=page.locator('.katex-error').count();assert katex==0
            assert not errors,errors
            assert not requests,requests
            results.append(dict(lang=lang,viewport=[ww,hh],steps=len(wanted),reveal_nodes_checked=sum(len(s.get('reveal',{}).get('enter',[])) for j,s in enumerate(doc['steps']) if j in wanted),katex_errors=katex,page_errors=errors,external_requests=requests))
            page.close()
    browser.close()
(OUT/'final-page-verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print('PASS:',len(results),'language/viewport combinations;',sum(r['steps'] for r in results),'steps, reveal visibility, offline loading and KaTeX checked.')
