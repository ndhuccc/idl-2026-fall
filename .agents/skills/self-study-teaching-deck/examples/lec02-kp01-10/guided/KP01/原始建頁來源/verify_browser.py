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
            stem='lec02-kp01-biological-neuron-and-mp-unit'+('-en' if lang=='en' else '')
            page.set_content((OUT/(stem+'.html')).read_text(),wait_until='load');page.wait_for_timeout(100)
            doc=json.loads((OUT/(stem+'.jbd.json')).read_text())
            for i,step in enumerate(doc['steps']):
                page.evaluate('(i)=>go(i,true)',i)
                if i>0:
                    ids=[x['nodeId'] for x in step['reveal']['enter']]
                    invisible=page.evaluate('''ids=>ids.filter(id=>{const e=[...document.querySelectorAll('[data-n]')].find(e=>e.dataset.n==id&&e.closest('.layer.on'));return !e||getComputedStyle(e).opacity!='1'})''',ids)
                    assert not invisible,(lang,ww,i,invisible)
            katex=page.locator('.katex-error').count();assert katex==0
            assert not errors,errors
            assert not requests,requests
            results.append(dict(lang=lang,viewport=[ww,hh],steps=len(doc['steps']),reveal_nodes_checked=sum(len(s.get('reveal',{}).get('enter',[])) for s in doc['steps']),katex_errors=katex,page_errors=errors,external_requests=requests))
            if lang=='zh' and ww==1280:
                page.add_style_tag(content='#bar{display:none!important} #notes{display:none!important} #stagewrap{inset:0!important}')
                page.evaluate('fit()');page.evaluate('go(15,true)') # page 8, first beat
                stage=page.locator('#stage');frames=[Image.open(io.BytesIO(stage.screenshot())).convert('RGB')]
                page.keyboard.press('ArrowRight')
                for delay in (20,45,70,130):
                    page.wait_for_timeout(delay);frames.append(Image.open(io.BytesIO(stage.screenshot())).convert('RGB'))
                frames.append(frames[-1].copy())
                frames[0].save(OUT/'漸進出場示例.gif',save_all=True,append_images=frames[1:],duration=[1100,70,70,100,150,1500],loop=0)
            page.close()
    browser.close()
(OUT/'browser-verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print('PASS:',len(results),'language/viewport combinations; all 164 steps, reveal visibility, offline loading and KaTeX checked.')
