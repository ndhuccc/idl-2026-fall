from pathlib import Path
import json,sys
from playwright.sync_api import sync_playwright
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).parent;OUT=ROOT/'output';ROUND=sys.argv[1] if len(sys.argv)>1 else 'round1'
dest=ROOT/ROUND;dest.mkdir(exist_ok=True)
report=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
    errors=[]
    for lang in ('zh','en'):
        page=browser.new_page(viewport={'width':1280,'height':720},device_scale_factor=1)
        page.on('pageerror',lambda e:errors.append(str(e)))
        fn='lec02-kp04-perceptron-limits-and-generalization'+('-en' if lang=='en' else '')+'.html'
        manifest=json.loads((OUT/(lang+'-pages.json')).read_text())
        page.set_content((OUT/fn).read_text(),wait_until='load');page.wait_for_timeout(200)
        # Inspect the generated offline HTML, including the real layer and reveal order.
        page.add_style_tag(content='#bar{display:none!important} #notes{display:none!important} #stagewrap{inset:0!important} *{transition:none!important;animation:none!important}')
        page.evaluate('fit()')
        for item in manifest:
            page.evaluate('(i)=>go(i,true)',item['final_step']);page.wait_for_timeout(40)
            page.evaluate('document.fonts.ready');page.wait_for_timeout(30)
            prefix=f'{lang}-p{item["page"]:02d}-'
            bad=page.evaluate('''prefix=>[...document.querySelectorAll('[data-n]')].filter(e=>e.dataset.n.startsWith(prefix)&&e.classList.contains('text')&&!e.classList.contains('h')).filter(e=>e.querySelector('.katex-error')||e.scrollHeight>e.clientHeight+3||e.scrollWidth>e.clientWidth+3).map(e=>({id:e.dataset.n,text:e.innerText.slice(0,110),client:[e.clientWidth,e.clientHeight],scroll:[e.scrollWidth,e.scrollHeight]}))''',prefix)
            box=page.locator('#stagewrap').bounding_box()
            page.screenshot(path=str(dest/f'{lang}-{item["page"]:02d}.png'),clip=box)
            report.append(dict(lang=lang,page=item['page'],overflow=bad))
        page.close()
    browser.close()
(dest/'browser-audit.json').write_text(json.dumps(dict(pages=report,page_errors=errors),ensure_ascii=False,indent=1))
for lang in ('zh','en'):
    files=sorted(dest.glob(lang+'-*.png'))
    for start in range(0,len(files),5):
        chunk=files[start:start+5];sheet=Image.new('RGB',(1280,len(files[start:start+5])*400),'#d7e0e7');draw=ImageDraw.Draw(sheet)
        for row,f in enumerate(chunk):
            im=Image.open(f);im.thumbnail((1216,362));sheet.paste(im,((1280-im.width)//2,row*400+28));draw.text((18,row*400+6),f.stem,fill='#17324d')
        sheet.save(dest/f'{lang}-sheet-{start//5+1}.png')
print(ROUND,':',len(report),'pages;',sum(len(p['overflow']) for p in report),'overflow candidates;',len(errors),'page errors')
