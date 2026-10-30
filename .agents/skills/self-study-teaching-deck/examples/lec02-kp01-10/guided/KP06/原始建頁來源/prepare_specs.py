from pathlib import Path
import json,hashlib
from specs import PAGES
r=Path(__file__).parent
manifest=json.loads((r/'skills-source-manifest.json').read_text())
for f in manifest['files']:
 p=r/'toolchain'/f['path'];b=p.read_bytes();sha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
 assert sha==f['sha'],str(p)
rows=[];prompts=[]
for i,p in enumerate(PAGES,1):
 kind,zh,en,tz,te,sec,brief,prop,mood,sz,se=p
 rows.append(dict(page=i,source='lectures/lec02-kp06-activation-functions.md',section=sec,question=zh,takeaway=tz,visual=brief,role='source + explicit clarification' if kind in ['gaussian','gradient','elu','prelu','maxout','plus','output'] else 'source',extension=i>24))
 prompts.append(f'''## 第{i}頁／共{len(PAGES)}頁：{zh}
用途：大學部 IDL KP06 教學；本頁唯一訊息：{tz}。估時1分鐘；延伸題由學生停下計算。
來源：ndhuccc/idl-2026-fall，commit {manifest['commit']}，lectures/lec02-kp06-activation-functions.md，{sec}。以 source-lecture.md 全文為準。
規格：1280×720，白底；標題上方0–86，主圖60–1000×170–605，右側1025–1255為原生機器人與泡泡；底部640–696一句結論。中文字60字左右，英文35詞左右；必要公式與條件不可刪。
逐區內容：標題「{zh}」／「{en}」。主圖：{brief}。結論「{tz}」／「{te}」。泡泡「{sz}」／「{se}」。藍頭紫身橙觸角機器人持{prop}道具、表情{mood}，只作教學比喻。
原生元件：函數用精算折線且每節點不超200點；區域以半透明形狀；流程用connector；不得image節點。軸範圍完整包住曲線，精確比例；座標標籤不重疊。藍為函數，紫虛線為導數，綠為結論，橙為條件警訊。
數學排版：所有數學用獨立KaTeX文字節點，公式由講義該節完整定義與建頁腳本 FORMULAS 對照表逐條生成；分式框至少110px。值域與參數條件必須同頁，原點不可微不寫成真導數。Softmax分母保留所有項。
漸進出場：第一拍看輸入與主曲線；第二拍揭示導數／比較與角色；第三拍帶走結論，附同步講者稿。額外反例標「補充」，延伸與小測驗備註只給提示。
避免：Gaussian畫成S形、Maxout對輸入導數誤寫1、無條件保證梯度不消失、把Softplus有限正側斜率寫1；PReLU與ELU值域須說明正參數假設；不要把機率範圍當成校準保證。
檢查：讀圖5秒可辨訊息；數值由數值腳本驗證；中英位置完全一致；字級至少16；三輪真實瀏覽器渲染、目視、修改。
''')
(r/'圖解證據表.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
(r/'圖解提示詞.md').write_text('\n'.join(prompts))
print('28 skill resources verified; 31 page specifications written before drawing.')
