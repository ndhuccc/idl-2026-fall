from pathlib import Path
import json,hashlib
from specs import PAGES,MATH
R=Path(__file__).parent;m=json.loads((R/'skills-source-manifest.json').read_text())
for f in m['files']:
 p=R/'toolchain'/f['path'];b=p.read_bytes();sha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert sha==f['sha'],str(p)
rows=[];out=[]
for i,p in enumerate(PAGES,1):
 kind,zh,en,tz,te,sec,visual,prop,mood,sz,se=p
 rows.append(dict(page=i,source_section=sec,title=zh,takeaway=tz,visual=visual,math=MATH.get(kind,[]),extension=i>23))
 out.append(f'''## 第{i}頁／共{len(PAGES)}頁：{zh}
用途：IDL第二章KP08，大學生；本頁唯一結論：{tz}。約1分鐘。來源：ndhuccc/idl-2026-fall @ {m['commit']}，lectures/lec02-kp08-adaline-and-logistic-regression.md，{sec}。
規格：1280×720，白底深字；標題0–86，主圖60–1000×170–620，角色與泡泡1025–1255×162–560，結論640–696。中文約60字、英文約35詞；必要公式不可刪，條件同頁可見。
逐區內容：標題「{zh}」／「{en}」。主視覺：{visual}。結論「{tz}」／「{te}」。泡泡「{sz}」／「{se}」。固定藍頭紫身橙觸角機器人，表情{mood}，拿{prop}道具；比喻與實算資料區分。
數學排版：以下逐條原樣用KaTeX獨立節點，不把LaTeX指令直接印出；分式／總和預留110px高。完整公式：{'；'.join('$'+s+'$' for s in MATH.get(kind,[])) or '純概念示意；無新增數值計算'}。
原生元件：精算函數折線每節點不超200點，幾何大小按數值；流程一律connector；不嵌入圖片。目標紫、預測藍、下降綠、似然橙，條件橙；圖形最大、文字只做標註。星形標接點、三角形警示、菱形判斷；漸層、陰影與半透明只輔助聚焦。
三拍：先問題／主圖，後新證據與角色，最後結論；同步講者稿，練習與檢核只給提示。上一頁問題銜接本頁，下一頁延伸本頁結論。
避免：牛頓法最大化一律加號、混用LMS負梯度與loglikelihood正梯度、非可分就必有有限MLE、最少分類錯誤等同最大似然、ADALINE輸出當機率、把一般Perceptron更新說成對0-1求梯度。
檢查：數值完整中間精度後取位數；軸內曲線；所有顯示結果有同頁算式；中英幾何一致；至少三輪渲染目視與修正，不只通過程式。
''')
(R/'圖解提示詞.md').write_text('\n'.join(out));(R/'圖解證據表.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print('Verified',len(m['files']),'Git blobs;',len(PAGES),'page specifications saved.')
