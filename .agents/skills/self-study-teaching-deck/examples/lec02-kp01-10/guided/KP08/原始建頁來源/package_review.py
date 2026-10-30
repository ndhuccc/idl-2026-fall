from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,shutil,zipfile
from PIL import Image,ImageDraw
R=Path(__file__).parent;O=R/'output';now=datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M');m=json.loads((R/'skills-source-manifest.json').read_text());nums=json.loads((R/'數值驗算.json').read_text())
for name,sha in [('source-lecture.md','752af1557a46b1ed721faf4907196c1abd7827a7'),('source-figures.py','ef5f0a3058d3eda68ae6bc3de88e362f5cba5d8c')]:
 p=R/name;raw=p.read_bytes()
 for b in [raw,raw[:-1],raw+b'\n']:
  if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==sha:p.write_bytes(b);break
 else:raise AssertionError(name+' source SHA mismatch')
report=f'''# IDL 第二章 KP08：ADALINE 與邏輯回歸審閱稿

完成查核：{now}（Asia/Taipei）。本機審閱草稿，尚未推送或發布；原講義與 KP01–KP07 未更動。

## 直接審閱

- 中文：lec02-kp08-adaline-and-logistic-regression.html。
- 英文：lec02-kp08-adaline-and-logistic-regression-en.html。
- 同名 .jbd.json 為可編輯原生簡報，不含 image 節點。
- 中英各34頁、103步（含根預覽步驟）、1977個原生節點、34條原生連接線，每頁三拍。兩語幾何、鏡頭與步驟一致。
- 方向鍵／空白鍵逐拍前進，左方向鍵回看，N 看講者稿，F 全螢幕。根步驟是整份縮圖，下一步進第一頁。
- 漸進出場示例.gif 是第16頁實際鍵盤播放擷取：一般牛頓式、代入負Hessian後的新式、帶走結論。問題提示第二拍退場，鏡頭略推近後回全景。

主線23頁，接3頁練習與講義原有8頁自我檢核。這是展開的圖解版，主線約25–30分鐘，練習另計。若維持講義建議15分鐘，可把第10–11、15、18–22頁列為選讀，課堂聚焦兩種模型、OR第一步與牛頓符號；完整推導及算例仍保留供課後閱讀。

## 最新技能與來源

課程 ndhuccc/idl-2026-fall，當次最新 main commit：`e494ab9dc2c18c139b4cb930118a4207d12de513`。完整遞迴樹未截斷，發現14項技能。沿用本對話已由教授選定的 page-to-teaching-diagram → prompt-to-jbd-page，使用 image-to-jbd-page 原生庫及 slides-to-html。相關四項技能28資源加課程check_slides共29個Git blob逐一核對，與KP07技能內容一致。沒有反向修改GitHub技能。

講義：lectures/lec02-kp08-adaline-and-logistic-regression.md，blob `752af1557a46b1ed721faf4907196c1abd7827a7`。圖程式：assets/figures/src/lec02-kp0810-figures.py，blob `ef5f0a3058d3eda68ae6bc3de88e362f5cba5d8c`。兩份本機原始快照的Git blob SHA亦核對相符。來源清單見skills-source-manifest.json。

課程上已有另一批第二章與第三章簡報更新，本稿獨立存放供審閱，沒有覆蓋既有內容。建頁腳本與完整工具副本在「建頁來源」；直接手改JSON會在重建時被覆蓋，請同步修改建頁程式。

## 小節與頁次

| 講義範圍 | 頁次 | 圖解 |
|---|---|---|
| 前言、§1 模型與LMS | 1–3 | 兩條學習路徑、訓練／推論分流、平方損失與梯度 |
| §1 OR第一批 | 4–7 | 四點資料、初始殘差、三個完整梯度、更新、重新計損失 |
| §1 最小平方與原圖 | 8–10 | 四個連續輸出、門檻、兩條精確邊界、十二次迭代 |
| §1 MADALINE | 11 | 講義簡化沿革與訓練範圍示意 |
| §2 模型與最大似然 | 12–14 | Sigmoid條件機率、Bernoulli似然、與BCE等價、更新方向對照 |
| §2 Hessian與牛頓法 | 15–16 | 欄樣本矩陣、負半定性、一般式到代入後加號 |
| §2 牛頓手算 | 17–19 | 原四筆一維資料、殘差與曲率、逆矩陣和參數移動 |
| §2 原曲線與結果、選讀 | 20–22 | 前兩步精算曲線與似然、兩種分類切點、可分與有限解條件 |
| 銜接KP09 | 23 | 局部基函數與無標籤學習 |
| 練習1–3 | 24–26 | 完整保留題意，只給提示 |
| 小測驗1–8 | 27–34 | 保留四個選項，等樣式呈現，不標答案 |

每頁延續藍頭、紫身、橙觸角的小機器人，依概念換門檻開關、直尺、籃子、旋鈕、警示牌與問號牌。輸入／預測藍、目標／中間量紫、下降方向綠、似然與條件橙。曲線、參數與長條來自計算；角色、考牌、分流架構與逐漸變陡的示意曲線不冒充實驗。

## 數值重現

- OR四筆資料為(0,0)、(0,1)、(1,0)、(1,1)，標籤0、1、1、1。設計矩陣欄順序為兩個特徵、常數一，參數順序(w1,w2,w0)；初始零，全批次學習率0.1。
- 第一批梯度(-2,-2,-3)，更新至(0.2,0.2,0.3)；新預測(0.3,0.5,0.5,0.7)，半平方和1.5降到0.34。完整代入與偏置梯度保留。
- 原程式十二次更新逐一重現，共13個真實損失點，使用折線連接，不製造中間迭代。另以最小平方求解得到(0.5,0.5,0.25)，輸出(0.25,0.75,0.75,1.25)，損失0.125。0.5門檻分類全對，但連續誤差不為零，1.25不是機率。
- 本二次問題最大Hessian特徵值約{max(nums['or_hessian_eigenvalues']):.6f}，穩定步長上界2/最大特徵值約{nums['eta_upper_bound']:.6f}；0.1在範圍內。沒有把此下降圖推成任意學習率都會下降。
- 邏輯回歸沿用x=(0,1,2,3)、y=(0,1,0,1)，偏置放第一個參數；X為2×4、每欄一筆。起點p全0.5，梯度(0,1)，A=-H=[[1,1.5],[1.5,3.5]]，行列式1.25。
- 第一牛頓步為(-1.2,0.8)，對數似然由-4ln2約-2.7726升至{nums['newton_trace'][1]['loglikelihood']:.4f}。第二步完整參數約(-1.3559834342,0.9039889561)，似然{nums['newton_trace'][2]['loglikelihood']:.4f}。錯用相反方向則似然為{nums['wrong_sign_loglikelihood']:.4f}，低於起點。
- 曲線沿用原程式負0.5到3.5的300點網格，分別畫起點、第一步與第二步，保持完整中間精度後才顯示近似值。每個折線節點不超181點。
- 收斂解約(-1.361900,0.907933)，四機率約0.204、0.388、0.612、0.796，邊界為1.5。最終梯度接近零；梯度與Hessian另以獨立中央差分核對。完整精度見數值驗算.json，近似顯示以該檔為準。

## 與講義相比的釐清

1. 用E代表LMS損失、r代表目標減預測，避免與KP07把e當殘差混淆。此處是半平方誤差「總和」，不是平均；學習率與梯度的尺度採同一約定。
2. 一般Perceptron更新不是對0–1錯誤數做普通梯度下降，不把「離散錯誤數沒有有效普通梯度」說成「Perceptron不能學」。ADALINE推論門檻也依標籤編碼而定，本例0/1才用0.5。
3. 講義說兩者梯度都是目標減預測乘輸入，但一邊是最小化損失、一邊是最大化似然，符號不同。第14頁明確比較ADALINE負梯度下降方向與loglikelihood正梯度上升方向，才具有相同的殘差形式。
4. 一般牛頓駐點式是beta-H逆g，尋找極大或極小都採這個形式。這裡H是對數似然Hessian，令A=-H=XWX轉置，代入後才得到加A逆g。不能說「最大化的牛頓法一概用加H逆g」。第16頁保留三步對照，第19頁以數字驗證。
5. H負半定表示對數似然凹，負對數似然凸；不自動代表嚴格凹、H可逆或MLE有限。樣本欄／列約定清楚標示；圖中的一般格子只是矩陣結構示意，後續算例明寫實際2×4矩陣。實作宜解A乘delta等於g，未鼓勵真的求逆；必要時阻尼或線搜尋。
6. 似然是固定輸入後，把Bernoulli條件機率乘起來，視為參數的函數；不是參數的貝氏後驗，除非另加先驗。第13頁保留原機率式、兩種對數式與BCE負值關係。
7. 本交錯資料的有限MLE不只因為「不可完全分」，還有滿秩且沒有準完全分離。第22頁與講者稿補充完全／準完全分離、秩及正則化條件；凹性也不保證每個標準牛頓步都上升。
8. 原講義把收斂後兩筆錯誤說成資料所致。此資料確實不能由一個門檻全分對，但MLE邊界1.5給兩錯，改用自變量切點x0=0.5給(0,1,1,1)，只錯一筆。第21頁明示這個補充反例，說明最大似然不是最小0–1錯誤；x0=0.5是輸入座標的切點，不是把它混稱為機率門檻0.5。
9. OR兩條線僅比較此次ADALINE與KP02那次Perceptron，不保證後者每次都停在同一條線，也不把圖示位置當成較好泛化的證據。
10. MADALINE年表沿用講義的簡化訓練範圍與年份，不宣稱做過完整歷史文獻驗證；1988／1990的後續改進不能全說成早於反向傳播普及。ADALINE的1960、Widrow–Hoff及LMS名稱保留在講者稿。
11. logistic曲線歷史用「早期用途」描述講義的Verhulst1845人口成長註記，不把1845宣稱成所有logistic形式的首次發明年。邏輯回歸輸出本身是連續機率，分類是另一步；相對LDA的判別式／生成式方向在第12頁及備註保留。
12. 下一KP的固定局部基函數只是特定RBF設計；一般RBF也可學中心與寬度。所有補充與限制均放在審閱稿、頁面或同步講者稿，原講義未動。

## 視覺與播放驗證

- 三輪真實渲染與目視：第一輪修正梯度配對表與算式重疊；第二輪補參數平面與更新方向、牛頓三步流程及分類目標反例；第三輪重新檢視兩語全部索引，移開機率與座標刻度、補初始損失完整代入、讓原生連線直接接到計算結果。68頁零溢出候選、零頁面錯誤。
- 最後再修整第20頁，讓數值似然的完整計算式與Sigmoid公式同頁；第27、28、34頁的選項小圖改成對應的正確直線、凸二次、Hinge、Sigmoid等示意。兩語這4頁共8張重新渲染並目視，均無溢出。
- check_deck_quality --strict與課程check_slides兩語通過。功能群12/12、每頁活潑效果和圖形數達標、1806個隱藏節點各揭示一次；無高圖層蓋字、空泡泡、空字框。最低字級16，中英幾何與鏡頭一致。
- 中英×1280×720／1920×1080共412個完整步驟驗證進退場、連線、公式與離線載入；最後改動的第2、20、27、28、34頁另在四組視窗驗證60步。無KaTeX／JS錯誤、無外部HTTP請求。見browser-verification.json及final-page-verification.json。
- 最終HTML字型／粗細／對齊另依課程check_html邏輯檢查，僅適配本機Chromium啟動與setContent載入，沒有改上游工具。結果見HTML字型檢查.txt。
- 所有講者稿可隨下載提供；練習及檢核的頁面、備註只給提示，不附選項答案。原講義快照仍保留原有解題示範與來源更正，僅作來源紀錄。
- 檢查器保留102個步驟有備註的提醒，以及18條折線因鄰近小三角而被啟發式標作手畫箭頭的提醒。後者是角色道具／汗滴或資料點三角附近的曲線與線條；所有流程與對應實際使用原生connector，沒有呼叫手畫箭頭函式。沒有修改檢查器壓掉提醒。

尚未操作實際JianBauDau App匯入／播放，請在App中確認最終顯示；未做PDF、完整課程doctor或網站建置。此目錄為審閱工作副本，不是完整課程checkout。\n'''
# Avoid hard-coding more precision than verified source calculations.
bs=nums['newton_trace'][-1]['beta'];report=report.replace('(-1.361900,0.907933)',f'({bs[0]:.6f},{bs[1]:.6f})')
(O/'審閱說明.md').write_text(report)
# Keep the original design plan and append final exact text for portable handoff.
p=R/'圖解提示詞.md';s=p.read_text();d=json.loads((O/'lec02-kp08-adaline-and-logistic-regression.jbd.json').read_text());pages=json.loads((O/'zh-pages.json').read_text());s+='\n# 三輪修訂與最終潤飾後的精確文案\n\n以下與事前規格一起提供。數值、曲線、公式與元件座標可由kp08_nums.py及build_kp08.py完整重建。\n'
for pg in pages:
 pre=f'zh-p{pg["page"]:02d}-';s+=f'\n## 第{pg["page"]}頁\n\n'+'\n\n'.join(n['text'] for n in d['nodes'].values() if n['id'].startswith(pre) and n.get('text'))+'\n'
p.write_text(s)
for name in ['數值驗算.json','圖解證據表.json','skills-source-manifest.json','圖解提示詞.md']:shutil.copy2(R/name,O/name)
shutil.copy2(R/'round3/browser-audit.json',O/'逐頁排版檢查.json');last=[]
shots=O/'預覽截圖';shots.mkdir(exist_ok=True)
for p in (R/'round3').glob('*.png'):shutil.copy2(p,shots/p.name)
for page in [20,27,28,34]:
 folder=R/f'final-polish-{page}';last.append(json.loads((folder/'browser-audit.json').read_text()))
 for lang in ['zh','en']:shutil.copy2(folder/f'{lang}-{page:02d}.png',shots/f'{lang}-{page:02d}.png')
(O/'最後修整排版檢查.json').write_text(json.dumps(last,ensure_ascii=False,indent=2))
for lang in ['zh','en']:
 for start in [16,26,31]:
  files=[shots/f'{lang}-{i:02d}.png' for i in range(start,min(start+5,35))];sheet=Image.new('RGB',(1280,len(files)*400),'#d7e0e7');draw=ImageDraw.Draw(sheet)
  for row,p in enumerate(files):
   im=Image.open(p);im.thumbnail((1216,362));sheet.paste(im,((1280-im.width)//2,row*400+28));draw.text((18,row*400+6),p.stem,fill='#17324d')
  sheet.save(shots/f'{lang}-sheet-{(start-1)//5+1}.png')
# Preserve the checker’s proximity warnings, with references for manual review.
ns=list(d['nodes'].values());tri=[n for n in ns if n['type']=='shape' and n.get('shapeKind') in ['triangle','arrow','polygon'] and n['width']<=30 and n['height']<=30];flags=[]
for ln in [n for n in ns if n['type']=='shape' and n.get('shapeKind')=='polyline']:
 x,y=ln['transform']['x'],ln['transform']['y'];near=[t['id'] for t in tri if t['sceneId']==ln['sceneId'] and x-20<=t['transform']['x']+t['width']/2<=x+ln['width']+20 and y-20<=t['transform']['y']+t['height']/2<=y+ln['height']+20]
 if near:flags.append(dict(line=ln['id'],nearby_triangles=near,review='scientific/data/mascot detail, not a flow arrow'))
assert len(flags)==18;(O/'箭頭提醒核對.json').write_text(json.dumps(flags,ensure_ascii=False,indent=2))
state=f'{now}（Asia/Taipei）：完成IDL第二章KP08中英文原生圖解審閱稿，各34頁、103步、1977節點、34原生連接線；最新e494ab9技能29資源核對。三輪68頁零溢出、412步播放及最後5頁60步驗證；功能12/12。原始OR與Newton實驗重算，符號／有限解／分類目標限制詳見審閱說明。工作主機Codex workspace，完整輸出/workspace/scratch/idl-lec02-kp08-review/output/。待教授審閱，未發布、未推送，原講義與前KP不變。'
(R/'COURSE_STATE更新草稿.md').write_text('# COURSE_STATE 更新草稿（未同步）\n\n'+state+'\n')
(R/'執行紀錄.md').write_text('# IDL KP08 執行紀錄\n\n執行者：Codex。任務：繼續KP08。\n\n'+state+'\n\n歷程：來源與技能核對、事前頁規格、精算、原生建頁、三輪渲染與修正、逐拍及字型驗證、打包。起始時間未另記錄，不估算時長。\n問題與處理：配對表與公式重疊、更新缺視覺指引、座標與機率標籤擠近、選項圖形不貼切，均已修正；教材敘述疑點在審閱稿明示。\n')
for name in ['COURSE_STATE更新草稿.md','執行紀錄.md']:shutil.copy2(R/name,O/name)
b=O/'建頁來源';b.mkdir(exist_ok=True)
for name in ['build_kp08.py','kp08_nums.py','pagekit.py','mascot.py','specs.py','prepare_specs.py','render_review.py','verify_browser.py','verify_final_pages.py','validate_native.py','check_html_local.mjs','source-lecture.md','source-figures.py','package_review.py','AGENTS.md','COURSE_STATE.md']:shutil.copy2(R/name,b/name)
for p in (R/'toolchain').rglob('*'):
 if p.is_file() and 'node_modules' not in p.parts and '__pycache__' not in p.parts:
  q=b/'toolchain'/p.relative_to(R/'toolchain');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
z=O/'IDL第二章KP08-審閱包.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in sorted(O.rglob('*')):
  if p.is_file() and p!=z:f.write(p,p.relative_to(O))
with zipfile.ZipFile(z) as f:assert f.testzip() is None;print('ZIP integrity PASS;',len(f.namelist()),'files;',z.stat().st_size,'bytes')
