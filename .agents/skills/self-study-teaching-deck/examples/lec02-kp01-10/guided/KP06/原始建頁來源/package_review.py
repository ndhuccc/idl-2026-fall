from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,shutil,zipfile,hashlib
R=Path(__file__).parent;O=R/'output';now=datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M')
report=f'''# IDL 第二章 KP06 活化函數：審閱稿

完成查核：{now}（Asia/Taipei）。輸出主機為 Codex workspace，本機審閱草稿，未推送／發布，原講義與 KP01–KP05 未更動。

## 直接審閱

- 中文：lec02-kp06-activation-functions.html
- 英文：lec02-kp06-activation-functions-en.html
- 可編輯：同名 .jbd.json，無 image 節點。
- 右方向鍵／空白鍵逐拍前進、左方向鍵回看、N 看講者備註、F 全螢幕。根步驟為整份框架預覽，進下一步開始第一頁。
- 漸進出場示例.gif：第6頁實際鍵盤播放擷取，三拍循環。數字與曲線為真實計算，角色及道具是教學比喻。

兩語各31頁、每頁3拍、94步（含根步驟）、1927個原生節點、37條原生連接線，幾何、鏡頭、揭示順序一致。24頁主線含1頁選讀推導，再加3題練習與4題自我檢核。這是展開圖解版，主線約25–30分鐘；若只講15分鐘可略過導數矩陣、推導與部分比較，保留完整頁供課後閱讀，沒有刪掉講義定義。

## 最新技能與來源

課程：ndhuccc/idl-2026-fall。最新 main commit `1c323c79783e70c217f967dfd686cadb66be1256`，完整遞迴盤點14項技能。沿用教師在本對話已選定的 page-to-teaching-diagram，接 prompt-to-jbd-page 與 image-to-jbd-page 原生庫，再用 slides-to-html。四項技能及28個支援檔逐一驗證 Git blob SHA，與 KP05 相同，因此使用已核對的完整副本，沒有宣稱有新版本差異。完整清單見 skills-source-manifest.json。

講義 `lectures/lec02-kp06-activation-functions.md`，blob `868613f186e380d69131d1acdf1d5d2f5d669dd8`；圖程式 `assets/figures/src/lec02-kp0507-figures.py`，blob `a4102512bc3bfa1bdae68db7796a78c5cb277d9f`。原始快照與可重建程式收在「建頁來源」。精算曲線採原圖定義與600個樣本，分成不超過181點的原生折線；新增單函數及導數圖按相同公式生成，不冒充來源實驗。

## 講義範圍與頁次

| 來源 | 頁次 | 視覺論證 |
|---|---|---|
| 前言、目標、§1分類 | 1–2 | 輸入經活化、13種家族關係 |
| §1 Linear、Step、Sign | 3–4 | 真實直線、階梯與端點約定、公式、值域、導數 |
| §2 Softmax及§1導數 | 5–7 | 共用分母、比例長條、手算、分配連動曲線 |
| §1–2 Sigmoid、tanh、Gaussian | 8–10 | 函數與導數同軸、飽和與峰頂 |
| §2 梯度因素 | 11 | 五個導數因子與同尺度長條 |
| §1、§3–4 ReLU家族 | 12–17 | 折角、卡住、固定／可學斜率、負側下限、平滑近似 |
| §3 原圖與手算 | 18–19 | 負二、正二，四函數完整代入值與斜率 |
| §1 Maxout | 20 | 多分支的上包絡、候選與輸入導數的區別 |
| §4 選擇 | 21–22 | 隱藏層觀察分流、任務到輸出活化 |
| 補充推導 | 23 | Sigmoid與Softplus導數 |
| 下一KP | 24 | 預測與目標流向損失 |
| 延伸練習 | 25–27 | 三題；頁面及備註只給提示 |
| 小測驗 | 28–31 | 四題；等樣式候選，無答案標記或答案備註 |

13種包含 Linear、Binary step、Sign、Softmax、Sigmoid、tanh、Gaussian、Softplus、ReLU、Leaky ReLU、PReLU、ELU、Maxout；Step與Sign共一列但算兩種，並未少算。公式、值域與導數均有對應頁；例外與來源修正詳列下節。

## 講義疑點與本稿釐清

| 講義原敘述／容易誤讀之處 | 本稿處理與依據 |
|---|---|
| Gaussian列入S形族 | 保留來源分組位置，但第2、10頁明示鐘形，依exp(-a²)精算圖與導數判定。 |
| 「跨層」 | 解釋為整向量／多候選運算，不暗示跨不同網路深度。 |
| Linear只適合回歸輸出；隱藏有Linear就坍縮 | 第3頁改為全線性網路才整體合併。單一線性層可存在其他層仍非線性的模型中；回歸例子限無限制實數目標。 |
| PReLU值域一概全實數 | 第15頁附alpha正條件；備註說明alpha零等於ReLU、負參數時兩側輸出非負。 |
| ELU一概光滑、值域負alpha到無限 | 第16頁附alpha正的值域條件；零點左斜率alpha、右斜率一，因此alpha一才可微。 |
| Maxout值域全實數、導數贏家一 | 第20頁給明示補充反例max(a,-a)=abs(a)，值域非負；區分對候選值偏導與對輸入梯度，唯一贏家時輸入梯度為其權重。平手可能不可微。 |
| 正側四函數斜率皆一 | 第17、19頁釐清Softplus導數為Sigmoid，正二處約0.881，有限正輸入不等於一；輸出約2.127，也不等於二。 |
| 五層最好情況梯度只剩0.00098 | 第11頁明示只乘五個活化導數因子，上界0.0009765625；完整梯度還含權重矩陣與上游梯度，不能作無條件梯度上界。 |
| ReLU避免梯度消失 | 正側活化因子一不衰減，不保證全模型不消失／爆炸；第28頁相應選項也避免過度保證。 |
| 負輸入使ReLU永遠無法更新 | 第13、30頁以所有相關訓練輸入皆在負側為條件；備註保留普通資料梯度範圍，非一次負輸入即永久卡住。Leaky非零斜率也不保證更新。 |
| Softmax和Sigmoid機率 | 第5、22頁保留有限分數、至少兩類、互斥分類條件。分量逐點運算不等於統計獨立；輸出零一之間不自動代表校準。 |

以上修正在審閱稿、提示詞與講者稿中明示；原講義完全保留。新增的Softmax連動例、Maxout反例、參數條件與校準提醒標為補充／釐清。沒有新增「問AI」泡泡，也沒有將必要公式省略成AI提問。

## 數值與檢查

- exp(2)+exp(1)+exp(0)=11.107337927389695，Softmax約(0.665241,0.244728,0.090031)，總和一。中間值保持完整精度。
- ReLU、Leaky(0.01)、ELU(alpha一)、Softplus在正負二的輸出與導數全部assert；9種函數在4個非折角點以獨立中央差分核對導數。詳細全精度值見數值驗算.json。
- 第一輪實際看渲染圖：修正Softmax公式轉義、家族圖標籤、比較頁缺少代入式。第二輪62頁零溢出，目視發現標題小圖的標籤／貼紙與座標重疊、推導頁公式接近軸標，已移開。第三輪重新渲染並查看兩語全部索引：補家族名、明示階梯定義、把測驗裝飾改成選項相關小圖、長條恢復真實比例；62頁零溢出候選、零頁面錯誤。最後再修英文單複數，重渲染兩語第2頁確認。
- check_deck_quality.py --strict及check_slides.py兩語通過；功能群12/12、每頁活潑效果與圖形數達標、1771個隱藏節點各揭示一次、無文字被高圖層實心圖遮住、無空泡泡或空文字。
- 中英×1280×720／1920×1080四組，共376步逐步驗證進場、退場、連接線可見、無KaTeX／JS錯誤、無外部HTTP請求。第6頁第二拍問題提示退場並推近鏡頭，第三拍回全景；每拍有同步講稿。
- 最終HTML另以課程check_html的相同字型／粗體／對齊比較邏輯檢查；僅為環境限制改本機啟動方式及由setContent載入，未更改上游工具，結果收在HTML字型檢查.txt。
- 原生結構、有效場景／連線掛點、字級至少16、雙語幾何與鏡頭相同。逐頁截圖及驗證JSON均隨包附上。

檢查器仍有兩類提醒：93個步驟含可公開教學講者稿，已檢視練習與小測驗均未附答案；23條折線因附近有小三角形被啟發式判作「手畫箭頭」，實際是機器人籃子內三角道具旁的手臂／脈搏線，以及零點警示旁的函數折線。建頁程式所有流程均用原生connector，未使用手畫箭頭函式。這些提醒保留在原始驗證紀錄中，未改檢查器壓掉。

限制：已驗證離線HTML，實際JianBauDau App的匯入與播放尚未操作，請在App內確認；未做PDF、完整課程doctor或網站建置。這不是完整課程checkout。若重建，請同步修改建頁來源而非只改JSON，以免被覆蓋。
'''
(O/'審閱說明.md').write_text(report)
for name in ['數值驗算.json','圖解證據表.json','skills-source-manifest.json','圖解提示詞.md']:
 shutil.copy2(R/name,O/name)
shutil.copy2(R/'round3-final/browser-audit.json',O/'逐頁排版檢查.json')
shots=O/'預覽截圖';shots.mkdir(exist_ok=True)
for p in (R/'round3-final').glob('*.png'):shutil.copy2(p,shots/p.name)
for lang in ['zh','en']:shutil.copy2(R/f'final-wording/{lang}-02.png',shots/f'{lang}-02.png')
# Update contact sheets containing the corrected wording.
from PIL import Image,ImageDraw
for lang in ['zh','en']:
 files=[shots/f'{lang}-{i:02d}.png' for i in range(1,6)];sheet=Image.new('RGB',(1280,2000),'#d7e0e7');draw=ImageDraw.Draw(sheet)
 for row,p in enumerate(files):
  im=Image.open(p);im.thumbnail((1216,362));sheet.paste(im,((1280-im.width)//2,row*400+28));draw.text((18,row*400+6),p.stem,fill='#17324d')
 sheet.save(shots/f'{lang}-sheet-1.png')
# Source bundle, including verified skill files but excluding caches and installed dependencies.
b=O/'建頁來源';b.mkdir(exist_ok=True)
for name in ['build_kp06.py','kp06_nums.py','pagekit.py','mascot.py','specs.py','prepare_specs.py','render_review.py','verify_browser.py','validate_native.py','check_html_local.mjs','source-lecture.md','source-figures.py','package_review.py','執行紀錄.md']:
 shutil.copy2(R/name,b/name)
for p in (R/'toolchain').rglob('*'):
 if p.is_file() and 'node_modules' not in p.parts and '__pycache__' not in p.parts:
  q=b/'toolchain'/p.relative_to(R/'toolchain');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
(R/'COURSE_STATE更新草稿.md').write_text(f'# COURSE_STATE 更新草稿（未同步GitHub）\n\n{now}（Asia/Taipei）：完成IDL第二章KP06活化函數中英審閱稿，各31頁、94步、1927原生節點與37連接線；沿用教師選定圖解流程並核對最新1c323c7版四項技能28資源。三輪目視與62頁零溢出、376步播放驗證通過。數值與講義釐清詳見審閱說明。輸出主機Codex workspace，完整路徑/workspace/scratch/idl-lec02-kp06-review/output/。待教授審閱，未發布、未推送。\n')
shutil.copy2(R/'COURSE_STATE更新草稿.md',O/'COURSE_STATE更新草稿.md')
# ZIP is made last, after the HTML typography result has been saved.
z=O/'IDL第二章KP06-審閱包.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in sorted(O.rglob('*')):
  if p.is_file() and p!=z:f.write(p,p.relative_to(O))
with zipfile.ZipFile(z) as f:assert f.testzip() is None;print('ZIP integrity PASS;',len(f.namelist()),'files;',z.stat().st_size,'bytes')
