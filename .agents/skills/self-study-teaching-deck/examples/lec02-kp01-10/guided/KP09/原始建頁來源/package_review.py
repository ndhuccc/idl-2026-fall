from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,shutil,zipfile
R=Path(__file__).parent;O=R/'output';now=datetime.now(ZoneInfo('Asia/Taipei')).isoformat();m=json.loads((R/'skills-source-manifest.json').read_text());d=json.loads((O/'lec02-kp09-rbf-and-som.jbd.json').read_text());pages=json.loads((O/'zh-pages.json').read_text());nums=json.loads((R/'數值驗算.json').read_text())
for local,p in [('source-lecture.md','lectures/lec02-kp09-rbf-and-som.md'),('source-figures.py','assets/figures/src/lec02-kp0810-figures.py'),('AGENTS.md','AGENTS.md'),('COURSE_STATE.md','COURSE_STATE.md')]:
 sha=next(x['sha'] for x in m['sources'] if x['path']==p);b=(R/local).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==sha,local
browser=json.loads((O/'browser-verification.json').read_text());audit=json.loads((R/'round3/browser-audit.json').read_text());assert len(audit['pages'])==62 and not any(p['overflow'] for p in audit['pages']) and not audit['page_errors'];font=(O/'HTML字型檢查.txt').read_text();assert '0 個問題' in font and '0 個節點因超出方框而縮小' in font,font
report=f'''# IDL 第二章 KP09：RBF 與 SOM 審閱稿

完成查核：{now}（Asia/Taipei）。獨立本機審閱稿，尚未提交、推送或發布；沒有覆蓋原講義、既有簡報或KP01–KP08。

## 開啟與播放

- 中文：lec02-kp09-rbf-and-som.html；英文：lec02-kp09-rbf-and-som-en.html。兩份皆離線單檔。
- 同名 .jbd.json 為可編輯的原生簡報，沒有圖片節點。
- 每語31頁、94步（包含根預覽步），每頁三拍；2003個原生節點、44條原生connector。兩語座標、元件型別、鏡頭與揭示順序一致。
- 方向鍵／空白鍵逐拍前進，左方向鍵回看，N開啟講者稿，F全螢幕。第一步為全份縮圖，再前進進入第一頁。
- 漸進出場示例.gif 由第18頁實際HTML鍵盤播放擷取，展示問題、移動連線與角色、結論三拍。暫時提問第二拍退出，鏡頭聚焦更新圖再回全景。

這是展開的圖解版，主線24頁約25–30分鐘，練習另計；並非把31頁全部塞進講義建議的15分鐘。若課堂只有15分鐘，可將矩陣細節、完整係數推導、寬度與兩種衰減排程列為課後閱讀，聚焦兩模型目的、XOR換座標及SOM手算。完整內容保留供審閱。

## 來源與技能

課程：ndhuccc/idl-2026-fall。當次核對 main：`{m['commit']}`；遞迴樹完整，14項課程技能。沿用本對話明確指定的 page-to-teaching-diagram → prompt-to-jbd-page，配合 image-to-jbd-page 原生庫與 slides-to-html；相關28項技能資源與課程check_slides共29個Git blob逐一對照最新版一致。來源、建頁程式和工具副本均包含在包內。

講義：lectures/lec02-kp09-rbf-and-som.md，blob `2ebf7098aa80ed91525491c728a8022af2b2e7bc`。
圖程式：assets/figures/src/lec02-kp0810-figures.py，blob `ef5f0a3058d3eda68ae6bc3de88e362f5cba5d8c`。原講義與原程式快照、AGENTS及COURSE_STATE也核對Git blob SHA相符。未修改上游工具或課程技能。

## 範圍對照

| 講義範圍 | 頁次 | 圖解主線 |
|---|---|---|
| 前言、§1高斯與RBF | 1–4 | 兩條路線、局部反應、K-means準備與固定特徵、線性輸出 |
| §1最小二乘 | 5–6 | 欄樣本矩陣、尺寸、正規方程、滿秩與奇異兩種解 |
| §1 XOR算例 | 7–13 | 手選中心、四筆距離、精算特徵、偏置、係數、四筆輸出、分界 |
| 選讀寬度 | 14 | 三組精算一維剖面，窄與寬的不同影響 |
| §2 SOM介紹與手算 | 15–19 | 無標籤原型、資料與格子兩空間、勝者、兩個鄰域更新、硬鄰域條件 |
| §2排程與來源圖 | 20–22 | 講義衰減索引、來源高斯軟鄰域、原程式4000步重現 |
| 重點與下一KP | 23–24 | RBF/SOM流程對照、萬有逼近的下一問 |
| 小練習1–3 | 25–27 | 保留題意與提示，不提供答案 |
| 自我檢核1–4 | 28–31 | 四選項原意保留，樣式中立，沒有答案標示 |

章節機器人延續藍頭、紫身、橙觸角，依概念拿直尺、收集籃、旋鈕、格子與開關。所有圖像是原生可編輯形狀與連線；科學數據圖採精算座標，角色是比喻。XOR正類用三角、負類用圓，額外以顏色區分。

## 數值與限制

1. XOR中心手選為(0,0)、(1,1)，二倍變異數等於一。完整特徵：00=(1,exp(-2))，01與10=(exp(-1),exp(-1))，11=(exp(-2),1)。不是K-means結果，兩正類重疊仍可分類。
2. 加常數一作偏置後，增廣設計矩陣秩為三。係數a約{nums['xor']['least_squares'][0]:.9f}，b約{nums['xor']['least_squares'][2]:.9f}。依對稱兩方程推導，保留精確exp值及完整中間精度，四輸出為0、1、1、0；數值殘差小於1e-12。畫面顯示的小數係數只是近似，不能拿截短係數聲稱精確輸出。
3. 輸出門檻0.5對應特徵和約{nums['xor']['boundary']:.12f}；負係數使不等號反向，正類在特徵和較小的一側。講義0.936及圖程式0.9355是不同取位，沒有改原始檔；本稿統一由完整解重算。半徑0.7虛線圈是exp(-0.49)約0.613的等反應線，不是高斯硬範圍。
4. RBF第一層固定是本課的訓練流程。K-means本身是資料驅動的前置估計；一般RBF也可以再調中心與寬度，不能說所有RBF都永遠不學第一層。最小二乘公式採Phi為m乘n、W為m乘k、Y為k乘n；加偏置後相應增廣m。滿列秩才可用普通逆；奇異時偽逆給最小範數最小二乘解。實作宜SVD或解方程，近奇異的泛化問題不會僅靠偽逆自動消失。
5. 寬度頁是新增精算一維剖面，兩中心0與1，標準差0.15、0.5、2，不是來源另一次訓練實驗。窄基函數空隙的輸出只有在無偏置時才趨近零；有偏置則趨近偏置，講者稿已釐清。
6. SOM手算輸入(0.9,0.7)，原型(0.2,0.2)、(0.5,0.5)、(0.8,0.8)。歐氏距離約0.860233、0.447214、0.141421，勝者三。硬鄰域為二、三，學習率0.5，更新為(0.7,0.6)、(0.85,0.75)，第一顆不動。原教材逐維絕對值相加是L1，本例1.2、0.6、0.2也選出三；不把兩種距離公式混用。
7. 格子鄰接與資料距離分開：勝者依資料空間距離選出，鄰居依固定格子索引找；移動的是資料原型，不是格子索引。SOM無標籤，勝者索引不是已知語意類別，也不應稱為有標籤分類器。
8. 第20頁照錄講義eta下一步=eta初始乘(1-tau/t)，另給初始率、tau從零到t減一；它和第21頁圖程式的排程不同，明示區分。鄰域逐漸縮小，沒有使用原教材每步對半的說法。
9. 原程式的8乘8實驗完全重現NumPy default_rng(0)：先抽128個初始常態值，再依序抽4000筆二維均勻輸入。學習率0.5*(1-tau/4000)+0.01，鄰域半徑3*(1-tau/4000)+0.4。初始分別0.51、3.4；最後執行tau3999，分別0.010125、0.40075，沒有執行tau4000。
10. 高斯鄰域h由格子索引歐氏距離計算，對所有有限距離為正，遠處仍有微小更新，和手算的硬鄰域外不動不同。獨立取出原程式未修改的模擬AST執行，初始與最終陣列均逐位元相同。每張網格64個節點，16條八點科學折線表示112條固定鄰接邊；不是手畫流程箭頭。這只是單一seed與均勻單位正方形的展示，不保證一般SOM完全無摺疊或保存所有近鄰。
11. 下一頁僅提出萬有逼近問題，不假稱證明定理；表達能力不等於容易最佳化或一定泛化。RBF與SOM機制不同，沒有把SOM說成普通多層前饋網路。

原講義的來源更正與小練習1解題示範完整保留於建頁來源快照，供來源追溯；本次簡報及講者稿的練習與檢核只保留題目、提示，沒有解答鍵。

## 三輪目視與驗證

- 第一輪渲染中英60張：發現兩空間對照圖的小節點文字溢出、矩陣示意順序與公式不一致、部分軸標靠近公式。修正框高與矩陣順序、拉開間距；增加四筆輸出核對頁。
- 第二輪渲染中英62張，零溢出候選。目視後將兩空間／勝者／更新圖改成等比例座標、補計算對應原生連線、調整SOM流程順序，換掉無標籤頁上的標籤道具，移開暫時提問。
- 第三輪重新渲染並目視兩語全部頁面與索引，62張零文字溢出、零頁面錯誤。公式、點位、線條、角色、選項與圖層均可讀。
- 原生schema v3、最低字級16、兩語幾何、鏡頭、元件對應、scene與reveal引用均通過。check_deck_quality --strict及課程check_slides兩語通過；功能群11/12，唯一未使用的功能群是行內多色／多字級強調，不影響技能要求至少九組。
- 1847個初始隱藏節點各揭示一次，每頁三拍。44個connector負責流程與資料對應，網格科學折線保留原拓撲。
- 中英×1280×720／1920×1080共{sum(x['steps'] for x in browser)}步完成逐步驗證，新增物件可見、退出物件隱藏；無KaTeX／JavaScript錯誤、無外部HTTP請求。最後僅修正第18頁鏡頭，使標題保持完整；該頁三拍另以兩語與兩視窗共12步複查（final-camera-verification.json），動畫重新等待出場完成再擷取。HTML實際字級／粗細／對齊另依課程check_html邏輯驗證，無問題與自動縮字；本機僅適配Chromium啟動與setContent載入。
- 檢查器提醒：每語93個有講者稿的步驟，下載者可讀；內容已檢查可用於教學。另每語18條折線被「鄰近小三角」啟發式標成手畫箭頭，實際為角色道具、科學圖線與資料三角等鄰近；所有流程均使用原生connector。沒有更改檢查器來壓掉提醒。

尚未操作實際JianBauDau App匯入或匯出PDF；未執行完整課程doctor／網站建置，因本目錄是審閱工作副本而非完整課程checkout。這些不列為已通過項目。
'''
(O/'審閱說明.md').write_text(report)
# Save review state locally; shared course repo remains untouched.
(R/'COURSE_STATE更新草稿.md').write_text(f'# KP09 審閱進度草稿\n\n{now}，IDL第二章KP09 RBF與SOM，中英各31頁，三拍，已完成三輪目視與數值／播放驗證；待教授審閱，未推送。\n\n輸入：ndhuccc/idl-2026-fall @ {m["commit"]} / lectures/lec02-kp09-rbf-and-som.md。\n輸出主機：本次Codex工作空間；根目錄：{O}。\n')
(R/'執行紀錄.md').write_text(f'# KP09 執行紀錄\n\n執行者：本對話Codex主代理；沒有委派子代理。完成時間：{now}；未保存精確開始時間，不虛構工時。\n\n任務：依最新版課程技能繼續KP09中英可編輯簡報，含活潑角色、精算圖及三拍漸進出場。\n\n輸入：GitHub ndhuccc/idl-2026-fall，commit {m["commit"]}，講義及產圖程式SHA見skills-source-manifest.json。\n\n輸出：主機為本次Codex容器，{O}；主要檔名lec02-kp09-rbf-and-som.html、-en.html、同名.jbd.json與IDL第二章KP09-審閱包.zip。\n\n歷程：來源核對→數值重現→事前頁規格→原生建頁→三輪渲染批判修訂→結構、來源AST、四組視窗播放、字型檢查→打包。\n\n錯誤與修復：初建英文群心說明框過窄，改兩短行；第一輪小原型框溢出，增加框高；字型檢查首次參數誤用檔名，改正為兩個目錄及/usr/bin/chromium後通過；最後減少第18頁鏡頭推近避免裁標題，修正動畫擷取等待時間，並另做12步複查。所有結果以包內最終驗證JSON為準。\n\n未提交、推送、發布，未改原講義或既有KP。\n')
# Append actual final text, avoiding duplicate sections on reruns.
p=R/'圖解提示詞.md';s=p.read_text().split('\n# 最終精確文案')[0]+'\n# 最終精確文案\n\n以下對應最終原生節點；幾何與計算見建頁程式及數值驗算。\n'
for pg in pages:
 pre=f'zh-p{pg["page"]:02d}-';s+=f'\n## 第{pg["page"]}頁\n\n'+'\n\n'.join(n['text'] for n in d['nodes'].values() if n['id'].startswith(pre) and n.get('text'))+'\n'
p.write_text(s)
for name in ['數值驗算.json','圖解證據表.json','skills-source-manifest.json','圖解提示詞.md','執行紀錄.md','COURSE_STATE更新草稿.md']:shutil.copy2(R/name,O/name)
shutil.copy2(R/'round3/browser-audit.json',O/'逐頁排版檢查.json')
shots=O/'預覽截圖';shots.mkdir(exist_ok=True)
for p in (R/'round3').glob('*.png'):shutil.copy2(p,shots/p.name)
src=O/'建頁來源';src.mkdir(exist_ok=True)
for p in R.iterdir():
 if p.suffix in ['.py','.mjs'] or p.name in ['source-lecture.md','AGENTS.md','COURSE_STATE.md','current-source.json']:shutil.copy2(p,src/p.name)
shutil.copytree(R/'toolchain',src/'toolchain',dirs_exist_ok=True,ignore=shutil.ignore_patterns('node_modules','__pycache__','*.pyc'))
required=['lec02-kp09-rbf-and-som.html','lec02-kp09-rbf-and-som-en.html','lec02-kp09-rbf-and-som.jbd.json','lec02-kp09-rbf-and-som-en.jbd.json','審閱說明.md','漸進出場示例.gif','browser-verification.json','HTML字型檢查.txt']
assert all((O/p).stat().st_size>0 for p in required)
z=O/'IDL第二章KP09-審閱包.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in sorted(O.rglob('*')):
  if p.is_file() and p!=z:f.write(p,p.relative_to(O))
with zipfile.ZipFile(z) as f:assert f.testzip() is None
print('PACKAGED',z, z.stat().st_size,'bytes;',len(pages),'pages per language')
