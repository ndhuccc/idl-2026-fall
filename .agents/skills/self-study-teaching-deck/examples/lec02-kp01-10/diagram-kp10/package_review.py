from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,shutil,zipfile,subprocess
from PIL import Image,ImageDraw
from specs import PAGES
R=Path(__file__).parent;O=R/'output';now=datetime.now(ZoneInfo('Asia/Taipei')).isoformat();m=json.loads((R/'skills-source-manifest.json').read_text());d=json.loads((O/'lec02-kp10-universal-approximation.jbd.json').read_text());nums=json.loads((R/'數值驗算.json').read_text())
for f in m['files']:
 b=(R/'toolchain'/f['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['sha']
for f in m['self_study_skill_files']:
 b=(R/f['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['sha']
for local,p in [('source-lecture.md','lectures/lec02-kp10-universal-approximation.md'),('source-figures.py','assets/figures/src/lec02-kp0810-figures.py'),('AGENTS.md','AGENTS.md'),('COURSE_STATE.md','COURSE_STATE.md')]:
 b=(R/local).read_bytes();sha=next(x['sha'] for x in m['sources'] if x['path']==p);assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==sha
browser=json.loads((O/'browser-verification.json').read_text());patch=json.loads((O/'final-pages-verification.json').read_text());assert len(browser)==4 and len(patch)==4
font=(O/'HTML字型檢查.txt').read_text();assert '0 個問題' in font and '0 個節點因超出方框而縮小' in font
for lang in ['zh','en']:assert not json.loads((R/f'final-preview-{lang}'/'overflow.json').read_text())
audit=json.loads((R/'round3/browser-audit.json').read_text());assert len(audit['pages'])==90 and not any(x['overflow'] for x in audit['pages'])
for n in [6,28,31,38]:
 p=json.loads((R/f'final-polish-{n}'/'browser-audit.json').read_text());assert len(p['pages'])==2 and not any(x['overflow'] for x in p['pages'])
# Re-run the new skill's checks on the final deck and evidence, preserving exact output.
checks=[];skill=R/'.agents/skills/self-study-teaching-deck/tools'
for name,args in [('coverage',['python3',str(skill/'check_coverage.py'),str(R/'source-lecture.md'),str(R/'圖解證據表.json')])]+[(lang,['python3',str(skill/'check_style.py'),'--style','diagram',str(O/('lec02-kp10-universal-approximation'+('-en' if lang=='en' else '')+'.jbd.json')),str(R/'圖解證據表.json')]) for lang in ['zh','en']]:
 v=subprocess.run(args,capture_output=True,text=True);assert v.returncode==0,v.stdout+v.stderr;checks.append(dict(check=name,returncode=0,output=v.stdout+v.stderr))
(O/'self-study-verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
report=f'''# IDL 第二章 KP10：萬有逼近定理｜圖解自學審閱稿

完成查核：{now}（Asia/Taipei）。依教授本次指定的 self-study-teaching-deck 與「圖解版」製作。本機獨立審閱稿，尚未提交、推送或發布；沒有改原講義、上游技能、既有簡報或KP01–KP09。

## 開啟與自學

- 中文：lec02-kp10-universal-approximation.html；英文：lec02-kp10-universal-approximation-en.html。兩份都是離線單檔，同名.jbd.json為可編輯原生來源。
- 中英各45頁、136步（含根預覽），每頁三拍；最終每語1285個原生節點、76條原生connector。兩語幾何、順序與鏡頭一致。
- 方向鍵／空白鍵逐拍前進，左方向鍵回看，N開啟自學導讀，F全螢幕。根步驟顯示全份縮圖，下一步進第一頁。
- 本次採圖解版：全寬圖像／算式、承接小標、深藍結論橫幅，不放吉祥物。新名詞與符號在初次出現的頁面或導讀解釋；每拍都有完整白話稿。
- 依新技能要求，三個迷思、三題練習、四題檢核均有獨立題目與後續解析。題目頁及其導讀不洩露答案。這次解析頁是刻意提供學生自學，和先前KP審閱稿只給提示的做法不同。
- 漸進出場示例.gif 為第10頁實際HTML鍵盤播放擷取：高度差構造、完整加總公式、結論；已等待轉場與顯示完成。

45頁由內容與自學步驟決定，沒有為講義建議的15分鐘刪公式、限制或題解。課堂可選頁講解，完整版本供學生自行閱讀。

## 來源與技能版本

課程 ndhuccc/idl-2026-fall，當次main `7762bd27c70e10761e0a9a1755a8dc0038d63321`。完整遞迴樹未截斷，發現15項頂層技能，包含本次新增試行技能。

- self-study-teaching-deck/SKILL.md blob：`c1b1d9ccfab9472bf2ac493434edde5bfd91df4e`。完整讀取技能、兩種風格、頁規格、自學檢查清單、coverage/style工具；參照圖解版範例與工具庫。
- 講義 lectures/lec02-kp10-universal-approximation.md blob：`dd9c8128e28f14d7eedaaf5db2f2cdc4f47b2af2`。
- 原圖程式 assets/figures/src/lec02-kp0810-figures.py blob：`ef5f0a3058d3eda68ae6bc3de88e362f5cba5d8c`。
- 新技能7個檔案、原生建頁／轉HTML／課程檢查的29個資源逐檔核對Git blob SHA；原講義、原程式與兩個課程規範快照亦相符，見skills-source-manifest.json。

本稿使用既有原生Page庫設計圖解版，沒有把導學版的吉祥物留在畫面。建頁程式、事前提示詞、頁規格與可重現數值均在包內。重建請修改建頁程式及specs.py；直接改JSON會被重建覆蓋。

## 範圍與頁次

| 講義範圍 | 頁次 | 自學路線 |
|---|---|---|
| 學習目標、符號、§1定理 | 1–6 | 表示與學習、網路參數、條件、量詞、上確界、Fourier類比 |
| §2階梯構造 | 7–11 | 等距切格、中點高度、有限sigmoid、望遠鏡和、參數及常數單元 |
| §2來源圖與誤差 | 12–16 | N=5/20完整精算曲線、六點雙對數圖、理想界、320格估算 |
| §3四步骨架 | 17–20 | 稠密、Hahn–Banach、Riesz、判別性與矛盾 |
| §4限制、三個迷思 | 21–27 | 存在性四個缺口；三題先判斷、再解析 |
| 選讀六模型 | 28 | 六模型事件時間線及機制差異 |
| 練習1 | 29–31 | 問題、完整四步解題、理想與有限陡度的核對 |
| 練習2、3 | 32–35 | 問題後接推理解析，不把充分條件當最低複雜度 |
| 四題自我檢核 | 36–43 | 每題中立選項後接答案與逐項導讀解析 |
| 重點、下一章問題 | 44–45 | 全單元回顧、梯度下降與反向傳播 |

完整「26個錨點→頁」對照附在本說明末尾；每頁也有來源錨點，並保存新增／釐清說明。新術語清單與數值鍵見圖解證據表.json。

## 數值重現與來源釐清

1. 原圖的標準sigmoid實作為0.5*(1+tanh(z/2))，本稿使用相同函式。k=2000、x網格為0到1的4001點；N=3、5、10、20、40、80六個完整近似陣列，和直接取出原程式未修改數值AST執行的結果逐位元一致。
2. 六個網格最大誤差約為0.866025、0.587785、0.309017、0.156434、0.078459、0.039260，顯示值用完整中間精度後才取位數。N=5與20的曲線保留所有4001個原始點，再分成每段不超181點的原生科學折線。
3. 原圖把4001點上的最大值寫成sup；有限網格只能提供取樣最大值，不是自動得到整段誤差的上界。本稿用e_G明示並保留原講義快照，沒有靜默修改來源。
4. 有限k的sigmoid和是平滑函數，不會在每格精確等於中點值；硬階梯S_N與平滑網路F_(N,k)分開記號。原講義相關「等於」改為理想階梯才精確成立，有限k只近似。
5. pi/N由正弦最大斜率2pi乘半格寬1/(2N)得到，是理想中點階梯的Lipschitz界。第15頁另列三角不等式分出平滑誤差，沒有把它當任意有限k的嚴格總誤差界。
6. 理想界小於0.01要求N>100pi≈314.159，所以整數315已滿足這個理想界；講義約320是規模估算。k=2000、N=320時取樣誤差約{nums['N320_sampled_error']:.9f}，比pi/320≈0.009817略大，正好呈現有限陡度與理想界的差異；不宣稱已取得整段誤差小於0.01的嚴格證書。
7. N先定義為切格數，實際有N−1個跳躍單元與常數v1。常數可作輸出偏置；若要完全符合無輸出偏置的定理式，標準sigmoid用w=0、theta=0、輸出權重2v1的一顆固定單元表示。這樣才可把N格數對應為含常數在內N個單元。
8. 練習一四高度0.125、0.375、0.625、0.875，三跳躍皆0.25，格界0.25、0.5、0.75。S4(0.6)=0.625、理想最大誤差0.125；有限陡度只接近，頁31新增k=20圖示平滑差距。練習二用理想pi/N界估算，N>20pi，最小整數63；這是充分條件，不是所有構造的最小單元數。
9. 定理量詞是每個連續f與每個epsilon，才選N與參數；不是同一固定網路同時解所有目標與精度。閉立方體是緊緻域，「有界」本身不能取代緊緻。連續誤差在此緊緻域可取最大值。
10. 證明只給骨架。span是有限線性組合，稠密是任意精度逼近；Hahn–Banach給非零連續線性泛函；Riesz給有限帶符號測度；最後的判別性才迫使測度為零。最後一步需要實質分析，沒有把一句條件當作完整證明。Cybenko的sigmoid論證與更一般的結果明示區分。
11. 有界版本不能直接涵蓋ReLU。講義引用的連續非多項式擴展可涵蓋ReLU，本稿作為文獻性補充說明，未另證。固定次數多項式活化不能以有限線性組合任意逼近所有連續函數。
12. Fourier「基底」類比用較精確的生成函數族表述，不宣稱神經單元線性獨立或唯一分解；兩類定理的全部條件也不同。深度效率只對某些函數有優勢，並非所有目標都一定更省或更好訓練。
13. 沿用六模型年代，MLP1986註明為反向傳播普及而非多層想法首次出現。M-P固定邏輯、ADALINE線性訓練、SOM無標籤等機制差異保留，未硬說六模型都有同樣非線性與誤差反傳。
14. 四題檢核保持原意，第二題明示泛函連續性；第三題把「不提供構造法」限定為本版存在性敘述，不否定其他構造性UAT。完整逐項排除理由在公開自學導讀。

## 驗證與限制

- 講義26錨點、45頁：check_coverage完全覆蓋；中英check_style diagram通過。來源每個主要概念、公式、算例、迷思、練習與檢核另做人工語意核對，沒有只靠錨點數當成內容品質。
- 每語1285節點，只用原生frame/text/shape/connector；最低文字18（技能最低要求16），公式24起。兩語幾何、元件、鏡頭、step與reveal引用一致；每頁三拍，每語1059個隱藏節點各揭示一次。沒有圖片、沒有高圖層蓋字、沒有空白文字框。
- check_deck_quality --strict與課程check_slides兩語通過，功能群9/12、76條原生connector、零手畫箭頭。未採用的功能是平滑頂點、退場及頁內鏡頭運動；精算圖保留折線，不為效果改動資料，版面以圖解與清楚分拍為主。
- 三輪實際渲染：第一輪修網路節點壓公式、小問號框與圖上刻度間距；第二輪零溢出後補曲線圖例、首次符號定義與完整網格誤差式；第三輪中英90張零文字溢出、零頁面錯誤，逐頁目視。最後再修第6、28、31、38頁的英文斷字、模型標籤、平滑參數與刻度，8張重新渲染目視，零溢出。
- 技能preview_deck近似預覽兩語各45頁零溢位；最後小修改另以實際離線HTML複查。近似工具本機只適配Chromium啟動，透過原工具的KATEX_DIR載入本機公式庫，不修改上游工具；不是拿近似圖取代實際HTML檢查。
- 中英×1280×720／1920×1080共544個步驟完整檢查揭示可見、離線載入與公式，無KaTeX／JS錯誤、無外部HTTP請求；最終4頁另複查48個步驟。HTML字級／粗細／對齊最終重查，零問題、零自動縮字。
- 每語135個步驟有講者備註的提醒保留：這是本技能刻意給學生看的自學導讀，已檢查為教學內容。沒有成績、學生名單或秘密，也沒有用修改檢查器的方式壓掉警告。

尚未在實際JianBauDau App匯入／播放，未做PDF匯出、完整課程doctor或網站建置；本目錄是審閱工作副本，不是完整課程checkout。這些不列為已完成檢查。

## 錨點到頁

'''
# Generate the exact anchor mapping using the shipped extractor without importing its CLI main.
listed=subprocess.run(['python3',str(skill/'check_coverage.py'),'--list',str(R/'source-lecture.md')],capture_output=True,text=True,check=True).stdout.splitlines();report+='| 錨點 | 頁 |\n|---|---|\n'
for a in listed:report+='| '+a+' | '+', '.join(str(p['page']) for p in PAGES if a in p['source_anchors'])+' |\n'
(O/'審閱說明.md').write_text(report)
# Local state only: do not mutate the shared course repo.
(R/'COURSE_STATE更新草稿.md').write_text(f'# KP10審閱進度草稿\n\n{now}，IDL第二章KP10，依教授指定self-study-teaching-deck的圖解版，中英各45頁，26/26錨點、題目與解析、自學導讀、三拍。完成三輪目視及最終修整、來源曲線逐點重現、結構／風格／播放／字型檢查。\n\n狀態：待教授審閱，未推送或發布。主機：本次Codex容器；輸出目錄：{O}。來源版本：{m["commit"]}。\n')
(R/'執行紀錄.md').write_text(f'# KP10執行紀錄\n\n完成時間：{now}。執行者：本對話Codex主代理，未派子代理。未保存精確開始時間，不虛構工時。\n\n任務：依self-study-teaching-deck製作IDL第二章KP10；教授已選圖解版。\n\n輸入：ndhuccc/idl-2026-fall @ {m["commit"]}，lectures/lec02-kp10-universal-approximation.md；原圖程式與技能SHA見manifest。\n\n輸出主機：本次Codex容器。完整路徑：{O}，主檔lec02-kp10-universal-approximation.html、-en.html及同名.jbd.json，ZIP為IDL第二章KP10-圖解自學審閱包.zip。\n\n歷程：讀最新技能→依技能詢問並取得圖解版選擇→規劃45頁／26錨點→提示詞→原程式數值重現→原生建頁→三輪實際HTML目視與修正→coverage/style/quality/course/preview/browser/font檢查→最後4頁複查→打包。\n\n錯誤與修復：初建時間線超框與英文長字框不足，調整幾何；第一輪圖表刻度壓公式及小問號溢出，拉開位置與加高；初次品質檢查只有6/12功能，增加有意義的警示形狀與行內重點及多段漸層後9/12通過；preview本機複本找不到nativefig，補正PYTHONPATH並使用原工具KATEX_DIR後兩語零溢位。原始工具與課程檔未改。\n\n未推送、未發布；未操作實際JBD App或整站建置。\n')
# Retain the original plan and append final exact node text as a portable implementation record.
p=R/'圖解提示詞.md';s=p.read_text().split('\n# 最終精確文案')[0]+'\n# 最終精確文案\n\n以下是最終中文版原生文字節點；英文及幾何見建頁腳本、證據表與JBD。\n'
for pg in PAGES:
 pre=f'zh-p{pg["page"]:02d}-';s+=f'\n## 第{pg["page"]}頁\n\n'+'\n\n'.join(n['text'] for n in d['nodes'].values() if n['id'].startswith(pre) and n.get('text'))+'\n'
p.write_text(s)
for n in ['圖解提示詞.md','圖解證據表.json','數值驗算.json','skills-source-manifest.json','COURSE_STATE更新草稿.md','執行紀錄.md']:shutil.copy2(R/n,O/n)
shutil.copy2(R/'round3/browser-audit.json',O/'逐頁排版檢查.json')
for lang in ['zh','en']:shutil.copy2(R/f'final-preview-{lang}'/'overflow.json',O/f'preview-overflow-{lang}.json')
shots=O/'預覽截圖';shots.mkdir(exist_ok=True)
for p in (R/'round3').glob('*.png'):
 if '-overview-' not in p.name and '-sheet-' not in p.name:shutil.copy2(p,shots/p.name)
for n in [6,28,31,38]:
 for lang in ['zh','en']:shutil.copy2(R/f'final-polish-{n}'/f'{lang}-{n:02d}.png',shots/f'{lang}-{n:02d}.png')
# Rebuild all contact sheets from the final individual images, avoiding stale thumbnails.
for lang in ['zh','en']:
 for start in range(1,46,5):
  chunk=[shots/f'{lang}-{j:02d}.png' for j in range(start,min(46,start+5))];sheet=Image.new('RGB',(1280,400*len(chunk)),'#dce3e9');draw=ImageDraw.Draw(sheet)
  for j,f in enumerate(chunk):
   im=Image.open(f);im.thumbnail((1216,362));sheet.paste(im,((1280-im.width)//2,j*400+28));draw.text((18,j*400+6),f.stem,fill='#17324d')
  sheet.save(shots/f'{lang}-sheet-{start//5+1}.png')
src=O/'建頁來源';src.mkdir(exist_ok=True)
for p in R.iterdir():
 if p.suffix in ['.py','.mjs'] or p.name in ['source-lecture.md','AGENTS.md','COURSE_STATE.md','內容規劃草稿.md']:shutil.copy2(p,src/p.name)
shutil.copytree(R/'toolchain',src/'toolchain',dirs_exist_ok=True,ignore=shutil.ignore_patterns('node_modules','__pycache__','*.pyc'))
shutil.copytree(R/'.agents',src/'.agents',dirs_exist_ok=True)
shutil.copytree(R/'reference-examples',src/'reference-examples',dirs_exist_ok=True)
shutil.copy2(R/'skills-source-manifest.json',src/'skills-source-manifest.json')
required=['lec02-kp10-universal-approximation.html','lec02-kp10-universal-approximation-en.html','lec02-kp10-universal-approximation.jbd.json','lec02-kp10-universal-approximation-en.jbd.json','審閱說明.md','漸進出場示例.gif','self-study-verification.json','source-reproduction.json','browser-verification.json','HTML字型檢查.txt']
assert all((O/p).stat().st_size>0 for p in required)
z=O/'IDL第二章KP10-圖解自學審閱包.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in sorted(O.rglob('*')):
  if p.is_file() and p!=z:f.write(p,p.relative_to(O))
with zipfile.ZipFile(z) as f:assert f.testzip() is None
print('PACKAGED',z,z.stat().st_size,'bytes')
