from pathlib import Path
import shutil,zipfile,json
R=Path(__file__).parent;O=R/'output'
report='''# IDL 第二章 KP03 簡報審閱稿

製作時間：2026-10-30 10:55–11:08（Asia/Taipei）。狀態：本機審閱草稿，未發布、未推送、未修改原講義。

## 來源與製作方式

- 課程：ndhuccc/idl-2026-fall，main `64e766a7019231ef86155a6a5550390832c217d9`。
- 講義：`lectures/lec02-kp03-signum-update-derivations.md`，blob `2520d7f9771c9f41e5246f47e33bca254ad30307`。
- 依課程 repo 的 page-to-teaching-diagram → prompt-to-jbd-page 原生節點流程製作；提示詞先於繪圖保存。以同一幾何配置輸出中英文，所有圖解為可編輯節點，無整頁圖片。
- 主線 19 頁：正負標籤、四種誤差、代數推導、學習率與偏置、法向量與距離、零分邊界、損失與梯度、更新算例、KP02 軌跡對照、更新條件與 KP04 引導。
- 延伸 7 頁：講義三題練習與四題自我檢核，僅提示，未提供答案。兩題原題有概念歧義，以下列出釐清方式。

## 檔案與操作

中英文各 26 頁、79 步（包含進入 KP 的根步驟）、1562 個原生節點。每頁 3 拍，圖示／推導／結論逐步出場。以 HTML 的右方向鍵逐拍前進，左方向鍵回看；每一步有相應講者備註。可離線開啟。

- 中文：`lec02-kp03-signum-update-derivations.html`；英文：`lec02-kp03-signum-update-derivations-en.html`。
- 同名 `.jbd.json` 為 JianBauDau 可編輯稿，含原生 fade reveal。
- `漸進出場示例.gif` 為中文第 15 頁用實際鍵盤操作擷取的三拍示範，已更新為活潑視覺版。
- `圖解提示詞.md`、`圖解證據表.json` 保存逐頁規格與追溯資料；`視覺改版提示詞.md` 保存角色、道具、對話與三輪改版設計。

## 講義概念釐清，供教授審閱

| 講義表述／問題 | 本簡報的處理 | 頁碼 |
|---|---|---|
| 在 sign(0)=+1 的約定下，以 d≥0／d<0 完整判定猜對／猜錯 | 先限非邊界點；另加負類邊界反例：s=0、輸出+1、標籤−1，距離為零卻猜錯 | 8–9 |
| 有錯必 E>0、E=0 必全對 | 邊界錯點可貢獻零分損失，明示「零損失不等於零錯誤」，停止條件仍看一整輪無誤分類 | 12、17 |
| 將 E=−Σ錯點 yᵢsᵢ 稱作距離總和 | 此式為未正規化分數損失。幾何距離須除以空間權重長度；幾何權重不包含偏置，並要求長度非零 | 6–7、12 |
| 直接在所有位置宣稱普通梯度成立 | 在非邊界處固定錯分集合求導；邊界可能無唯一梯度。sign 約定的更新可取相應有效次梯度 | 13–14 |
| 自我檢核第三題將絕對值改成標籤乘分數視為等式改寫 | 改問為何使用帶標籤分數，明示不同函數；保留零點可微性與分類方向的思考線索 | 25 |
| 第四題將「最大化所有正確點距離」與錯分損失最小化視為普遍等價 | 改為比較代數與錯分損失下降，在 η′=2η 後得到相同更新；同步釐清選項 A | 26 |

這些修正在簡報內明示，原始講義尚未更動；第 25–26 頁須特別審閱題意調整。未宣稱正確點更新必然使權重長度單調增長，也未將分類用分數誤當正規化幾何距離。

## 驗證

- 三輪瀏覽器渲染、檢視與修正。初版修正英文溢出、公式與標籤問題。教授指出視覺不夠活潑後，另完成三輪視覺改版：第一輪加入全 26 頁原生角色、對話與情境道具；第二輪用直角尺區分法向量、單刻度盤區分學習率、下坡球區分梯度、階梯區分逐步對照，移除自我檢核角色的答案提示；第三輪調整角色指圖、抓頭、張臂等動作。最終全 52 頁實際渲染並檢視，零溢出候選、零頁面錯誤。
- 中英文 1280×720 與 1920×1080 共四組，316 個步驟逐步驗證 reveal 節點可見、零 KaTeX 錯誤、零 JS 錯誤、零外部 HTTP 請求。見 `browser-verification.json`。
- 兩份稿通過原生結構與課程 check_deck_quality（page-width 1440、strict）及 check_slides。字級至少 16、雙語幾何／鏡頭相同、場景與 reveal 引用完整。見 `native-validation.json`、`structure-verification.json`。
- check_slides 的提醒：每語 78 個教學步驟含講者備註；備註也會包含於下載檔案。本稿備註為逐拍授課提示，無練習題答案。
- 程式驗算四種標籤／輸出、四點精確距離、垂直投影、法向量、非邊界有限差分梯度、偏置更新及分數 −0.5→1.5；以同一順序逐筆核對 KP02／KP03 共 16 次訪問權重相同、5 次更新、第四輪無更新。零距離／零損失但仍誤分類的反例亦通過。見 `數值驗算.json`。
- 最終 52 張完整截圖與雙語縮圖索引收於 `預覽截圖/`，逐頁版面檢查在 `逐頁排版檢查.json`。

限制：已驗證匯出的 HTML 與 JBD 資料／規則，尚未在實際 JianBauDau App 內播放或匯出 PDF；本工作區僅有來源與必要工具副本，未執行完整課程 repo doctor／網站建置。
'''
report=report.replace('製作時間：2026-10-30 10:55–11:08（Asia/Taipei）。','初版製作：2026-10-30 10:55–11:08（Asia/Taipei）；依教授意見完成活潑視覺改版。')
report=report.replace('無整頁圖片。','無整頁圖片。\n- 視覺改版：每頁右側預留約 20% 區域給固定「小感知器」；藍頭紫身橘天線，依概念換表情、姿勢與道具。這些插圖為教學比喻，不代表新增實驗數據；原科學圖等比縮放，所有公式完整保留。')
(O/'審閱說明.md').write_text(report)
for f in ('數值驗算.json','圖解提示詞.md','圖解證據表.json','視覺改版提示詞.md'):shutil.copy2(R/f,O/f)
shutil.copy2(R/'visual-round3/browser-audit.json',O/'逐頁排版檢查.json')
shots=O/'預覽截圖'
if shots.exists():shutil.rmtree(shots)
shots.mkdir(exist_ok=True)
for p in (R/'visual-round3').glob('*.png'):shutil.copy2(p,shots/p.name)
(R/'COURSE_STATE更新草稿.md').write_text('''# COURSE_STATE 更新草稿（未套用至 repo）

2026-10-30：IDL 第二章 KP03 中英文圖解簡報審閱稿完成，各 26 頁、79 步、每頁 3 拍；19 頁主線與 7 頁延伸。三輪視覺檢查、數值與離線 HTML 漸進出場驗證通過。零分邊界、分數損失與距離定義，以及自我檢核第 3–4 題釐清詳見審閱說明。等待教授審閱；未發布。輸出位於 /workspace/scratch/idl-lec02-kp03-review/output/。
''')
log=Path('/workspace/my-jarvis/logs/2026-10-30-idl-lec02-kp03-圖解簡報審閱稿.md');log.parent.mkdir(parents=True,exist_ok=True)
log.write_text('''# IDL 第二章 KP03 圖解簡報審閱稿

- 執行者：Codex；教授指示：繼續 KP03，沿用課程 repo page-to-teaching-diagram 與漸進出場。
- 時間：2026-10-30 10:55–11:08（Asia/Taipei），約 13 分鐘。
- 狀態：完成本機審閱稿，待教授審閱；未發布、未推送、未修改原講義。
- 輸入：GitHub ndhuccc/idl-2026-fall main 64e766a7019231ef86155a6a5550390832c217d9，lectures/lec02-kp03-signum-update-derivations.md。
- 輸出主機：本次 Codex workspace；完整路徑 /workspace/scratch/idl-lec02-kp03-review/output/；中英文各 26 頁，JBD／離線 HTML／審閱包。
- 證據：審閱說明.md、數值驗算.json、browser-verification.json、native-validation.json、structure-verification.json、逐頁排版檢查.json、預覽截圖/。
- 問題與修正：三輪檢視修正英文溢出、標籤折行、跳脫字元與空公式。講義零分邊界、損失與距離混稱、自我檢核第 3–4 題歧義只在稿內釐清，原講義未改。
- 驗證：52 頁最終渲染零溢出候選；4 種語言／視窗組合 316 步 reveal 驗證；數值及原生結構／課程檢查通過。78 個備註提醒已在報告說明。實際 JBD App／PDF 未測。
- 後續：教授審閱 KP03 後按指定繼續 KP04。
''')
with log.open('a') as f:f.write('\n## 依教授意見完成視覺改版\n\n教授指出原版缺乏活潑視覺元素。以 page-to-teaching-diagram 的角色與趣味插圖規則，加入全 26 頁原生小感知器、概念道具、情境表情與對話，並調整指圖／操作／困惑等姿勢；每頁三拍。另完成三輪實際渲染檢視，保留初稿備份 /workspace/scratch/idl-lec02-kp03-review-before-visual-revision/。改版證據在 視覺改版提示詞.md、visual-round1/ 至 visual-round3/，最終審閱包已更新。\n')
z=O/'IDL第二章KP03-審閱包.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in sorted(O.rglob('*')):
  if p.is_file() and p!=z:f.write(p,p.relative_to(O))
print(z, z.stat().st_size)
with zipfile.ZipFile(z) as f:assert f.testzip() is None;print('ZIP integrity PASS;',len(f.namelist()),'files')
