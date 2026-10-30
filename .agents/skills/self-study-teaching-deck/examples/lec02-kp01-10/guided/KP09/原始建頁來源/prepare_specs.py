from pathlib import Path
import json,hashlib
from specs import PAGES,MATH
R=Path(__file__).parent;m=json.loads((R/'skills-source-manifest.json').read_text());rows=[];out=[]
for f in m['files']:
 b=(R/'toolchain'/f['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['sha']
for i,p in enumerate(PAGES,1):
 kind,zh,en,tz,te,sec,visual,prop,mood,sz,se=p
 rows.append(dict(page=i,source_section=sec,title=zh,takeaway=tz,visual=visual,math=MATH.get(kind,[]),extension=i>24,source_role='原講義內容與算例；width為新增剖面概念圖，gaussian/grid為來源程式精算'))
 out.append(f'''## 第{i}頁／共{len(PAGES)}頁：{zh}
用途：IDL第二章KP09，大學生。唯一結論：{tz}；估約1分鐘，練習另計。前後頁依次銜接，保持藍紫綠橙角色。來源：ndhuccc/idl-2026-fall @ {m['commit']}，lectures/lec02-kp09-rbf-and-som.md，{sec}。
規格：1280×720，白底深色字；標題0–86，主圖60–1000×170–620，藍頭紫身橙觸角機器人與泡泡1025–1255×162–560，結論640–696。文字約中文60字／英文35詞，必要條件與算式不可刪，過密就拆頁。
逐區：標題「{zh}」／「{en}」。主視覺：{visual}。結論「{tz}」／「{te}」。角色表情{mood}，拿{prop}道具；泡泡「{sz}」／「{se}」。精算科學曲線與比喻角色區分，不能用概念示意替代原實驗。
數學：{'；'.join('$'+s+'$' for s in MATH.get(kind,[])) or '本頁新增標註見最終精確文案及建頁腳本；純概念頁無數值推導'}。所有符號、表頭、軸標與等式用KaTeX，不直接印出指令；分式與總和框至少110px，字級至少20，最低通用字級16。先保留完整精度，最後用近似號取位數。
原生：JBD schema v3，形狀／文字／多邊形／科學折線／connector。流程一定用connector，不用手畫箭頭；每個科學折線最多200點。資料藍、固定特徵紫、更新綠、條件與輸入橙；以形狀雙重編碼。XOR三角正類、圓負類。格子鄰接與資料空間分開，SOM網格64點112邊，模擬4000步、seed0，初值先抽常態，再依序抽均勻輸入。
漸進：先問題／圖，再證據與角色，最後一句帶走結論；三拍同步講者稿。更新頁暫時問題第二拍退場，鏡頭推近後回全景。兩語元件位置、數量、鏡頭與出場順序相同。
避免：手選XOR中心說成K-means結果、把四捨五入係數的輸出說成精確零一、漏偏置、混用硬鄰域與高斯鄰域、格子距離當輸入距離、SOM勝者索引當語意類別、每步鄰域對半、用單一seed宣稱普遍拓撲保證。練習及檢核不顯示答案，備註亦同。
檢查：來源與SHA、重現數值、每頁5秒主旨、條件、公式與結果同頁、連線、圖層、文字不溢出；至少三輪實際渲染、目視、批判、修正。精確最終文案與數值由build_kp09.py、kp09_nums.py及數值驗算.json提供。
''')
(R/'圖解提示詞.md').write_text('\n'.join(out));(R/'圖解證據表.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(len(PAGES),'specifications; 29 verified skill/tool blobs.')
