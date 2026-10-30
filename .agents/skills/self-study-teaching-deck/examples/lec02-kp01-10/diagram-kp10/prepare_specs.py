from pathlib import Path
import json
from specs import PAGES
R=Path(__file__).parent
(R/'圖解證據表.json').write_text(json.dumps(PAGES,ensure_ascii=False,indent=2))
lines=[]
for p in PAGES:
 lines.append(f'''## 第{p['page']}頁／共{len(PAGES)}頁：{p['title']}
用途：IDL第二章KP10，大學生自學；圖解版，不用吉祥物。來源錨點：{'、'.join(p['source_anchors'])}。
唯一結論：{p['takeaway']}。承接：{p['bridge']}。
規格：1280×720白／淺底深字，標題32–36，右上頁碼18；小標18–20；主視覺全寬至少約60%，主體50–1230×150–630；結論640–700深藍白字24。標註至少18、公式至少24，文字z30高於實心形狀。
逐區：標題「{p['title']}」／「{p['title_en']}」；主圖{p['visual']}；底部「{p['takeaway']}」／「{p['takeaway_en']}」。
數學：{'；'.join('$'+m+'$' for m in p['math']) or '本頁不新增數值公式；必要符號及最終生成式見原生文字節點'}。
元件：原生shape/text/polyline/connector。精算曲線每節不超200點，不使用圖片；目標紫、近似藍、誤差與條件橙、理想界綠，形狀與顏色雙編碼；星形標中點，虛線標分格，圖像比喻明示為示意。流程一律原生connector。
三拍：問題與主圖→證據／算式→結論；每拍自學稿依序提問、解釋、檢查。完整白話導讀：{p['notes'][0]} 英文：{p['notes'][1]}。
避免：圖像只有方框文字、把有限sigmoid當精確硬階梯、把取樣最大誤差當已證sup、把N格和N−1跳躍混淆、漏常數、存在等於可學、把ReLU硬套有界版本、把證明骨架說成完整證明。題目頁不洩露答案，解析緊接下一頁；使用者已指定新技能的解析頁政策。
檢查：同頁結果有算式、來源／新增清楚、數字由kp10_nums.py算出，必要條件不刪；無文字溢出、無遮字、公式不裁切；中英幾何與順序一致；三輪真實渲染目視修正，coverage/style/quality/course檢查通過。
''')
(R/'圖解提示詞.md').write_text('\n'.join(lines));print(len(PAGES),'pages specified before native drawing.')
