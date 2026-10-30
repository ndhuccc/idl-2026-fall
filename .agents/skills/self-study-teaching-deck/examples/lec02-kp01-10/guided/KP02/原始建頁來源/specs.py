from pathlib import Path
import json
from kp02_nums import ONE,save
ROOT=Path(__file__).parent
SOURCE='https://github.com/ndhuccc/idl-2026-fall/blob/64e766a7019231ef86155a6a5550390832c217d9/lectures/lec02-kp02-perceptron-learning-rule.md'
# kind, title zh/en, conclusion zh/en, source section, complete figure specification.
PAGES=[
('cover','KP02 猜錯才改權重','KP02 Update only when wrong','先算、再比，猜錯才改。','Predict, compare, and update only on a mistake.','§1','輸入圓點流向神經元，再到預測燈；正確標籤由上方接入比較菱形；錯誤紅箭頭回到權重旋鈕，正確綠箭頭直接往下一筆。'),
('supervised','正確答案提供回饋，感知器才知道怎麼改','Labels tell the Perceptron how to change','監督式學習：修正由標籤與錯誤驅動。','Supervised learning uses labels and mistakes.','§1','Rosenblatt於1958年提出感知器；左側權重旋鈕與模型，右側兩盞等大燈分別是預測$o$與標籤$y$，比較後紅色回饋箭頭回旋鈕。與古典Hebbian規則不同，細節在講者備註。'),
('bias','補上一個固定輸入，偏置也能一起學','Add a constant input to learn the bias too','把偏置當成第三個權重，一起更新。','Treat the bias as a third weight and update it too.','符號表',r'三條並行路徑：$x_1$乘$w_1$、$x_2$乘$w_2$、固定$x_0=1$乘$w_0$，匯入總和$s$。明列$\mathbf{x}=(x_1,x_2,1)$、$\mathbf{w}=(w_1,w_2,w_0)$、$s=w_1x_1+w_2x_2+w_0$；$w_0$為偏置。'),
('loop','每看一筆資料，都先算再比對','For each sample, predict before comparing','猜對就前進，猜錯才繞回修正。','Correct predictions move on; mistakes trigger an update.','§1',r'依序畫輸入$\mathbf{x}_i$、總和$s=\mathbf{w}^{\top}\mathbf{x}_i$、階梯輸出$o_i=f(s)$、比較標籤$y_i$、誤差$e_i=y_i-o_i$。圖例$f(s)=1$當$s\ge0$，否則0；錯誤沿回饋箭頭更新，正確去下一筆。'),
('errors','三種誤差，決定三種動作','Three errors select three actions','猜對不動；少放電就加，多放電就減。','Keep correct weights; add when too low and subtract when too high.','§2',r'三條通道：$e=y-o=0$對應不動；$e=1-0=+1$對應$\mathbf{w}+\eta\mathbf{x}$；$e=0-1=-1$對應$\mathbf{w}-\eta\mathbf{x}$。每列以不同方向箭頭和輸出燈表示；$y$是正解、$o$是預測、$\eta>0$為學習率。'),
('rule','誤差乘上輸入，就是每路的修正方向','Error times input gives each weight correction','輸入為零的那一路，權重不動。','An input of zero leaves its weight unchanged.','§2、重點整理',r'完整規則$\mathbf{w}^{\prime}=\mathbf{w}+\eta e\mathbf{x}$，$e=y-o$。三個帶旋鈕的路徑分別代表$x_1,x_2,x_0=1$；式子每項以同色線連到對應輸入與修正量$\Delta w_j=\eta e x_j$。$\eta>0$決定步長。'),
('direction','一次修正，會把這一筆往正確方向推','One update moves this sample in the right direction','只保證方向正確，不保證別筆也變好。','The direction improves for this sample, not necessarily for others.','§2',r'完整三步推導：$s^{\prime}=\mathbf{w}^{\prime\top}\mathbf{x}$；$=(\mathbf{w}+\eta e\mathbf{x})^{\top}\mathbf{x}$；$=s+\eta e\|\mathbf{x}\|^2$。因$x_0=1$所以$\|\mathbf{x}\|^2\ge1$。數線畫原$s$向右與向左的兩個分支，分別標$e=+1$與$e=-1$，箭頭長度為正量示意，非精確數值。'),
('dataset','用固定順序，從全零開始學 OR','Learn OR from zeros in a fixed order','示例：權重全零，學習率是一。','Example: start with zero weights and learning rate one.','§3〔補充〕',r'精確四點座標：$(0,0)\to0$、$(0,1)\to1$、$(1,0)\to1$、$(1,1)\to1$，用圈與三角雙重編碼標籤。每點旁有訪問順序1–4；$\mathbf{w}=(0,0,0)$、$\eta=1$、$\mathbf{x}=(x_1,x_2,1)$，階梯在$s=0$輸出1。'),
]
for i,r in enumerate(ONE['updates'],1):
    a,b=r['pair'];s=r['s'];y=r['y'];o=r['o'];e=r['e']
    zh=['先壓低偏置，修正不該放電的原點','第二路與偏置一起增加，修正漏掉的放電','原點又猜錯，偏置再往下降','第一路與偏置一起增加，修正另一筆','最後壓低偏置，四種輸入終於分開'][i-1]
    en=['Lower the bias to silence the origin','Raise the second weight and the bias','The origin is wrong again: lower the bias','Raise the first weight and the bias','Lower the bias once more to separate all four'][i-1]
    PAGES.append((f'update{i}',f'第{i}次更新：'+zh,f'Update {i}: '+en,'把這一筆的誤差，分配到有輸入的路徑。','Apply this sample’s error to the active input paths.','§3〔補充〕',f'第{r["epoch"]}輪，輸入{r["pair"]}、擴增向量{r["x"]}、正確標籤{y}；舊權重{r["old"]}；s={s}、o={o}、e=y-o={e}；學習率1；修正量{r["dw"]}；新權重{r["new"]}；同一筆修正後s={r["s_after"]}。完整向量代入式；三路權重用零線長條與正負箭頭畫出舊、新與修正，長度每單位80px。三拍依序顯示舊預測、修正、新權重及結論。'))
PAGES += [
('boundaries','三輪後，分隔線才把四個點分對','After three passes, the line separates all four','每輪結束的權重不同，邊界也跟著移。','Each pass ends with different weights and a different boundary.','§3、lec02-kp02-or-training',r'三個等尺度座標圖，依次為第1輪$(0,1,0)$的$x_2=0$、第2輪$(1,1,0)$的$x_1+x_2=0$、第3輪$(1,1,-1)$的$x_1+x_2=1$。填色為$s\ge0$；資料形狀表示標籤；前兩圖原點紅圈表示錯分。座標與裁切由程式精算。'),
('stop','整輪沒有猜錯，才確認可以停下來','Stop only after a complete pass without mistakes','前三輪改了五次，第四輪零更新才停止。','Five updates occur in three passes; pass four confirms zero updates.','§3、§4','沿時間軸四個輪次，更新次數分別2、2、1、0，總和$2+2+1=5$；最後權重$(1,1,-1)$。四個資料點依序流過驗證閘，標記四筆都滿足正確標籤；不換成百分比。'),
('interference','修好原點，另一筆更新又能把它改壞','Another sample can spoil a corrected point','一次修正，只保證目前這一筆往對方向走。','An update improves the current sample’s score direction only.','重點整理、迷思','使用實際第1與第2次更新：原點$s=w_0$先由0變-1，再因$(0,1)$的更新變回0。三個時間點用原點圓與亮暗燈；三個權重分別$(0,0,0)$、$(0,0,-1)$、$(0,1,0)$。$f(0)=1$，因此原點再被錯判。'),
('separable','能用一條線分開，才有有限次停止的保證','Finite convergence needs linearly separable data','補充：資料不可分時，不能期待零錯誤停止。','Added: nonseparable data cannot reach a zero-error pass.','§4〔補充〕','左右兩個二元資料平面：左OR以$x_1+x_2=1$分開；右XOR兩類交錯，一條線不能分開。清楚標「補充、幾何示意」，兩類用圈和三角雙重編碼；不畫錯誤可分直線。不假稱有限步證明。停止規則指循環看全部固定訓練資料、無上限時的零錯誤停止。'),
('rate','從全零開始，正的學習率只會縮放權重','From zero, a positive learning rate only scales the weights','補充：順序固定、起點為零，預測不變。','Added: fixed order and zero start give the same predictions.','§4〔補充〕',r'兩個精確相同的OR邊界，左$\eta=1$、$\mathbf{w}=(1,1,-1)$；右$\eta=2$、$\mathbf{w}=(2,2,-2)$。逐筆權重關係$\mathbf{w}^{(2)}=2\mathbf{w}^{(1)}$，因此$s^{(2)}=2s^{(1)}$，含零的正負判斷相同；限定零初始化、固定正學習率、相同輸入順序。'),
('batch','批次先把修正量加起來，再更新一次','A batch sums the corrections before one update','單筆就是批次大小等於一的情況。','A batch of size one gives the single-sample rule.','§2',r'三筆樣本以三個小輸入群作概念示意，分別形成誤差項$(y_i-o_i)\mathbf{x}_i$，箭頭匯入加總器，再乘學習率更新。完整式$\mathbf{w}^{\prime}=\mathbf{w}+\eta\sum_{i=1}^{b}(y_i-o_i)\mathbf{x}_i$；$b$是批次筆數；這些誤差用同一組更新前權重算，不宣稱與連續逐筆更新結果相同。'),
('next','把標籤換成正負一，下一步看幾何','Next, signed labels will reveal the geometry','本節用零與一，KP03 改用正負一。','This unit uses zero and one; KP03 uses signed labels.','留給下一知識點問題',r'左側零與一標籤流入本節更新規則，右側正負一標籤流向問號與向量方向圖；符號$\mathbf{w}^{\prime}=\mathbf{w}+\eta(y-o)\mathbf{x}$完整保留。指出尚未推導下一節公式，不偷加結論。'),
('practice1','延伸：這一筆要不要更新','Practice: does this sample need an update?','提示：先算總和，再看正解減預測。','Hint: compute the score, then subtract prediction from label.','小練習1',r'原題權重$(1,1,-1)$、輸入$(1,0)$、標籤$y=0$、$\eta=1$。三個權重旋鈕連到總和、輸出問號與更新後權重問號。保留問題「輸出？要更新嗎？更新後權重？」不放答案。'),
('practice2','延伸：換了資料順序，哪一筆先更新','Practice: which sample updates first in a new order?','提示：全零權重時，四筆的總和都是零。','Hint: zero weights give a score of zero for every sample.','小練習2','新的OR資料順序$(1,1)$、$(1,0)$、$(0,1)$、$(0,0)$，各附標籤1、1、1、0；從$(0,0,0)$開始。四個依序站點與未知的第一個更新位置；提示$s=0$時$o=1$，不標正確站點。'),
('practice3','延伸：學習率變兩倍，預測為什麼不變','Practice: why does doubling the rate preserve predictions?','提示：檢查每一步的權重是否差兩倍。','Hint: compare the weight vectors at each step.','小練習3','兩路訓練流程由同一個零權重出發，上方學習率1、下方2；資料順序相同；中間權重以問號呈現，要求檢查比例，不在練習頁給結果。'),
('quiz1','自我檢核：誰在一九五八年提出感知器','Self-check: who proposed the Perceptron in 1958?','提示：回想感知器加入回饋的歷史。','Hint: recall the history of adding feedback.','自我檢核1','四個等大人物圖示與完整選項：A Widrow與Hoff；B Rosenblatt（Cornell Aeronautical Laboratory）；C McCulloch與Pitts；D Minsky與Papert。各選項等色，不標答案。'),
('quiz2','自我檢核：這和 Hebbian 規則有何不同','Self-check: how does this differ from Hebbian learning?','提示：看看有沒有使用正確標籤。','Hint: check whether correct labels are used.','自我檢核2','四個等大候選關係圖：A完全相同；B監督式，本質與古典Hebbian不同；C完全不學習；D Hebbian是後繼。每個選項搭配不表示對錯的圖示，完整保留文字。'),
('quiz3','自我檢核：猜對時，權重會怎麼變','Self-check: what happens to weights after a correct guess?','提示：先算正確標籤減預測。','Hint: compute label minus prediction.','自我檢核3',r'條件$o_i=y_i$與完整更新式。四個等大旋鈕候選：A權重放大$\eta$倍；B更新量為$\mathbf{0}$，不更新；C權重反轉；D學習率歸零。不得加正誤或突出顏色。'),
('quiz4','自我檢核：批次版怎麼回到單筆版','Self-check: how does a batch become a single-sample update?','提示：讓批次筆數等於一。','Hint: set the batch size to one.','自我檢核4','上方批次加總器與$b=1$；四個選項：A互相矛盾；B批次對各樣本誤差項求和，單筆去掉求和；C批次版不收斂；D只在分類時等價。完整公式在題幹同頁；不放答案。')]

def main():
    save();evidence=[];prompts=[]
    for i,(kind,zh,en,tz,te,sec,brief) in enumerate(PAGES,1):
        evidence.append(dict(page=i,kind=kind,title_zh=zh,title_en=en,takeaway_zh=tz,takeaway_en=te,source=SOURCE,section=sec,figure_brief=brief,extension=i>20))
        prompts.append(rf'''## 第{i}頁／共{len(PAGES)}頁：{zh}
唯一訊息：{tz}；英文：{te}。預估1–2分鐘；約60中文字／35英文詞，超量則拆頁，不刪計算條件。
用途：大學生深度學習導論，第二章KP02教學圖解。
模式：依規格直接建JianBauDau原生頁，不先生圖。
來源：{SOURCE}，{sec}；所需內容已完整列於下方。
核心結論：{tz}
規格：1280×720；中英文分檔；白底深藍字；輸入藍、權重紫、正確綠、錯誤紅、步長與偏置橘。文字至少20px，輔助標記至少16px。
版面：上方86px標題，中央110–615px圖解最大區域；底部640–696px結論。每頁2–3拍：問題或舊狀態、證據與修正、結論。圖形與文字分節點，不靠文字卡堆疊。
逐區內容：標題「{zh}」（英文「{en}」）；{brief}；底部「{tz}」（英文「{te}」）。
圖例：圈與三角區分真實標籤；紅圈只標錯分；半透明綠區表示$s\ge0$。箭頭區分訊號、回饋與更新，正負方向配文字。
數學排版：所有數學物件以LaTeX呈現；使用$\mathbf{{x}}=(x_1,x_2,1)$、$\mathbf{{w}}=(w_1,w_2,w_0)$、$s=\mathbf{{w}}^\top\mathbf{{x}}$、$o=f(s)$、$e=y-o$、$\eta>0$、$\mathbf{{w}}^\prime=\mathbf{{w}}+\eta e\mathbf{{x}}$。$y$正確標籤、$o$預測、$w_0$偏置、$\eta$學習率；本頁用到者就地定義。正確渲染LaTeX，不顯示原始指令。
限制條件：輸入向量含固定偏置輸入1；階梯在$s\ge0$輸出1，否則0；本例按固定資料順序、零初始化、學習率1。收斂保證只對可分的有限資料與標準逐筆流程；不宣稱每次更新立即改對或不再改錯。練習不放答案。
底部重點：{tz}
避免：四捨五入錯誤、把輪數當更新次數、把批次當同樣結果的逐筆更新、把正方向當一次改對、圖文不一致、文字被蓋住、曲線跑出座標區。
檢查：與kp02_nums.py的完整16筆訪問、五次更新、最後權重、輪次與同比例學習率結果逐一核對。中英文位置與拍數一致；圖層文字最高；三輪渲染後再交付。
原生元件：橢圓作輸入、旋鈕與類別零；三角形作類別一與箭頭頭；菱形作比較；多邊形填達標區；折線作坐標、連接線、邊界與更新方向；矩形作有比例的權重長條。無圖片節點，題目選項圖示不洩漏答案。
''')
    (ROOT/'圖解證據表.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    (ROOT/'圖解提示詞.md').write_text('\n'.join(prompts))
    print(f'{len(PAGES)} complete figure prompts saved before page construction')
if __name__=='__main__':main()
