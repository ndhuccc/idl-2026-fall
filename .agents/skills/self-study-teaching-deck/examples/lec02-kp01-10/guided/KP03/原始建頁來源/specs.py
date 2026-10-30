from pathlib import Path
import json
from kp03_nums import save
ROOT=Path(__file__).parent
SOURCE='https://github.com/ndhuccc/idl-2026-fall/blob/64e766a7019231ef86155a6a5550390832c217d9/lectures/lec02-kp03-signum-update-derivations.md'
PAGES=[
('cover','KP03 正負標籤與更新推導','KP03 Signed labels and the update rule','代數與幾何兩條路，得到同一更新。','Algebra and geometry lead to the same update.','單元主張',r'左側正負一標籤進入代數差值，右側兩類散點與邊界法向量進入幾何路徑；兩支箭頭匯入中央規則$\mathbf{w}^{\prime}=\mathbf{w}+2\eta y\mathbf{x}$，旁邊標「只在猜錯時」。'),
('coding','換成正負一，也要換成符號輸出','Signed labels require a signed output','約定：總和等於零時，輸出正一。','Convention: a zero score outputs plus one.','符號表、§1',r'零與一兩個標籤分別映射成負一與正一；右側精確階梯：$o=\operatorname{sign}(s)$，$o=-1$當$s<0$，$o=+1$當$s\ge0$。零點下方空心、上方實心。$s$為加權總和，$o$為輸出。'),
('four','同號差零，異號只會差正負二','Equal signs give zero; opposite signs give plus or minus two','四種組合，已經決定全部更新情況。','Four combinations determine every update case.','§1',r'用二乘二矩陣同時比較$y=+1,-1$與$o=+1,-1$；兩個對角格顯示$y-o=0$不更新，另兩格顯示$1-(-1)=2$、$-1-1=-2$。同號圈、異號箭頭雙重編碼，不用四列密集表格。'),
('algebra','猜錯時，誤差剛好等於標籤的兩倍','On a mistake, the error is twice the label','係數二來自反號標籤的差。','The factor two comes from subtracting opposite signs.','§1',r'分步顯示「只在$o\ne y$」：$o=-y$；$y-o=y-(-y)=2y$；$\Delta\mathbf{w}=\eta(y-o)\mathbf{x}=2\eta y\mathbf{x}$。兩個反號圓點以相減箭頭變成兩份同方向的$y$，把倍數畫成圖。'),
('rule','把二併進學習率，寫法就一致了','Absorb the factor two into the learning rate','偏置也一起改；只更新猜錯的點。','Update the bias too, and only on a mistake.','§1',r'左右兩路規則比較$\mathbf{w}^{\prime}=\mathbf{w}+2\eta y\mathbf{x}$與$\mathbf{w}^{\prime}=\mathbf{w}+\eta^{\prime}y\mathbf{x}$，橋接$\eta^{\prime}=2\eta$；本頁更新用擴增向量$\mathbf{x}=(x_1,x_2,1)$、$\mathbf{w}=(w_1,w_2,w_0)$。底部獨立顯示$w_0^{\prime}=w_0+2\eta y$。'),
('normal','權重向量，垂直於決策邊界','The weight vector is perpendicular to the boundary','法向量，就是與邊界垂直的向量。','A normal vector is perpendicular to the boundary.','§2',r'精確直線$x_1+x_2=1.5$；沿線選示例點$(0.5,1)$、$(1,0.5)$，位移$(0.5,-0.5)$；權重$\mathbf{w}=(1,1)$垂直向右上，畫直角記號。通式$\mathbf{w}^{\top}(\mathbf{x}_A-\mathbf{x}_B)=0$來自兩點邊界式相減。此頁幾何權重不含偏置。'),
('distance','把分數除以權重長度，才得到距離','Divide the score by the weight norm to obtain distance','幾何距離用的權重，不含偏置。','The geometric weight vector excludes the bias.','§2',r'點$\mathbf{x}$向直線作垂足，畫精確最短線段，沿法向量方向標帶符號距離$r$。完整公式$r=(\mathbf{w}^{\top}\mathbf{x}+w_0)/\|\mathbf{w}\|$；$\|\mathbf{w}\|>0$；$s=\mathbf{w}^{\top}\mathbf{x}+w_0$是未正規化的分數。用同一比例的水平與垂直座標，避免圖中直角失真。'),
('labeldistance','再乘正確標籤，正負才表示是否在對的一側','Multiply by the label to test the correct side','離開邊界時，正距離對，負距離錯。','Away from the boundary: positive is right, negative is wrong.','§2',r'使用OR四點與$\mathbf{w}=(1,1),w_0=-1.5$。圈為$y=-1$、三角為$y=+1$；兩個錯分三角紅圈，投影線與正確點顏色區分。公式$d=yr=ys/\|\mathbf{w}\|$，標記$s\ne0$才有$d>0$對、$d<0$錯的雙向判準；邊界另頁處理。'),
('tie','邊界上的負類點，距離零仍然猜錯','A negative point on the boundary is wrong even at zero distance','補充釐清：零距離不能直接判定猜對。','Clarification: zero distance alone does not establish correctness.','§2邊界釐清',r'反例：$\mathbf{w}=(1,1),w_0=0$、原點$\mathbf{x}=(0,0)$、$y=-1$。原點落在$x_1+x_2=0$上，$s=0$、$o=\operatorname{sign}(0)=+1$，與$y$不同，卻$d=0$。加「補充釐清」標籤；不修改來源講義，審閱報告列出矛盾。'),
('distancesA','原點在對的一側，左上點卻在錯的一側','The origin is correct, but the upper-left point is not','用精確根號計算，最後才取兩位小數。','Use the exact square root; round only the final result.','§3距離',r'同一座標圖與線$x_1+x_2=1.5$，強調$(0,0)$、$(0,1)$兩點與各自垂足。$\|\mathbf{w}\|=\sqrt{2}$；原點$d=(-1)(-1.5)/\sqrt2\approx1.06$，左上點$d=(+1)(-0.5)/\sqrt2\approx-0.35$；總和由$x_1+x_2-1.5$逐點算出。'),
('distancesB','右下點猜錯，右上點已經猜對','The lower-right point is wrong; the upper-right is correct','對錯取決於帶標籤的距離正負。','Correctness depends on the label-weighted distance sign.','§3距離',r'同一線與座標，強調$(1,0)$、$(1,1)$。分數分別$s=1+0-1.5=-0.5$與$s=1+1-1.5=0.5$；$y=+1$，距離分別$-0.5/\sqrt2\approx-0.35$與$0.5/\sqrt2\approx0.35$；清楚保留公式與分母。'),
('loss','只把錯分點的負分數，加進損失','The loss sums the negative scores of misclassified points','這是分數損失，不是距離的總和。','This is a score-based loss, not a sum of distances.','§2、來源更正',r'兩個錯點$(0,1),(1,0)$由紅圈流入加總器；$M$為錯分點集合，$E=-\sum_{i\in M}y_i s_i$，$s_i=\mathbf{w}^{\top}\mathbf{x}_i+w_0$。本例$E=-1(-0.5)-1(-0.5)=1$。同頁小標「邊界錯分可能貢獻零」；不宣稱有錯時E必為正，或E為零就能停止。'),
('gradient','錯分損失的下降方向，是加上標籤乘輸入','Descending the loss adds label times input','在非邊界處，先把錯分集合固定來求導。','Away from the boundary, differentiate with a fixed mistake set.','§2',r'逐步顯示$\nabla_{\mathbf{w}}E=-\sum_{i\in M}y_i\mathbf{x}_i$與$\partial E/\partial w_0=-\sum_{i\in M}y_i$；以單一錯點$(0,1)$、$y=+1$作向量箭頭示意，負梯度沿正的輸入方向。幾何權重不含偏置，邊界處標「可能沒有唯一梯度」，備註解釋按符號約定選擇次梯度。'),
('descent','減掉負梯度，便得到和代數相同的規則','Subtracting the negative gradient gives the algebraic rule','兩條路相同，學習率名稱要對齊。','The rules agree after matching the learning-rate names.','§2',r'完整三步：$\mathbf{w}^{\prime}=\mathbf{w}-\eta^{\prime}\nabla E$；$=\mathbf{w}+\eta^{\prime}\sum_{i\in M}y_i\mathbf{x}_i$；單一錯點且$\eta^{\prime}=2\eta$時$\mathbf{w}^{\prime}=\mathbf{w}+2\eta y_i\mathbf{x}_i$。圖形是代數與幾何兩支路合流，不宣稱最大化所有正確距離是同一個一般最佳化目標。'),
('numeric_update','修正錯點，兩個權重與偏置分別怎麼改','How one mistake changes the weights and bias','本例只把第二路權重與偏置往上調。','Here only the second input weight and bias increase.','§3一次更新',r'輸入$(0,1)$、標籤$y=+1$、$\eta=0.5$，舊幾何權重$(1,1)$、偏置$-1.5$；擴增向量$(0,1,1)$、舊完整權重$(1,1,-1.5)$。完整$\mathbf{w}_{\rm aug}^{\prime}=(1,1,-1.5)+2(0.5)(+1)(0,1,1)=(1,2,-0.5)$。三路舊值藍點、新值紫圈，更新方向綠箭頭。三拍依序呈現資料與舊值、修正、新值與結論。'),
('numeric_score','更新後，這一點從負分跨到正分','The update moves this point from a negative to a positive score','這次跨過了零，但一般不保證一次就改對。','This update crosses zero; one-step correction is not guaranteed in general.','§3一次更新',r'兩個等尺度面板：舊線$x_1+x_2=1.5$與新線$x_1+2x_2=0.5$，同一個點$(0,1)$紅圈變綠圈。完整$s_{\rm old}=1(0)+1(1)-1.5=-0.5$與$s_{\rm new}=1(0)+2(1)-0.5=1.5$。差值$1.5-(-0.5)=2=2(0.5)(1+1)$，最後一項包含偏置輸入的一。'),
('trace','學習率取一半，逐筆軌跡就和 KP02 相同','Half the rate reproduces every KP02 weight update','五次更新、第四輪停止，權重逐筆相同。','Five updates, stopping on pass four, with identical weight vectors.','§3軌跡',r'上方$2\eta=1$、本節$\eta=0.5$與KP02學習率1；資料順序固定，零初始化，sign(0)=+1。時間路徑五個狀態$(0,0,-1)$、$(0,1,0)$、$(0,1,-1)$、$(1,1,0)$、$(1,1,-1)$，各標更新序號與訪問點；全部由程式重現16次訪問，頁面標〔程式計算〕，更新公式完整保留。'),
('gate','猜錯才打開更新開關，猜對必須不動','Open the update gate only for a mistake','更新條件不能從公式旁邊省掉。','The mistake condition is an essential part of the rule.','重點整理、迷思',r'同號對角通道$o=y$流向$\Delta\mathbf{w}=\mathbf0$；反號通道$o\ne y$流向$\Delta\mathbf{w}=2\eta y\mathbf{x}$。圖中大菱形閘門決定兩條路，不宣稱更新所有正確點一定讓權重範數單調增長。'),
('next','兩條路得到同一規則，但一條線能分開所有資料嗎','Two routes give one rule, but can one line separate every dataset?','下一個 KP：感知器的能力邊界。','Next: the limits of the Perceptron.','留給下一知識點問題',r'代數「反號差兩倍」與幾何「錯分損失下降」兩支路匯入$\mathbf{w}^{\prime}=\mathbf{w}+2\eta y\mathbf{x}$且只在猜錯時；右側四個交錯類別點與問號，表示下一節的可分性問題，不展開KP04的證明。'),
('practice1','延伸：換成另一個錯點，更新後分數是多少','Practice: update the other misclassified point','提示：先更新權重與偏置，再代回輸入。','Hint: update weights and bias, then recompute the score.','小練習1',r'條件$\mathbf{w}=(1,1),w_0=-1.5$、$\mathbf{x}=(1,0)$、$y=+1$、$\eta=0.5$。畫兩條輸入與偏置進入更新閘，輸出新權重與分數均為問號。不放答案。'),
('practice2','延伸：原點離這條邊界多遠','Practice: how far is the origin from this boundary?','提示：先算權重長度，再算距離。','Hint: compute the weight norm before the distance.','小練習2',r'精確直線$3x_1+4x_2-5=0$；原點至線的垂足與直角記號；條件$\mathbf{w}=(3,4),w_0=-5$。原題問無符號幾何距離，提示$|w_0|/\|\mathbf{w}\|$，不加標籤y、不顯示距離數值。'),
('practice3','延伸：誤差為什麼不會是正負一','Practice: why can the error not be plus or minus one?','提示：把四種標籤與輸出配對列完。','Hint: list all four label/output pairs.','小練習3',r'四格$(y,o)=(1,1),(1,-1),(-1,1),(-1,-1)$，相減器輸出全部用問號；要求列出$e=y-o$，不填解答。'),
('quiz1','自我檢核：符號輸出可以取哪些值','Self-check: which values can a sign output take?','提示：回想符號函數的兩個輸出。','Hint: recall the two outputs of the sign function.','自我檢核1',r'四個等大候選燈與選項：A只有0；B只有+1；C +1或-1；D任何實數；數學值全部LaTeX，不標正解。'),
('quiz2','自我檢核：反號時，更新項能化簡成什麼','Self-check: how does the update simplify on a mistake?','提示：先用輸出等於負標籤來代入。','Hint: substitute output equal to the negative label.','自我檢核2',r'條件$y,o\in\{-1,+1\},o\ne y$與待化簡項$\eta(y-o)\mathbf{x}$；四個候選：A $\eta y\mathbf{x}$；B $2\eta y\mathbf{x}$；C $-2\eta\mathbf{x}$；D 0；等大小、不標對錯。'),
('quiz3','自我檢核：為什麼不用絕對值當這個分數','Self-check: why use a label-weighted score rather than an absolute value?','提示：想想零點的可微性與分類方向。','Hint: consider differentiability at zero and class direction.','自我檢核3〔原題釐清〕',r'對照$|s|$與$ys$兩個不同函數的小圖，標「不同函數，不是相等改寫」。四選項：A讓距離變大；B絕對值在零點不可微；C讓權重成單位向量；D讓誤差變小。不把未除範數的分數當幾何距離。'),
('quiz4','自我檢核：兩條推導如何得到相同更新','Self-check: how do the two derivations yield the same update?','提示：對齊兩種寫法的學習率。','Hint: match the learning-rate definitions.','自我檢核4〔原題釐清〕',r'左代數路、右錯分損失梯度路，各接同一個待判斷更新問號。四選項：A調整學習率定義後相同；B方向相反；C差一個負號；D無法比較。避免宣稱最大化所有正確點距離等於標準感知器目標。')]

def main():
    save();out=[];prompts=[]
    for i,(kind,zh,en,tz,te,sec,brief) in enumerate(PAGES,1):
        out.append(dict(page=i,kind=kind,title_zh=zh,title_en=en,takeaway_zh=tz,takeaway_en=te,source=SOURCE,section=sec,figure_brief=brief,extension=i>19))
        prompts.append(rf'''## 第{i}頁／共{len(PAGES)}頁
唯一訊息：{tz}；英文：{te}。講授約1–2分鐘；約60中文字／35英文詞，超量拆頁，不刪前提。
用途：IDL第二章KP03，大學生教學示意圖。
模式：依文字規格直接製作可編輯原生頁，不先生成圖片。
來源：{SOURCE}，{sec}；完整內容與版面在下方，不要求製圖端讀網址。
核心結論：{tz}
規格：1280×720，白底、深藍字；中英分檔；輸入藍、權重紫、正確綠、錯誤紅、門檻與步長橘；一般字至少20px，註記至少16px。
版面：標題0–86px，主圖110–615px佔最大區域，底部640–696px結論。每頁2–3拍，先條件、再機制或計算、最後結論；公式放獨立全寬框。分數與求和框至少110px高。
逐區內容：標題「{zh}」（英文「{en}」）；{brief}；底部「{tz}」（英文「{te}」）。
圖例：圓點為負類、三角為正類；紅圈是錯分，綠色區是分數非負的預測正類區；箭頭用來表示流程、法向量或更新方向，同頁標明其意義。問題選項不突出正解。
數學排版：全部符號與公式用LaTeX。$y$是標籤、$o$輸出、$s$分數、$eta>0$學習率；幾何頁$mathbf{{w}}$不含$w_0$；更新向量若含偏置，明列$(x_1,x_2,1)$。距離分母必須是空間權重的範數，不是擴增範數。正確渲染LaTeX，不顯示指令；不用cases、不用中文	ext。
限制條件：$operatorname{{sign}}(0)=+1$；負類點在零分邊界會錯分但距離和分數損失可為零；距離公式要求$|mathbf{{w}}|>0$；錯分損失為未正規化分數，不是除過範數的幾何距離。非邊界處取梯度，邊界按符號約定選次梯度。練習與檢核只提示、不放答案。
底部重點：{tz}
避免：把零距離直接說成正確、把零損失當停止條件、把所有資料都納入感知器更新、距離用擴增權重算、幾何直角因縮放扭曲、未說明係數二與學習率、文字遮蓋與密集卡片。
檢查：四組標籤差值、四個距離、單步更新與分數增加、16筆OR訓練軌跡及零邊界反例均由kp03_nums.py驗算；精確值最後才四捨五入。
原生元件：橢圓作負類點與標籤，三角作正類與箭頭頭，菱形作更新閘，折線畫座標、邊界、投影與法向量，多邊形作半透明預測區；文字另開高圖層節點。不使用image節點。
''')
    (ROOT/'圖解證據表.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));(ROOT/'圖解提示詞.md').write_text('\n'.join(prompts))
    print(len(PAGES),'self-contained prompts saved before drawing')
if __name__=='__main__':main()
