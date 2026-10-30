from pathlib import Path
import json
ROOT=Path(__file__).parent
# Unique claim, source, complete visual requirements, and short role-based dialogue are fixed before drawing.
PAGES=[
('cover','KP04 感知器的限制與泛化','KP04 Perceptron limits and generalization','分得開，與分得好，是兩道關卡。','Separating the data and generalizing are different tests.','單元主張','兩條分路：左方XOR四點與直尺，右方訓練全對到新點問號；並排表示力與泛化兩種限制。','warning','confused','一條線，\n真的夠嗎？','Is one line\nalways enough?'),
('xor','同類點在對角，XOR 就是這個難題','XOR places each class on opposite corners','兩個輸入不同，正解才是一。','The XOR label is one only when the inputs differ.','§1','精確單位方形四點(0,0),(1,1)為0圓；(0,1),(1,0)為1三角，附完整座標與標籤，類別固定藍圓綠三角。','tags','think','先看正解，\n別急著畫線。','Read the labels\nbefore drawing.'),
('lines','換一條線，還是會留下錯點','A different line still leaves mistakes','這兩條嘗試都不全對；普遍結論另有證明。','Neither candidate works; the general claim needs proof.','§1圖說釐清','等比例並排：直線x1+x2=0.5預測上方為1，錯1點；直線x1+x2=1.5同方向預測，錯3點。完整標示線式與紅圈錯分，不沿用來源圖說的2錯。','ruler','confused','換線，\n錯點也換了。','New line,\nnew mistakes.'),
('updates','十輪過去，每輪更新仍未歸零','After ten passes, updates never reach zero','這次實跑：前兩輪三次，之後每輪四次。','This run has three updates twice, then four each pass.','§1實跑','從全零起點、eta=1、s>=0輸出1；順序00,01,10,11。原生長條圖實算10輪3,3,4,4,4,4,4,4,4,4；縱軸0到4，橫軸輪次1到10。','steps','confused','不是沒在學，\n是一直改。','Still updating,\nnever finishing.'),
('cycle','第四輪改了四次，又回到同一組權重','Four updates return to the same weights','回到原處，並不表示這一輪沒有更新。','Returning to a state is different from making no updates.','§1、數值重現','第四輪起點(-1,0,0)，四點依序更新到(-1,0,-1)、(-1,1,0)、(0,1,1)、(-1,0,0)。四節點以原生connector連成回圈，每節點標訪問點與新權重；圖註w_aug=(w1,w2,w0)。','dial','work','繞回來，\n不等於停下來。','A cycle is\nnot a stop.'),
('proof','兩類連線在中心相交，一條線無法隔開','The class segments cross at the same midpoint','非可分的結論來自證明，不只來自十輪實跑。','Nonseparability is proved, not inferred from ten passes.','選讀、引第1章KP08','單位方形同類兩線段都通過(0.5,0.5)，半透明凸包線段加中心星形；完整兩個中點式((0,0)+(1,1))/2=((0,1)+(1,0))/2=(0.5,0.5)。幾何直觀解釋線性半平面保留中點，第1章不等式證明引用備註。','square','think','兩類中點，\n竟然重合！','Both classes\nshare a midpoint!'),
('grid','格點搜尋最多對三筆，仍不是窮盡所有實數','The grid gets at most three right; it is not every real weight','343 組整數權重是驗算，不能代替證明。','The 343 integer triples check the claim; they do not prove it.','§1格點搜尋','7乘7格點面板，標w1,w2,w0屬{-3,...,3}，總數7^3=343；四個樣本圖示中最多三個勾一個紅圈；max correct=3/4。不得把有限搜尋冒充所有實數權重。','warning','think','搜過一批，\n不等於全部。','A finite search\nis not all weights.'),
('boundaries','同一批訓練點，可以被兩條線全部分對','Two boundaries can fit the same training points perfectly','全對只決定能停止，沒有選出唯一邊界。','Perfect training fit does not select a unique boundary.','§2','依source rng3重現兩類各12點；訓練seed100與102，改起點與順序；等尺度兩圖以未四捨五入權重繪A/B邊界，各標24/24，保持完全相同資料點與尺度。','rate','think','兩張考卷，\n都拿滿分。','Both score full\nmarks in training.'),
('newpoint','同一個新點，兩條邊界卻給相反答案','The same new point receives opposite predictions','兩條訓練全對的線，對新點仍會分歧。','Two perfect training fits can disagree on a new input.','§2','同批24點與A/B兩線等尺度；新點(2.5,2.4)用紫星，A分數2.464386...輸出1，B分數-0.414848...輸出0。此頁只放圖解與輸出；數值分數與完整算式在下頁。','warning','confused','滿分之後，\n答案還不同？','Perfect fits,\ndifferent answers?'),
('scores','用完整權重計算，新點確實落在不同側','Full-precision scores place the point on different sides','中間值不取整，最後才標近似值。','Keep full precision and round only the displayed result.','§2計算','輸入x=(2.5,2.4)，兩行完整sA=wA1(2.5)+wA2(2.4)+bA≈2.464>0,oA=1；sB同式≈-0.415<0,oB=0。用三段有號長條對應乘積與偏置，權重顯示8位小數且用approx，不把截短係數當精確。','dial','think','同一個點，\n兩次代入。','Same input,\ntwo score sums.'),
('unseen','不知道新點正解，就不能宣布誰泛化較好','Without the true label, disagreement does not name a winner','訓練全對，不是對未見資料的保證。','A perfect training fit is no guarantee on unseen data.','§2、補充釐清','兩張訓練全對考卷到同一紫星問號的原生connector；候選選擇用未見標記資料比較，使用獨立驗證選擇，最終評估另用測試；此頁只用訓練／未見資料並列概念圖，不捏造新點標籤或誤差率。','cards','think','先看新題正解，\n再談誰較好。','Check unseen\nlabels first.'),
('remedies','表示力與邊界選擇，要用不同方法處理','Expressive power and boundary selection need different remedies','多層加非線性處理 XOR；連續損失仍要驗證泛化。','Use layers and nonlinearity for XOR; evaluate generalization separately.','§3','上下分流：無直線可分→多層+非線性→KP05；可分但邊界不唯一→連續損失／LMS→KP08。所有因果流用native connector，明示ADALINE不保證更佳泛化，連續損失本身仍是線性模型不能解XOR。','gate','work','問題不同，\n工具也不同。','Different limits,\ndifferent tools.'),
('history','三個歷史節點，帶出兩條發展方向','Three historical milestones motivate two directions','XOR 的限制可驗算，寒冬成因不能歸給一本書。','XOR is a mathematical limit; a research winter has multiple causes.','§3歷史','原生時間軸1958 Rosenblatt Perceptron、1960 Widrow&Hoff ADALINE/LMS、1969 Minsky&Papert單層限制，以connector銜接但標年代順序而非因果；後續多層+非線性。不聲稱第一個神經元或單一書造成寒冬。','warning','think','看見限制，\n再找新方向。','Find the limit,\nthen new directions.'),
('next','把神經元接起來，能越過一條線的限制嗎','Can connected neurons overcome the one-line limit?','下一個 KP：多層感知器與非線性。','Next: multilayer perceptrons and nonlinearity.','留給KP05問題','左邊四個XOR資料點經原生connector進入兩個隱藏神經元、再到輸出問號；標「概念示意」與「非線性？」提示，不提供KP05具體權重或完整解法。','cards','think','前一層輸出，\n交給下一層？','One layer feeds\nthe next?'),
('practice1','延伸：把 XNOR 的四條條件排出來','Practice: write the four XNOR conditions','提示：將同類條件相加，再比較是否矛盾。','Hint: add same-class conditions and compare.','練習1','XNOR四點圖00、11為1三角，01、10為0圓；右側條件空白問號，不抄講義解題示範；保留提示，不放答案。','tags','think','換一組正解，\n試著推一次。','New labels,\nyour proof.'),
('practice2','延伸：為什麼 AND 可以，XOR 卻不行','Practice: why does AND work while XOR does not?','提示：畫出兩題的正類點，再試一條直線。','Hint: plot positive points and try a straight line.','練習2','AND與XOR並排單位方形與正確標籤，不畫答案分界線；各自一把直尺概念圖，不新增證明答案。','ruler','think','先找正類點，\n再拿尺試試。','Find positives,\nthen try a ruler.'),
('practice3','延伸：訓練全對時，你還想知道什麼','Practice: what else do you need after perfect training fit?','提示：想想怎麼取得及使用沒看過的資料。','Hint: think how to obtain and use unseen data.','練習3','兩個候選模型與各24/24訓練成績卡；未見題目為問號卷，原生connector連向評估，未見資料正解不提供。','cards','think','滿分以後，\n還要問什麼？','After full marks,\nwhat comes next?'),
('quiz1','自我檢核：收斂保證適用於哪種資料','Self-check: which data support the convergence guarantee?','提示：回想 XOR 的表示力限制。','Hint: recall the expressive limit shown by XOR.','自我檢核1釐清','原題限定適用措辭改為「收斂保證」，選項保留任意多類別、二元線性可分、任意迴歸、需非線性邊界；四個語意圖示不提供答案。','cards','think','保證停下來，\n需要什麼條件？','What guarantees\na stop?'),
('quiz2','自我檢核：哪個邏輯問題是著名反例','Self-check: which logic task is the famous counterexample?','提示：同類點分居正方形對角線。','Hint: each class occupies opposite corners.','自我檢核2','保留AND、OR、XOR、NOT四選項，各用相同圓角標籤與問號，不以顏色或角色暗示答案，不展開歷史更細主張。','cards','think','回想剛才，\n哪題畫不開？','Which task\nresisted a line?'),
('quiz3','自我檢核：可分資料仍可能泛化差，原因是什麼','Self-check: why can separable data generalize poorly?','提示：想想同一批資料上的兩條邊界。','Hint: recall two boundaries on the same data.','自我檢核3','保留選項A收斂慢、B起點影響邊界可能貼近某類、C連續損失、D需要非線性；選項長則每項獨立2行四區比較，角色不揭答案。','cards','think','不是只問\n訓練有沒有全對。','Look beyond\ntraining accuracy.'),
('quiz4','自我檢核：ADALINE 採用哪種學習目標','Self-check: which learning objective does ADALINE use?','提示：連續損失是 KP08 的主題。','Hint: continuous loss is the topic of KP08.','自我檢核4釐清','原題「改善泛化靠的是」改問採何種目標，避免保證；保留A非線性活化、B LMS連續損失、C更多層、D曲線邊界，不附答案。','cards','think','先認識目標，\n再檢驗表現。','Know the objective,\nthen evaluate.'),
]
def write():
 src='https://github.com/ndhuccc/idl-2026-fall/blob/75f6a2f8fcfdcd82f974c36a91885046a02307b4/lectures/lec02-kp04-perceptron-limits-and-generalization.md'
 out=['# KP04 逐頁圖解提示詞（建立前）\n'];e=[]
 for i,(kind,zh,en,tz,te,sec,brief,prop,mood,sz,se) in enumerate(PAGES,1):
  out.append(f'''## 第{i}頁／共{len(PAGES)}頁：{zh}
用途：大學生課堂教學；本頁唯一訊息：{tz}／{te}。估計講授時間：主線約1分鐘，延伸自主練習；文字預算：約60中文字或35英文詞，公式完整優先，證明細節放備註。
來源：{src}，{sec}。前後頁銜接：依頁序由表示力到學習循環，再到未見資料與兩條解法；最後以原題檢核。
模式：原生JianBauDau設計，不先生圖。規格：16:9，1280×720，白底深色字，藍圓類0、綠三角類1、紫星未見點、紅圈錯分。
版面：上方標題；中間左側為主圖與必要公式(x50–1000,y150–600)，右側約20%為小感知器(x1025–1255,y162–560)，底部一句結論(y640–696)。
逐區內容：{brief}
趣味主角：固定藍頭紫身橘天線小感知器；道具{prop}，表情{mood}；泡泡文字「{sz.replace(chr(10),'／')}」／「{se.replace(chr(10),' / ')}」，與泡泡同步進場。角色是教學比喻，不是新增實驗。
原生元件：圖表用精確程式座標、shape/text/polyline/polygon，流程與對應用connector；角色橢圓、圓角矩形與曲線；主角徑向光暈、細陰影，必要處半透明填色、星形或旋轉印章。每頁至少3種活潑效果。英文位置與中文相同。
數學排版：所有變數、下標、數值代入、算式、座標均用LaTeX。每個顯示結果的必要算式同頁可見；權重原始精度保留，顯示截短係數時標近似。
分拍：條件與主圖 → 證據／推導與角色提示 → 結論；需要聚焦的頁以同中心x、改zoom與y推近再回全景。暫時提示在下一拍退場，完整數據不退場。
限制條件：模擬為固定起點順序，不推廣每輪更新數；有限格點搜尋不是實數域證明；未見點真實標籤未知，不能評誰較好；LMS不保證泛化改善。練習與自我檢核不放解答。
避免：只有卡片、只有吉祥物裝飾、手畫流程箭頭、同色異義、科學圖失真、截切、空泡泡、答案暗示。
檢查：來源對應、數值assert、同頁公式、圖層、泡泡同步、每隱藏節點只揭一次；三輪渲染看圖修正。
''');e.append(dict(page=i,kind=kind,source=src,section=sec,claim=tz,visual=brief,extension=i>14))
 (ROOT/'圖解提示詞.md').write_text('\n'.join(out));(ROOT/'圖解證據表.json').write_text(json.dumps(e,ensure_ascii=False,indent=2))
if __name__=='__main__':write()
