from pathlib import Path
import json
R=Path(__file__).parent
PAGES=[
('cover','KP05 多層感知器與非線性','KP05 Multilayer networks and nonlinearity','接成多層之後，非線性才讓表示力增加。','Nonlinearity makes depth more expressive.','單元主張','用雙輸入、兩隱藏、一輸出的原生connector網路串起兩層；與可合併的純線性流水線對照，旋轉貼紙標「關鍵：非線性」。','gate','think','接力之間，\n還要轉個彎！','Depth needs\nnonlinearity!'),
('roles','輸入接資料，隱藏層轉換，輸出給結果','Input receives data; hidden layers transform it','第一個隱藏層直接看輸入；隱藏不是看不見資料。','The first hidden layer directly receives the input.','§1釐清','畫2-2-1網路，三種角色分色；第一hidden由raw input直連；不宣稱hidden看不到原始資料或完全無監督訊號。','cards','think','隱藏，\n不是蒙住眼睛。','Hidden does not\nmean blind.'),
('dense','全連接是層間全接，前饋是方向不回頭','Dense describes connections; feedforward describes direction','這個兩路、兩路、一路網路，共六條權重連線。','This two-two-one network has six weighted connections.','§1','2x2輸入到隱藏4條，加2x1到輸出2條；完整式2·2+2·1=6，所有連線native，沒有回圈；不畫偏置節點，明示bias另計。','steps','think','接法與方向，\n是兩件事。','Pattern and\nflow differ.'),
('neuron','一顆神經元先加權，再做活化','Each neuron sums first, then activates','加權和與活化輸出，是前後兩步。','The weighted sum and activation output are separate steps.','§1符號','前層z進入sum，再進sigma到zi；完整ai=sum_l w_il z_l+bi、zi=sigma(ai)，b顯式列出，不混偏置併入。','dial','work','先算總和，\n再決定輸出。','Sum first.\nActivate second.'),
('layer','矩陣一次算一層，方向與維度要一致','A matrix computes a layer; dimensions must agree','矩陣每列對應一個輸出；輸入輸出都用欄向量。','Here each matrix maps column inputs to column outputs.','§1矩陣約定釐清','用A_k=W_k^T對應原講義，zk=sigma(Ak zprev+bk)；A大小m×n，zprev n×1，b與znew m×1。以2輸入3輸出示意，A 3×2與向量尺寸可視化。','ruler','think','先把尺寸，\n對齊再相乘。','Match shapes,\nthen multiply.'),
('gates','OR 與 NAND 先判斷，再交給 AND','OR and NAND feed an AND gate','這組手工權重，讓兩層網路能表示 XOR。','These hand-set weights represent XOR.','§2補充手工權重','完整2-2-1 native網路，所有6權重1/−1正確；bias OR−1 NAND+1.5 AND−2。f(s)=1當s>=0否則0。s1=x1+x2−1、s2=−x1−x2+1.5、s3=h1+h2−2。標補充：手工權重非訓練。','dial','work','先問至少一個，\n再排除兩個都有。','At least one,\nbut not both.'),
('case00','輸入全零時，OR 擋住最後的 AND','For two zeros, OR blocks the final AND','三次加權與活化，輸出得到零。','Three sums and activations produce zero.','§2表格第1列','x=(0,0)，三步完整代入s1=0+0−1=−1,h1=f(−1)=0；s2=−0−0+1.5=1.5,h2=f(1.5)=1；s3=0+1−2=−1,o=f(−1)=0。原生流程加燈號，不只結果。','gate','work','少了一路，\n最後就不亮。','One gate is off;\nAND stays off.'),
('case01','只有第二路為一時，兩顆隱藏神經元都亮','With only the second input on, both hidden gates fire','零分也輸出一，這個約定很重要。','A zero score outputs one under this convention.','§2表格第2列','x=(0,1)，s1=0+1−1=0,h1=f(0)=1；s2=−0−1+1.5=.5,h2=f(.5)=1；s3=1+1−2=0,o=f(0)=1。零分標星形或光暈，提醒>=0。','gate','happy','在線上，\n也算亮！','At zero,\nthe gate fires!'),
('case10','只開第一路，仍然走到相同的 XOR 輸出','With only the first input on, XOR still outputs one','交換兩個輸入，這組對稱權重給同一結果。','Swapping inputs preserves this symmetric result.','§2表格第3列','x=(1,0)，s1=1+0−1=0,h1=1；s2=−1−0+1.5=.5,h2=1；s3=1+1−2=0,o=1。仍保留每次f代入，來源主線算例，延伸練習要求自行重算不再附解。','gate','happy','換一路開，\n結果一樣。','Swap the input;\nsame result.'),
('case11','兩路全開時，NAND 擋住最後輸出','With both inputs on, NAND blocks the output','OR 亮還不夠，NAND 必須也亮。','OR alone is not enough; NAND must fire too.','§2表格第4列','x=(1,1)，s1=1+1−1=1,h1=f(1)=1；s2=−1−1+1.5=−.5,h2=f(−.5)=0；s3=1+0−2=−1,o=f(−1)=0。','gate','work','兩個都有，\n這次要關掉。','Both inputs on?\nXOR turns off.'),
('features','隱藏層改座標，把 XOR 變成可切開的點','The hidden layer turns XOR into separable features','兩個正類輸入，合到同一個隱藏座標。','The two positive inputs share one hidden coordinate.','§2座標轉換','左XOR四點，右特徵座標h1,h2：00→01、01/10→11（標兩筆疊合）、11→10。右AND邊界h1+h2=2，線上輸出1，positive只在11；不用虛構第四個特徵點。所有對應關係native connector或各自標記。','tags','happy','先換座標，\n再一刀切開。','Transform first,\nthen separate.'),
('constant','拿掉階梯，同一組權重卻互相抵消','Remove the steps and these weights cancel','這個特例是常數；不是所有線性網路都為常數。','This example is constant; all-linear networks need not be.','§3特例','完整y=(x1+x2−1)+(−x1−x2+1.5)−2=−1.5，兩路反向等量線段相消；A1=[[1,1],[-1,-1]],A2=[1,1], A2A1=[0,0]。偏置單獨合成−1+1.5−2=−1.5。','warning','confused','這組剛好抵消，\n不是通則！','These cancel;\na special case.'),
('affine','兩層仿射映射，仍可合成一層','Two affine layers merge into one affine map','偏置會一起合併，不能只乘權重矩陣。','Merge the biases as well as the matrices.','§3一般含偏置','完整h=A1x+b1；y=A2h+b2；y=(A2A1)x+(A2b1+b2)，兩層原生pipeline轉成一層，標輸入輸出維度，不宣稱一般常數。','rate','think','權重相乘，\n偏置也要帶著走。','Multiply weights;\ncarry the biases.'),
('depth','全線性加再多層，仍只是一個線性映射','More all-linear layers still form one linear map','沒有偏置時，矩陣連乘就是合併規則。','Without biases, matrix multiplication merges the layers.','§3一般多層','A3 A2 A1 x=Vx，V=A3A2A1；三個線性節點原生connector串接，第三拍換單節點等價，先三層退場部分提示但核心公式保持。明示本頁不含bias，含bias看前頁。','steps','think','一直照比例，\n接力也不會彎。','Linear relays\nstay linear.'),
('experiment','同一組權重，只換活化就改變等值線','The same weights give different contours after changing activation','固定種子與權重，差別只在兩個隱藏層的活化。','Weights are fixed; only hidden activations change.','§3圖程式實驗','rng2，維度2→4→3→1，無bias，output線性。依原程式300×300網格及1e−9等值門檻，畫identity與ReLU原程式保留>40點的等值線；折線超200點時分段。明示數值等值線、不保證四區；不把整片零輸出誤當普通零線。','ruler','think','同一套權重，\n換個活化看看。','Same weights,\ndifferent activations.'),
('learning','能表示 XOR，與自動學出權重，是兩件事','Representing XOR differs from learning its weights','非線性提供可能性，不保證一定學得會。','Nonlinearity enables representation; it does not guarantee learning.','迷思、選讀','兩張工作台：手工旋鈕→正確XOR；空白權重→訓練問號，沒有把forwardnetwork冒充learningalgorithm。教材補充標記保留，第三章backprop指向備註。','cards','think','搭得出來，\n還要學得出來。','Represent it,\nthen learn it.'),
('next','階梯會做判斷，卻不適合直接靠梯度學','A step makes decisions but offers no useful gradient','下一個 KP：選擇可用來學習的非線性活化。','Next: nonlinear activations suitable for learning.','留給KP06問題','精確階梯s0上下兩段，s≠0導數0，s=0不可微；右一個activation問號，原生connector引向KP06。不要說所有ReLU處處可微。','warning','think','會判斷，\n還要能指路。','Decision rules\nneed gradients.'),
('practice1','延伸：自己重算第一路為一的完整傳遞','Practice: recompute the full forward pass for one-zero','提示：先算兩個隱藏輸出，再算最後輸出。','Hint: compute both hidden outputs before the final output.','練習1','保留x=(1,0)，s1,h1,s2,h2,s3,o問號與三組公式，不在本延伸頁或備註重複答案。','dial','think','這一輪，\n換你操作。','Your turn\nto compute.'),
('practice2','延伸：兩個矩陣可以合成哪一個矩陣','Practice: which single matrix replaces the pair?','提示：先做矩陣乘法，別把順序反過來。','Hint: multiply matrices in the correct order.','練習2','原題A1=[[1,0],[1,1]]、A2=(1,2)，y=A2A1x；原生3節點flow，合併矩陣問號。只給提示，不提供乘積。','ruler','think','先看方向，\n再把矩陣相乘。','Check the order,\nthen multiply.'),
('practice3','延伸：十層恆等活化，為何不增加表示力','Practice: why do ten identity layers add no expressive power?','提示：十個矩陣連乘的結果仍是什麼？','Hint: what is the product of ten matrices?','練習3','用一排折疊式矩陣積與合併問號，保留十層但用省略號表示中間，不宣稱十個實際畫點；明示概念示意。','rate','think','多接幾站，\n一定更有力量嗎？','Do more stages\nalways add power?'),
('quiz1','自我檢核：前饋指的是哪一種資訊方向','Self-check: what direction does feedforward describe?','提示：回想剛才連線的方向。','Hint: recall the direction of the connections.','自我檢核1','保留四選項：回饋前層、只向前不回頭、只有一層、隨意連線。各選項平等圖示；不標答案。','cards','think','看看線，\n有沒有回頭？','Do connections\nturn back?'),
('quiz2','自我檢核：讀層數時，應先確認哪個約定','Self-check: which convention must be checked when counting layers?','提示：輸入層的角色與是否計入總數，要分開。','Hint: separate the input role from layer-count conventions.','自我檢核2原題釐清','原題四選項沒有合理正解且把input誤稱hidden；明示修訂題意。選項A是否計入輸入層、B輸入層就是隱藏層、C所有文獻數法相同、D輸入層一定最後。保留檢核目的不附答案。','cards','think','名稱相同，\n數法可能不同。','Same word,\ndifferent counts.'),
('quiz3','自我檢核：無偏置且全線性時，網路等於什麼','Self-check: what is a bias-free all-linear network equivalent to?','提示：將每一層的矩陣依序相乘。','Hint: multiply the layer matrices in order.','自我檢核3條件釐清','保留四選項：非線性模型、一次線性映射、指數複雜、無法訓練；題幹加無偏置，避免affine與linear混稱。','cards','think','先把條件，\n讀完整。','Read the\nconditions first.'),
('quiz4','自我檢核：XOR 的表示力為何需要非線性','Self-check: why is nonlinearity needed to represent XOR?','提示：如果把活化全換成恆等，會怎樣？','Hint: what if all activations become identity functions?','自我檢核4','保留更多hidden、nonlinear、較大學習率、較小批次四選項，圖示同等權重不洩答案；不把必要條件誤當充分或訓練保證。','cards','think','有了可能，\n不等於保證。','Possibility is\nnot a guarantee.'),
]
def write():
 src='https://github.com/ndhuccc/idl-2026-fall/blob/5a6e17c42083153c62d4dcce1673d86eba8d9469/lectures/lec02-kp05-mlp-and-nonlinearity.md';ps=[];ev=[]
 for i,(kind,zh,en,tz,te,sec,brief,prop,mood,sz,se) in enumerate(PAGES,1):
  ps.append(f'''## 第{i}頁／共{len(PAGES)}頁：{zh}
用途：大學生教學，唯一訊息：{tz}／{te}。估時主線每頁約1分鐘，延伸自主練習。前後銜接：結構→XOR手算→座標變換→全線性合併→實驗→梯度問題。
來源：{src}，{sec}。
規格：1280×720，白底深色字，中文英文各一份，同位置同鏡頭；主圖x50–1000,y150–610，角色x1025–1255,y162–560，結論y640–696。字級至少16，全文約60中文字或35英文詞，完整公式優先，細節進備註。
逐區內容：{brief}
角色：延續藍頭紫身橘天線小感知器，道具{prop}、表情{mood}；對話「{sz.replace(chr(10),'／')}」／「{se.replace(chr(10),' / ')}」。角色比喻非實驗；泡泡與文字同拍。
原生元件：所有流程與對應native connector；科學圖為驗證座標的polyline/polygon；神經元橢圓，判斷菱形，零分關鍵星形；角色曲線、徑向光暈、陰影；標題多色漸層。每頁至少三類活潑效果，全份有旋轉貼紙，必要處退場與聚焦。
數學排版：公式、數值代入、維度、座標、權重、偏置一律LaTeX。矩陣統一Ak=Wk^T欄輸入約定，source手工A2A1不與W^T混用；每個結果保留同頁算式。
分拍：條件／主圖→推導／角色→結論；暫時提示可在第二拍退場；一般合併頁第二拍推近，第三拍回全景。
限制與避免：手工權重明示補充，非學習實驗；ReLU比較原種子2與權重固定、無偏置、output線性，數值門檻1e−9，不保證區域數或泛化。不得宣稱hidden看不到raw input、所有線性網路恆常、input層就是hidden、非線性必學會。練習檢核不附答案。
檢查：數值assert、隱藏節點揭示一次、connector引用、字級、同頁公式、泡泡同步、三輪看圖批評修正。
''');ev.append(dict(page=i,kind=kind,source=src,section=sec,claim=tz,visual=brief,extension=i>17))
 (R/'圖解提示詞.md').write_text('\n'.join(ps));(R/'圖解證據表.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2))
if __name__=='__main__':write()
