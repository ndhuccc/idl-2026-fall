"""兩種風格共用的內容規格(第 2 週 KP07 損失函數)。
標題與結論來自 guided/建頁來源/specs.py;本檔補:講義錨點、承接小標、自學導讀稿(中英)。
頁序與頁數由內容決定(此例 29 頁),不是為了湊時數。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "guided", "建頁來源"))
import specs

S = specs.PAGES   # (kind, 標題zh, 標題en, 結論zh, 結論en, 來源, 視覺, 道具, 表情, 泡泡zh, 泡泡en)
R1, R2, R3 = "1. 回歸：MSE、MAE、Huber", "2. 分類：交叉熵與 KL 散度", "3. Hinge：分對了但離邊界太近，也要罰"
CORR = "來源更正與取捨"

ANCHORS = [
 ["學習目標"], ["符號表", R1], [R1], [R1], [R1], [R1], [R1], [R1], [R1, "重點:圖"],
 [R2], [R2], [R2, "自我檢核#4"], [R2], [R2], [R2, CORR],
 [R3, "重點:圖"], [R3], [R3, CORR], ["想深入"],
 ["本單元重點整理", "重點:一句話重點", "重點:白話說明", "重點:正式定義", "重點:例子"], ["常見迷思#1", "常見迷思#2"],
 ["重點:留給下一個知識點的問題"],
 ["小練習#1", "解題示範"], ["小練習#2"], ["小練習#3"],
 ["自我檢核#1"], ["自我檢核#2"], ["自我檢核#3"], ["自我檢核#4"]]
NOTE = {12: "二元交叉熵的完整式是補充說明(講義只提 sigmoid 用於二元輸出,並在自我檢核 4 要求條件);算式與 §2 的交叉熵一致。",
        18: "「分數餘裕不是幾何距離」是補充說明,用來對應來源更正「$yf_o\\ge m$」,不改變講義結論。"}

# (承接 zh, 承接 en)
BR = [
 ("承接 KP06:輸出層已算出預測,這個單元問:它錯多少?", "After KP06 the output layer gives a prediction; this unit asks how wrong it is."),
 ("上一頁說要量「錯多少」,第一步是算預測與目標的差。", "To measure how wrong, the first step is the gap between prediction and target."),
 ("有了殘差,第一把尺:先平方再加總。", "With residuals in hand, the first ruler: square, then add."),
 ("第二把尺:不平方,直接取絕對值再加總。", "The second ruler: skip the square, just take absolute values."),
 ("平方太凶、絕對值太寬容,Huber 把兩者接起來。", "Squared is harsh and absolute is lenient; Huber joins them."),
 ("套用 Huber 的兩段公式到同一組殘差。", "Apply Huber's two pieces to the same residuals."),
 ("三把尺的總和各是多少?先釐清加總與平均。", "Each ruler gives a sum; first, sum versus mean."),
 ("光看總和不夠,要看離群值占了幾成。", "Totals are not enough; check what share the outlier takes."),
 ("把三把尺畫在同一個座標,看對大誤差的反應。", "Draw the three rulers on one axis to see how they react to big errors."),
 ("回歸量距離;分類改量「給正解多少機率」。", "Regression measures distance; classification measures the probability given to the right answer."),
 ("用 KP06 的 Softmax 輸出,算正解不同時的交叉熵。", "Use KP06's softmax output to compute cross-entropy for different right answers."),
 ("兩類問題的交叉熵怎麼寫?把兩種結果都寫進來。", "How is two-class cross-entropy written? Include both outcomes."),
 ("目標本身是一份機率分布時,改用 KL 散度。", "When the target is itself a distribution, use KL divergence."),
 ("用具體數字逐項算 KL。", "Compute KL term by term with numbers."),
 ("交叉熵與 KL 有什麼關係?", "How are cross-entropy and KL related?"),
 ("第三種分類尺:Hinge,連分對但太近也要罰。", "A third classification ruler: hinge, which punishes even right-but-too-close."),
 ("用三個分數代入 Hinge。", "Plug three scores into hinge."),
 ("Hinge 的「餘裕」到底是不是距離?", "Is hinge's margin really a distance?"),
 ("(選讀)回頭看 Huber:兩段在接點接得平順嗎?", "(Optional) Back to Huber: do the two pieces join smoothly?"),
 ("六種損失都看過了,整理成一張選擇表。", "All six losses seen; gather them into one choosing table."),
 ("最後澄清兩個常見迷思。", "Finally, two common misconceptions."),
 ("損失有了,下一步:怎麼讓它變小?", "We have a loss; next, how to make it smaller?"),
 ("自己試試:換一組殘差,用三把尺。", "Try it: new residuals, three rulers."),
 ("自己試試:分類的交叉熵。", "Try it: classification cross-entropy."),
 ("自己試試:Hinge 遇到負標籤。", "Try it: hinge with a negative label."),
 ("自我檢核:想想看,再對照提示。", "Self-check: think first, then use the hint."),
 ("自我檢核:Huber 的 $\\delta$。", "Self-check: Huber's $\\delta$."),
 ("自我檢核:軟標籤用哪個損失。", "Self-check: which loss for soft labels."),
 ("自我檢核:交叉熵對輸出層的要求。", "Self-check: what cross-entropy needs from the output layer."),
]
