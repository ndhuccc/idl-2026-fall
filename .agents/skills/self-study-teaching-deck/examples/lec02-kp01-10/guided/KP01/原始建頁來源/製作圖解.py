from pathlib import Path
import json

ROOT=Path(__file__).parent
SOURCE='https://github.com/ndhuccc/idl-2026-fall/blob/54845cd20fd4523b8ccbb72a0e14dada28c567d8/lectures/lec02-kp01-biological-neuron-and-mp-unit.md'
# Each entry is a self-contained figure brief, written before construction.
PAGES=[
('cover','KP01 神經元與門檻','KP01 Neurons and thresholds','把訊號加起來，達到門檻才輸出一。','Add the signals; output one when the threshold is reached.','全單元','左至右三個大圖形：三個輸入圓點、整合圓形、輸出燈；箭頭表示訊號傳遞。下方一句核心結論。',r'$z=\sum_j w_jx_j$；$z\ge\theta$；$o=1$'),
('anatomy','四個部件接力，把訊號傳出去','Four parts relay a signal','接收、整合、傳出、接力。','Receive, integrate, transmit, relay.','§1','樹突分支連向中央細胞體，細胞體含核；右側長軸突分叉至突觸，接下一顆神經元。四組引線標籤，箭頭標訊號方向。示意圖不按生理比例。','none'),
('switch','達到門檻才放電，輸出不是音量','Firing is a switch, not a volume knob','達標與更強，都只回答一。','At or above the threshold, the answer stays one.','§1、重點整理','三個等比例長條分別長1、2、3，門檻垂直虛線固定在2；旁邊三個燈號，第一灰、其餘綠。純計算機制示意，不宣稱真實脈衝振幅。',r'$\theta=2$；$z=1<2\Rightarrow o=0$；$z=2\ge2\Rightarrow o=1$；$z=3\ge2\Rightarrow o=1$'),
('weighted','每路先乘權重，再把貢獻加起來','Weight each input, then add its contribution','權重決定每路訊號的影響。','A weight sets how much an input contributes.','§2、符號表','三路輸入圓點經帶權重的箭頭連到加總圓形，再接總和。圖下方完整加權和公式；圖中輸入與權重在各路就地標示，總和定義緊貼公式。',r'$z=\sum_{j=1}^{d}w_jx_j$；$x_j$ 輸入，$w_j$ 權重，$d$ 輸入路數，$z$ 加權總和'),
('bias','減掉門檻，就能統一和零比較','Subtract the threshold to compare with zero','偏置就是負的門檻。','The bias is the negative threshold.','§2','上半圖用同一數線的總和與門檻距離，下一行把整段向左平移門檻長度，門檻變零；完整顯示三步等價關係，正負方向與箭頭一致。',r'$z\ge\theta$；$z-\theta\ge0$；$s=z+w_0$；$w_0=-\theta$；$s\ge0$'),
('step','階梯函數把總和變成零或一','The step function turns a sum into zero or one','剛好等於零，也輸出一。','Exactly zero also outputs one.','§2','精確座標：橫軸s、縱軸o；負半軸在0的水平線，非負半軸在1；原點下方空心、上方實心。旁邊一盞開關燈顯示二元而非連續輸出。',r'$o=f(s)$；$f(s)=0\;(s<0)$；$f(s)=1\;(s\ge0)$；$s$ 活化前總和，$o$ 輸出'),
('sum','三路輸入一、零、一，總和就是二','Inputs one, zero, one add up to two','沒有訊號的那一路，貢獻零。','The inactive input contributes zero.','§3','三個輸入圓點1、0、1，箭頭上w=1，右側用兩個等長方塊拼出總和2，中間0用虛線空框；完整算式顯示每項乘積與相加。',r'$\mathbf{x}=(1,0,1)$；$w_1=w_2=w_3=1$；$z=1\times1+1\times0+1\times1=2$'),
('threshold','總和不變，只移門檻就改變輸出','Keep the sum fixed and move the threshold','門檻一、二、三，輸出一、一、零。','Thresholds one, two, three give outputs one, one, zero.','§3','三列相同刻度的長條z=2，門檻分別1、2、3；每列顯示偏置與s運算以及燈號。門檻2正好對齊條末，特別圈出等號；三列最多三組數值。',r'$z=1+0+1=2$；$s=z-\theta$；$w_0=-\theta$；$2-1=1\Rightarrow o=1$；$2-2=0\Rightarrow o=1$；$2-3=-1\Rightarrow o=0$'),
('and','兩路都為一，才通過門檻二','AND needs both inputs to be one','AND：兩個條件都成立。','AND: both conditions must hold.','§4','正方形四點分別(0,0)、(0,1)、(1,0)、(1,1)，實心點輸出1、空心0；直線x1+x2=2只經右上角，達標區半透明綠；每點旁標s運算與輸出，不使用擁擠四列表格。',r'$w_1=w_2=1$；$\theta=2$；$s=x_1+x_2-2$；$o=1\;(s\ge0)$'),
('or','任一路為一，就通過門檻一','OR needs at least one input to be one','OR：至少一個條件成立。','OR: at least one condition must hold.','§4','與AND完全相同尺度的四點平面，直線x1+x2=1通過(0,1)、(1,0)，線上及右上方達標；四點分別標完整s代入，填空心圖例同頁可見。',r'$w_1=w_2=1$；$\theta=1$；$s=x_1+x_2-1$；$o=1\;(s\ge0)$'),
('compare','AND 變 OR，只需要把門檻降低','Lowering the threshold turns AND into OR','輸入與權重沒變，邊界移動了。','The inputs and weights stay fixed; the boundary moves.','§4','左右兩個相同尺度四點圖，左AND門檻2、右OR門檻1，中間箭頭標降低門檻；線上用實線、輸出用實空心圓，淡色表示達標區。兩側均保留判斷公式。',r'$s=x_1+x_2-2$；$s=x_1+x_2-1$；$w_1=w_2=1$；$o=1\Leftrightarrow s\ge0$'),
('not','要把輸出反過來，權重就要是負的','A negative weight reverses the response','補充：正權重不能做 NOT。','Added: a positive weight cannot implement NOT.','§4〔補充〕','左側輸入0到亮燈1、輸入1到暗燈0兩路；中間負權重的下降箭頭；右側精確直線s=-x+0.5，x=0與1標代入式，零點在0.5。補充標籤清楚可見。',r'$w=-1$；$w_0=0.5$；$s=-x+0.5$；$x=0:\ s=-1\times0+0.5=0.5,\ o=1$；$x=1:\ s=-1\times1+0.5=-0.5,\ o=0$'),
('learn','能算邏輯，還不等於會自己學','Computing logic is not the same as learning','下一步：讓資料幫我們改權重。','Next: use data to change the weights.','§2、重點整理、下一知識點問題','左側人手轉旋鈕設定權重與門檻，中央固定神經元計算，右側資料點與正確標籤通向帶問號的權重旋鈕。以虛線表示下一單元的待解問題，不畫已存在的學習保證。',r'$o=f(\sum_j w_jx_j+w_0)$；$w_0=-\theta$'),
('practice1','延伸：先加總，再檢查是否達標','Practice: add first, then test the threshold','提示：先算總和，再減門檻。','Hint: find the sum, then subtract the threshold.','小練習1','三路1、1、0經權重皆1進入加總器，再經門檻2的閘門；輸出燈內只放問號，不顯示答案。下方保留計算空格與符號。',r'$\mathbf{x}=(1,1,0)$；$w_1=w_2=w_3=1$；$\theta=2$；$z=\sum_jw_jx_j$；$s=z-\theta$；$o=f(s)=?$'),
('practice2','延伸：至少兩路為一，門檻該設多少','Practice: choose a threshold for two active inputs','提示：比較兩路亮起和只有一路亮起。','Hint: compare two active inputs with one active input.','小練習2','兩列各三個圓點，上列1、1、0下列1、0、0，權重全1；中央可移門檻標問號，上列目標燈亮、下列暗。這是題目指定目標，不是暗示某個未驗算門檻。',r'$w_1=w_2=w_3=1$；$\theta=?$；$\mathbf{x}=(1,1,0)$；$\mathbf{x}=(1,0,0)$'),
('practice3','延伸：正權重為什麼做不出 NOT','Practice: why can positive weights not make NOT?','提示：輸入變大，總和往哪個方向走？','Hint: when the input rises, which way does the sum move?','小練習3','左側NOT目標0到1、1到0；右側待完成的正權重數線，輸入由0向1移動，s方向保留問號；不畫結論方向，不放答案。',r'$s=wx+w_0$；$w>0$；$x:0\to1$'),
('quiz1','自我檢核：誰負責整合收到的訊號','Self-check: which part integrates the signals?','提示：想想訊號在哪裡累積。','Hint: where do the signals accumulate?','自我檢核1','同一顆神經元四部件全部用相同飽和度，不凸顯正解；四選項附對應部件小圖示：A樹突、B突觸、C細胞體、D軸突；不放答案。','none'),
('quiz2','自我檢核：人工神經元模仿什麼','Self-check: what does an artificial neuron copy?','提示：辨認計算機制，不只看外形。','Hint: look for the computation, not just the shape.','自我檢核2','四個候選圖等大小：A細胞代謝用循環箭頭、B累積訊號過閾值開關放電用簡化流程、C外形用神經元輪廓、D遺傳物質複製用雙鏈幾何；各圖有原文選項，不凸顯答案。','none'),
('quiz3','自我檢核：誰在何年提出這個模型','Self-check: who proposed the model, and when?','提示：回想固定權重的早期模型。','Hint: recall the early model with fixed weights.','自我檢核3','四個歷史候選等大的時間牌與人像圖示，A Rosenblatt 1958、B McCulloch與Pitts 1943、C Widrow與Hoff 1960、D Minsky與Papert 1969；不加正誤符號。','1958、1943、1960、1969為原文選項年份，非運算數字'),
('quiz4','自我檢核：這裡用的是哪一種活化函數','Self-check: which activation function is used here?','提示：輸出只有零或一。','Hint: the output has only two possible values.','自我檢核4','四個相同大小、正確形狀的函數小圖：A Sigmoid、B ReLU、C二元階梯、D tanh；各軸符號用LaTeX，四個候選不顯示對錯。曲線由公式程式生成，非任意繪製。',r'$\operatorname{sigmoid}(s)=1/(1+e^{-s})$；$\operatorname{ReLU}(s)=\max(0,s)$；$f(s)=\mathbf{1}[s\ge0]$；$\tanh(s)$')
]

def main():
    data=[]; prompts=[]
    for i,(kind,zh,en,tz,te,sec,brief,math) in enumerate(PAGES,1):
        entry=dict(page=i,kind=kind,title_zh=zh,title_en=en,takeaway_zh=tz,takeaway_en=te,source=SOURCE,section=sec,brief=brief,math=math,extension=i>=14)
        data.append(entry)
        prompts.append(f'''## 第{i}頁／共{len(PAGES)}頁
本頁唯一訊息：{tz}
估計講授：1–2分鐘（延伸頁依需要選用）。中文約60字內，英文約35詞內；超量拆頁。
用途：大學部深度學習導論，依第二章KP01製作可編輯的教學圖解簡報。
模式：全新原生幾何設計，不先生成圖片；對應英文頁標題「{en}」。
來源：{SOURCE}，{sec}。圖解所需事實與版面已完整列於下方。
核心結論：{tz} / {te}
規格：1280×720，繁體中文與英文分檔，白色或極淺灰底、深藍文字；輸入藍、整合紫、達標綠、門檻橘。字級至少20px，註記至少16px。
版面：頂部80px標題；中央約460px最大圖形；底部640–696px一句結論。左右留白48px。閱讀順序左到右、上到下。
逐區內容：標題「{zh}」；{brief}；結論「{tz}」。定義與必要前提同頁可見。
圖例：箭頭表示訊號或計算流程；實心綠燈代表輸出一、空心灰燈代表零；門檻用橘色虛線，顏色與形狀雙重區分。題目頁不凸顯正確選項。
數學排版：{math}。數學物件全用LaTeX，中文不放進\\text；正確渲染上下標，不印出原始LaTeX指令。
限制條件：依來源的二元輸入與階梯輸出；達到門檻也放電。生物圖為簡化機制示意；固定權重模型不會自己學習。NOT例沿用來源的「補充」。練習與自我檢核不附答案。
底部重點：{tz}
避免：深色背景、長段落、只有文字卡、錯把等號當不達標、圖與算式數字不一致、文字被圖形遮住、未標示新結論。
檢查：逐項比對來源與數值腳本；圖佔最大區域；文字圖層高於底圖；中英版相同頁數、位置及揭示步驟。
原生元件：橢圓作輸入、燈號與細胞體；折線作樹突、軸線與精確曲線；三角箭頭表示方向；多邊形作達標區域、菱形作判斷；矩形只作長條與背景。文字獨立節點。兩個beat依序出現圖與結論，講者備註隨beat分段。
''')
    (ROOT/'圖解提示詞.md').write_text('\n'.join(prompts),encoding='utf-8')
    (ROOT/'圖解證據表.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(data)} self-contained figure prompts written before drawing')

if __name__=='__main__': main()
