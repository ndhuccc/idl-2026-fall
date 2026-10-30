from pathlib import Path
import sys,json,math,copy,datetime
ROOT=Path(__file__).parent
TOOLS=ROOT.parent/'idl-kp01-toolchain'/'.agents/skills/image-to-jbd-page/tools'
sys.path.insert(0,str(TOOLS))
from jbd_page import Page
from new_deck import make
from 製作圖解 import PAGES

INK='#17324d'; MUT='#50647a'; BLUE='#2678b8'; PURPLE='#7959ba'; GREEN='#168269'; ORANGE='#b76716'; GREY='#91a0ac'; RED='#b24b52'; LIGHT='#f4f8fb'
PAL=[BLUE,PURPLE,GREEN,ORANGE]
LANG='zh'; P=None

def tr(a,b):return a if LANG=='zh' else b
def text(x,y,w,h,s,size=25,c=INK,bold=False,align='left',name=None):
    n=P.text(x,y,w,h,s,size=size,color=c,bold=bold,align=align,z=30,line=1.22)
    if name:n['name']=name
    return n
def label(x,y,w,h,s,size=26,c=INK,bold=True):return text(x,y,w,h,s,size,c,bold,'center')
def eq(x,y,w,s,size=27,h=62,c=INK):return label(x,y,w,h,'$'+s+'$',size,c,False)
def arrow(x1,y1,x2,y2,c=BLUE,sw=5,dash=False):return P.line([(x1,y1),(x2,y2)],c,sw,arrow='end',head=14,dash=dash,z=7)
def lamp(cx,cy,on=None,r=46):
    P.ellipse(cx,cy,r,fill='#e5f5ed' if on else '#f3f5f7',stroke=GREEN if on else GREY,sw=4,z=4)
    P.rect(cx-r*.42,cy+r*.8,r*.84,16,fill=GREEN if on else GREY,radius=5,z=4)
    if on is True:
        for a in range(0,360,60):
            rad=math.radians(a);P.line([(cx+math.cos(rad)*(r+12),cy+math.sin(rad)*(r+12)),(cx+math.cos(rad)*(r+24),cy+math.sin(rad)*(r+24))],GREEN,3,z=3)
    label(cx-r,cy-27,2*r,60,'?' if on is None else ('$1$' if on else '$0$'),36,GREEN if on else MUT)
def dot(cx,cy,value,c=BLUE,r=33):
    P.ellipse(cx,cy,r,fill='#e9f3fb' if value else '#ffffff',stroke=c,sw=3,z=3)
    label(cx-r,cy-24,2*r,50,str(value),30,c)
def neuron(cx=555,cy=355,scale=1,annotations=True):
    def xy(x,y):return(cx+(x-555)*scale,cy+(y-355)*scale)
    def line(pts,c=BLUE,sw=6):return P.line([xy(*p) for p in pts],c,sw*scale,z=3,smooth=True)
    for sy in (240,355,470):
        line([(225,sy),(320,sy),(410,340 if sy<355 else 370)])
        for d in (-28,28):line([(245,sy),(205,sy+d),(165,sy+d*1.4)],sw=4)
    P.ellipse(*xy(495,355),85*scale,68*scale,fill='#f0e9fc',stroke=PURPLE,sw=4,z=4)
    P.ellipse(*xy(495,355),26*scale,fill='#d5c5ef',stroke=PURPLE,sw=2,z=5)
    line([(580,355),(700,355),(810,355)],ORANGE,9)
    for xx in (620,685,750):
        x,y=xy(xx,355);P.rect(x,y-16*scale,38*scale,32*scale,fill='#f8deb9',radius=10,z=4)
    for dy in (-65,0,65):
        line([(810,355),(855,355+dy),(915,355+dy)],ORANGE,5)
        P.ellipse(*xy(920,355+dy),10*scale,fill=ORANGE,z=4)
        line([(943,355+dy-21),(948,355+dy),(943,355+dy+21)],GREEN,5)
    if annotations:
        for x,y,w,s,c in [(94,145,250,tr('樹突：接收','Dendrites: receive'),BLUE),(400,145,260,tr('細胞體：整合','Soma: integrate'),PURPLE),(650,492,260,tr('軸突：傳出','Axon: transmit'),ORANGE),(940,160,270,tr('突觸：接力','Synapse: relay'),GREEN)]:
            text(x,y,w,65,s,26,c,True)
        P.line([(240,210),(260,245)],BLUE,2,z=4);P.line([(500,210),(500,280)],PURPLE,2,z=4)
        P.line([(750,490),(750,385)],ORANGE,2,z=4);P.line([(1040,220),(943,290)],GREEN,2,z=4)
        arrow(320,565,910,565,c=MUT,sw=3);text(450,570,390,40,tr('訊號傳遞方向','Direction of signal flow'),21,MUT)

def pipeline(example=False,unknown=False):
    vals=[1,0,1] if example else ['x_1','x_2','x_d']
    for i,y in enumerate((245,365,485)):
        if example:dot(215,y,vals[i])
        else:
            P.ellipse(215,y,36,fill='#e9f3fb',stroke=BLUE,sw=3,z=3);eq(175,y-30,80,vals[i],27)
        arrow(255,y,540,365,BLUE,4)
        eq(325,y-47,145,'w_'+str(i+1) if i<2 else 'w_d',23,c=BLUE)
    P.ellipse(600,365,62,fill='#f0e9fc',stroke=PURPLE,sw=4,z=4);eq(542,320,116,r'\sum',45,h=100,c=PURPLE)
    arrow(668,365,820,365,PURPLE)
    if unknown:lamp(925,365,None,60)
    else:
        P.ellipse(920,365,70,fill='#eef6fb',stroke=BLUE,sw=4,z=4);eq(845,320,150,'z',48,h=100,c=BLUE)

def axis_plot(x=155,y=225,w=590,h=310,xmin=-2,xmax=2,ymin=-.3,ymax=1.4):
    fx=lambda v:x+(v-xmin)/(xmax-xmin)*w
    fy=lambda v:y+h-(v-ymin)/(ymax-ymin)*h
    arrow(x,fy(0),x+w+20,fy(0),MUT,3);arrow(fx(0),y+h,fx(0),y-15,MUT,3)
    return fx,fy

def gate(kind,x=150,y=240,size=305,calc=True):
    theta=2 if kind=='AND' else 1
    lo=-.18;hi=1.22
    fx=lambda a:x+(a-lo)/(hi-lo)*size;fy=lambda a:y+size-(a-lo)/(hi-lo)*size
    # Exact clipped half-plane x1+x2 >= theta in the square viewport.
    polygon=[(lo,lo),(hi,lo),(hi,hi),(lo,hi)]
    clipped=[]
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        ina=sum(a)>=theta;inb=sum(b)>=theta
        if ina:clipped.append(a)
        if ina!=inb:
            tt=(theta-sum(a))/(sum(b)-sum(a));clipped.append((a[0]+tt*(b[0]-a[0]),a[1]+tt*(b[1]-a[1])))
    P.poly([(fx(a),fy(b)) for a,b in clipped],fill='#e6f5ec',z=1)
    arrow(fx(lo),fy(0),fx(hi),fy(0),MUT,2);arrow(fx(0),fy(lo),fx(0),fy(hi),MUT,2)
    eq(fx(hi)-15,fy(0)+18,65,'x_1',22);eq(fx(0)-105,fy(hi)-12,64,'x_2',22)
    a=max(lo,theta-hi);b=min(hi,theta-lo)
    P.line([(fx(a),fy(theta-a)),(fx(b),fy(theta-b))],ORANGE,4,z=4)
    for xx,yy in [(0,0),(0,1),(1,0),(1,1)]:
        on=xx+yy>=theta
        P.ellipse(fx(xx),fy(yy),12,fill=GREEN if on else '#ffffff',stroke=GREEN if on else MUT,sw=3,z=6)
        eq(fx(xx)-45,fy(yy)+(12 if yy==0 else -50),90,f'({xx},{yy})',19)
    if calc:
        for i,(xx,yy) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            ss=xx+yy-theta;on=ss>=0
            yy0=220+i*79
            P.ellipse(730,yy0+26,12,fill=GREEN if on else '#ffffff',stroke=GREEN if on else MUT,sw=3,z=5)
            eq(758,yy0,425,f'{xx}+{yy}-{theta}={ss}' + r'\quad\Rightarrow o='+str(int(on)),24)
        eq(x,y-65,size,'x_1+x_2='+str(theta),27,c=ORANGE)
    return fx,fy

def plotfun(x,y,w,h,kind):
    fx,fy=axis_plot(x,y,w,h,xmin=-3,xmax=3,ymin=-1.3,ymax=3.1 if kind=='relu' else 1.3)
    points=[]
    for i in range(121):
        xx=-3+i*.05
        yy=(1/(1+math.exp(-xx)) if kind=='sigmoid' else max(0,xx) if kind=='relu' else math.tanh(xx) if kind=='tanh' else (1 if xx>=0 else 0))
        if kind=='step' and i==60:
            P.line(points,BLUE,4,z=4);points=[]
        points.append((fx(xx),fy(yy)))
    P.line(points,BLUE,4,z=4)

TEACH={
'cover':('多路訊號進來後，先加在一起，再問有沒有達到門檻。這個單元要把這件事變成能親手計算的規則。','Several signals arrive. Add their contributions, then test whether the threshold is reached. We will turn this mechanism into a rule we can compute by hand.'),
'anatomy':('樹突接收訊號，細胞體在這張簡化圖中代表整合，軸突把放電訊號送出，突觸把訊號接力給下一顆神經元。人工模型取用的是整合與門檻的機制。','Dendrites receive signals. The soma represents integration in this simplified diagram. The axon transmits a pulse, and synapses relay it onward. The artificial model abstracts integration and thresholding.'),
'switch':('把門檻固定為二，總和一沒有達標；總和二剛好達標，總和三也達標。後兩者輸出都只是一，模型不把輸出當成音量。這是計算機制的示意。','Fix the threshold at two. A sum of one falls short; two reaches it exactly; three exceeds it. The last two both output one. This is a simplified computational mechanism, not a physiological amplitude model.'),
'weighted':('每一路輸入先乘上它的權重，然後把所有乘積加起來。字母 d 表示輸入共有幾路，z 是還沒有減掉門檻的加權總和。','Multiply each input by its weight, then add the products. The symbol d is the number of inputs. The weighted sum z does not yet include the threshold adjustment.'),
'bias':('原本的問題是總和有沒有達到門檻。兩邊同減門檻後，就改成與零比較；把負門檻叫作偏置，便能寫成總和加偏置。圖中兩個位置一起左移，距離不變。','Start by comparing the sum with the threshold. Subtract the threshold from both sides to compare with zero instead. Calling the negative threshold the bias gives sum plus bias. Both marked positions shift equally, so their distance is unchanged.'),
'step':('負半軸的輸出是零，非負半軸的輸出是一。零的位置下面是空心點，上面是實心點，表示等於零時取上面的一。','The negative half-axis outputs zero; the nonnegative half-axis outputs one. At zero, the lower point is open and the upper point is filled. Equality belongs to the upper branch.'),
'sum':('三路輸入是一、零、一，權重都為一。每路的乘積分別是一、零、一，所以有兩路提供一，總和是二。中間的虛線格代表零貢獻。','The three inputs are one, zero, one, with all weights one. Their contributions are one, zero, one. Two active inputs each contribute one, so the sum is two. The dashed slot marks zero contribution.'),
'threshold':('三列的藍條完全一樣，因為總和一直是二。門檻往右移時，剩下的量依序是一、零、負一。剛好剩零仍然算達標。','All three blue bars are identical because the sum stays two. Moving the threshold gives pre-activation values one, zero, and minus one. Exactly zero still meets the condition.'),
'and':('橫軸與縱軸是兩路輸入。四個角是全部可能的二元輸入。只有右上角的總和達到二，所以只有那一點填實；右側逐一列出代入計算。','The axes are the two inputs. The four corners are all possible binary input pairs. Only the upper-right corner reaches a sum of two, so only that point is filled. The substitutions on the right verify each case.'),
'or':('這次門檻降為一。兩個恰有一路為一的角點就在邊界上，因此也輸出一；兩路皆零的角點還沒有達標。','Lower the threshold to one. The two corners with exactly one active input lie on the boundary and therefore output one. The all-zero corner remains below the threshold.'),
'compare':('兩張圖的座標、四個輸入點與權重都相同。改變的只是門檻，讓直線往左下移；原本不達標的兩個角點因此變成達標。','Both diagrams use the same axes, four input points, and weights. Only the threshold changes, moving the line down and left. The two single-active-input corners now reach the threshold.'),
'not':('這是講義補充的一個單輸入例子。權重取負一，偏置取零點五。輸入由零增為一時，加總結果從正的零點五降成負的零點五，輸出因而反過來。','This added example from the lecture has one input. Set its weight to minus one and its bias to one half. Raising the input from zero to one lowers pre-activation from positive one half to negative one half, reversing the output.'),
'learn':('McCulloch 與 Pitts 在一九四三年提出早期神經元模型。這裡的權重由人預先設定，會算邏輯並不代表會自己修正參數。接下來的感知器要回答如何利用資料修改權重。','McCulloch and Pitts proposed the early model in 1943. Here a person sets the weights in advance. Computing logic does not mean the model can adjust its own parameters. The perceptron will introduce learning from data.'),
'practice1':('用三路輸入一、一、零，權重全為一，門檻為二。請先寫出加權總和，再減門檻，最後套用階梯函數。先讓學生自行算完。','Use inputs one, one, zero, all weights one, and threshold two. Write the weighted sum, subtract the threshold, and apply the step function. Give students time to work it out.'),
'practice2':('這一題給的是希望達成的輸出：至少兩路為一才放電。比較圖中的兩種輸入，找一個能把它們分開的門檻，並檢查其他二元輸入。','The desired behavior is to fire when at least two inputs are active. Compare the two input patterns, choose a threshold that separates them, and check the remaining binary patterns.'),
'practice3':('左邊是 NOT 所需要的輸出。右邊先不畫出方向，請學生判斷權重為正時，輸入增加會讓加總結果如何改變，再比較能否符合左邊的要求。','The left shows the desired NOT behavior. The direction on the right is intentionally left open. Ask how a positive weight changes the sum when the input rises, then compare with the required behavior.'),
'quiz1':('四個選項分別是樹突、突觸、細胞體、軸突。請回想這些部件各自的工作，再選出負責整合訊號的部件。','The options are dendrites, synapse, soma, and axon. Recall the role of each part and select the one that integrates incoming signals.'),
'quiz2':('四個候選分別是細胞代謝、累積與門檻放電、神經元外形、遺傳物質複製。請用計算機制來說明選擇的理由。','The alternatives are cell metabolism, accumulation and threshold firing, the neuron’s shape, and DNA copying. Explain the choice in terms of the computation.'),
'quiz3':('這題保留來源的四組人名與年份。請辨認提出 McCulloch–Pitts 模型的人物與時間，並區分後來不同模型或討論的歷史。','This question keeps the four name-and-year options from the source. Identify the origin of the McCulloch–Pitts model and distinguish it from later models and discussions.'),
'quiz4':('四張小圖分別是 Sigmoid、ReLU、二元階梯與 tanh。請觀察輸出可取的值與零點的規則，不只看名稱。','The four graphs show sigmoid, ReLU, binary step, and tanh. Compare their output values and behavior at zero, rather than choosing by name alone.')}

def draw(kind):
    if kind=='cover':
        for i,y in enumerate((245,365,485)):
            dot(240,y,[1,0,1][i],BLUE,40);arrow(285,y,530,365,BLUE,4)
        P.ellipse(600,365,76,fill='#f0e9fc',stroke=PURPLE,sw=4,z=4)
        label(520,320,160,90,tr('加總','ADD'),35,PURPLE)
        arrow(686,365,870,365,PURPLE);P.poly([(775,315),(812,365),(775,415),(738,365)],fill='#fff0d8',stroke=ORANGE,sw=3,z=4)
        lamp(990,365,True,65)
        text(700,465,290,70,tr('達到門檻','Reach the threshold'),25,ORANGE,True)
        text(58,142,1164,46,tr('深度學習導論｜第二章','Introduction to Deep Learning | Chapter 2'),22,MUT)
    elif kind=='anatomy':
        neuron();text(65,95,800,45,tr('生物神經元的簡化機制示意','A simplified view of a biological neuron'),22,MUT)
    elif kind=='switch':
        text(60,105,1150,42,tr('教學簡化：只比較是否達標','Simplified model: test whether the threshold is reached'),23,MUT)
        for i,z in enumerate((1,2,3)):
            y=220+i*125
            for j in range(z):P.rect(225+j*135,y,128,56,fill='#d9edf9',stroke=BLUE,sw=2,radius=6,z=2)
            P.line([(495,y-13),(495,y+75)],ORANGE,4,dash=True,z=4)
            eq(60,y,160,'z='+str(z),28);eq(670,y-3,270,f'{z}'+(r'<2\Rightarrow ' if z<2 else r'\ge2\Rightarrow ')+'o='+str(int(z>=2)),26)
            lamp(1050,y+30,z>=2,31)
        eq(395,150,210,r'\theta=2',28,c=ORANGE)
    elif kind=='weighted':
        pipeline();eq(355,500,600,r'z=\sum_{j=1}^{d} w_jx_j',35,h=110)
        text(65,112,1100,62,tr('輸入：每路訊號　　權重：每路的重要性','Inputs: incoming signals     Weights: their importance'),24,MUT)
        eq(670,230,525,'d',24);text(970,237,210,45,tr('輸入路數','number of inputs'),22,MUT)
        text(875,455,230,50,tr('加權總和','Weighted sum'),25,BLUE,True)
    elif kind=='bias':
        text(60,103,1100,48,tr('總和減掉門檻，就是離達標還差多少','The sum minus the threshold tells us whether the bar is reached'),23,MUT)
        for row,y in enumerate((255,400)):
            arrow(150,y,800,y,MUT,3)
            P.ellipse(580 if row==0 else 385,y,12,fill=BLUE,z=5)
            P.line([(385 if row==0 else 190,y-55),(385 if row==0 else 190,y+55)],ORANGE,4,dash=True,z=4)
            eq(280 if row==0 else 110,y+35,180,r'\theta' if row==0 else '0',27,c=ORANGE)
            eq(505 if row==0 else 310,y-70,150,'z' if row==0 else 's',31,c=BLUE)
            eq(850,y-35,340,r'z\ge\theta' if row==0 else r's=z-\theta\ge0',30)
        arrow(580,305,390,350,PURPLE,3);eq(570,302,200,r'-\theta',29,c=PURPLE)
        eq(140,520,540,r'w_0=-\theta',32,c=ORANGE);eq(700,520,470,r's=z+w_0',32,c=PURPLE)
        label(215,583,390,36,tr('偏置','Bias'),21,MUT)
        label(750,583,390,36,tr('活化前總和','Pre-activation sum'),21,MUT)
    elif kind=='step':
        fx,fy=axis_plot(x=130,y=205,w=630,h=330)
        P.line([(fx(-2),fy(0)),(fx(0),fy(0))],BLUE,7,z=5)
        P.line([(fx(0),fy(1)),(fx(2),fy(1))],GREEN,7,z=5)
        P.ellipse(fx(0),fy(0),10,fill='#fff',stroke=BLUE,sw=4,z=6);P.ellipse(fx(0),fy(1),10,fill=GREEN,z=6)
        eq(730,fy(0)+16,80,'s',28);eq(fx(0)-80,165,65,'o',28)
        eq(170,fy(0)+28,150,'s<0',25,c=BLUE);eq(565,fy(1)-65,150,r's\ge0',25,c=GREEN)
        eq(fx(0)-65,fy(1)-28,48,'1',25);eq(fx(0)-65,fy(0)-32,48,'0',25)
        lamp(1010,310,True,52);eq(835,415,365,r'o=f(s)',32)
        eq(800,485,410,r'f(s)=1\;(s\ge0)',26);eq(800,540,410,r'f(s)=0\;(s<0)',26)
        text(55,104,1160,40,tr('橫軸：活化前總和　　縱軸：輸出','Horizontal: pre-activation sum    Vertical: output'),23,MUT)
    elif kind=='sum':
        for i,(y,val) in enumerate(zip((220,350,480),(1,0,1))):
            dot(200,y,val);eq(250,y-33,300,rf'w_{i+1}x_{i+1}=1\times{val}={val}',25,c=BLUE)
            arrow(585,y,720+i*125,350,BLUE,3)
        for j in range(3):
            P.rect(660+j*140,330,125,65,fill='#d9edf9' if j!=1 else '#fff',stroke=BLUE if j!=1 else GREY,sw=3,dash=j==1,radius=8,z=2)
            label(660+j*140,336,125,52,str([1,0,1][j]),32,BLUE if j!=1 else MUT)
        eq(615,470,570,'z=1+0+1=2',36);text(615,535,570,40,tr('總和：兩路各貢獻一','Total: two inputs each contribute one'),23,MUT)
        eq(55,105,1160,r'\mathbf{x}=(1,0,1),\quad w_1=w_2=w_3=1',30)
    elif kind=='threshold':
        eq(55,100,1160,r'z=1+0+1=2,\quad s=z-\theta,\quad w_0=-\theta',28)
        for i,theta in enumerate((1,2,3)):
            y=220+i*123
            for j in range(2):P.rect(265+j*105,y,100,50,fill='#d9edf9',stroke=BLUE,sw=2,radius=6,z=2)
            for j in range(4):P.line([(265+j*105,y+56),(265+j*105,y+67)],MUT,2,z=2)
            P.line([(265+theta*105,y-20),(265+theta*105,y+67)],ORANGE,4,dash=True,z=5)
            eq(55,y-8,190,r'\theta='+str(theta),29,c=ORANGE)
            eq(640,y-10,300,f's=2-{theta}={2-theta}',28)
            eq(960,y-8,90,'o=',25);lamp(1115,y+25,2>=theta,30)
        text(350,575,780,40,tr('橘虛線是門檻，藍色長度是總和','Orange dashed line: threshold. Blue length: sum.'),22,MUT)
    elif kind in ('and','or'):
        theta=2 if kind=='and' else 1;gate(kind.upper())
        eq(50,100,1170,r'w_1=w_2=1,\quad s=x_1+x_2-'+str(theta)+r',\quad o=1\;(s\ge0)',28)
        text(160,580,1040,40,tr('實心：輸出一　空心：輸出零　淺綠：達標區','Filled: output one   Open: output zero   Green area: threshold reached'),21,MUT)
        text(750,155,440,55,tr('四種輸入，逐一代入','Test all four inputs'),24,PURPLE,True)
    elif kind=='compare':
        gate('AND',x=125,y=240,size=290,calc=False);gate('OR',x=790,y=240,size=290,calc=False)
        label(120,140,310,50,'AND',34,PURPLE);label(785,140,310,50,'OR',34,BLUE)
        arrow(505,355,710,355,ORANGE,6);text(490,405,250,65,tr('降低門檻','Lower the threshold'),25,ORANGE,True)
        eq(65,545,470,'s=x_1+x_2-2',27);eq(740,545,470,'s=x_1+x_2-1',27)
        eq(365,100,550,r'w_1=w_2=1,\quad o=1\Leftrightarrow s\ge0',24)
        text(380,590,560,35,tr('實心一，空心零；線上也達標','Filled one, open zero; the boundary counts'),21,MUT)
    elif kind=='not':
        text(55,102,1160,40,tr('補充｜單輸入的反相邏輯閘','Added | A one-input inverter'),23,ORANGE,True)
        for y,v in ((250,0),(455,1)):
            dot(130,y,v);arrow(175,y,390,y,BLUE,4);lamp(470,y,not v,40)
            eq(60,y+55,510,rf's=-1\times{v}+0.5={.5-v:g}',25)
        fx,fy=axis_plot(x=725,y=240,w=390,h=270,xmin=-.1,xmax=1.1,ymin=-.65,ymax=.65)
        P.line([(fx(0),fy(.5)),(fx(1),fy(-.5))],PURPLE,6,z=5)
        for xx in (0,1):P.ellipse(fx(xx),fy(.5-xx),10,fill=PURPLE,z=6)
        eq(715,150,460,r'w=-1,\quad w_0=0.5',28);eq(700,540,500,'s=-x+0.5',32,c=PURPLE)
        eq(1115,fy(0)+5,45,'x',24);eq(fx(0)-58,180,55,'s',24)
        eq(fx(.5)-50,fy(0)+15,100,'0.5',22);eq(fx(1)-20,fy(0)+15,55,'1',22)
        eq(720,585,470,r'o=1\;(s\ge0),\quad o=0\;(s<0)',24,h=45)
    elif kind=='learn':
        for cx,c in ((230,ORANGE),(630,PURPLE),(1030,BLUE)):
            P.ellipse(cx,345,72,fill='#f1f5f9',stroke=c,sw=4,z=3)
        # Manual setting dial.
        for a in range(0,360,60):
            rad=math.radians(a);P.line([(230+50*math.cos(rad),345+50*math.sin(rad)),(230+62*math.cos(rad),345+62*math.sin(rad))],ORANGE,3,z=4)
        P.line([(230,345),(260,308)],ORANGE,7,z=5)
        eq(570,300,120,r'\sum',44,h=100,c=PURPLE);label(980,300,100,90,'?',52,BLUE)
        arrow(315,345,540,345,ORANGE);arrow(715,345,940,345,BLUE,dash=True)
        text(105,170,280,70,tr('人先設定','A person sets them'),28,ORANGE,True)
        text(500,170,300,70,tr('固定權重運算','Compute with fixed weights'),28,PURPLE,True)
        text(930,170,270,70,tr('從資料學習？','Learn from data?'),28,BLUE,True)
        for j in range(4):dot(920+j*65,495,j%2,BLUE,20)
        eq(95,520,1090,r'o=f(\sum_j w_jx_j+w_0),\quad w_0=-\theta',32,h=110)
        text(70,106,1100,42,tr('McCulloch–Pitts（1943）：固定權重，不自行學習','McCulloch–Pitts (1943): fixed weights, no learning'),23,MUT)
    elif kind=='practice1':
        eq(70,100,1140,r'\mathbf{x}=(1,1,0),\quad w_1=w_2=w_3=1,\quad\theta=2',28)
        for y,v in ((235,1),(350,1),(465,0)):
            dot(200,y,v);arrow(245,y,570,350,BLUE,4)
        P.ellipse(630,350,65,fill='#f0e9fc',stroke=PURPLE,sw=4,z=4);eq(570,300,120,r'\sum',45,h=110,c=PURPLE)
        arrow(705,350,950,350,PURPLE);lamp(1050,350,None,60)
        eq(170,520,960,r'z=\sum_j w_jx_j,\quad s=z-\theta,\quad o=f(s)=?',30,h=110)
    elif kind=='practice2':
        text(70,103,1140,42,tr('三路輸入，權重全部是一；找出合適的門檻','Three inputs, all weights one: choose a threshold'),23,MUT)
        for row,vals in enumerate(((1,1,0),(1,0,0))):
            y=270+row*205
            for j,v in enumerate(vals):dot(190+j*135,y,v)
            arrow(525,y,850,y,BLUE);eq(575,y-65,240,r'\theta=?',32,c=ORANGE);lamp(1000,y,row==0,44)
        text(1035,340,180,50,tr('目標輸出','Target output'),21,MUT)
    elif kind=='practice3':
        eq(60,103,1150,r's=wx+w_0,\quad w>0,\quad x:0\to1',30)
        for y,v in ((260,0),(460,1)):
            dot(180,y,v);arrow(230,y,425,y,BLUE);lamp(505,y,not v,42)
        text(80,565,520,45,tr('NOT 想要的輸出','Desired NOT response'),25,PURPLE,True)
        arrow(740,350,1150,350,MUT,3);P.ellipse(815,350,14,fill=PURPLE,z=5)
        label(900,235,200,100,'?',65,PURPLE);text(755,410,430,90,tr('輸入增加時，總和會往哪走？','When input rises, which way does the sum move?'),28,PURPLE,True)
    elif kind=='quiz1':
        neuron(cx=550,cy=340,scale=.95,annotations=False)
        opts=[tr('A　樹突','A  Dendrites'),tr('B　突觸','B  Synapse'),tr('C　細胞體','C  Soma'),tr('D　軸突','D  Axon')]
        for i,s in enumerate(opts):
            x=60+i*305;P.ellipse(x+22,535,17,fill='#edf3f8',stroke=BLUE,sw=2,z=2);text(x+50,505,230,70,s,26,INK,True)
    elif kind=='quiz2':
        opts=[tr('A　完整的細胞代謝','A  Cell metabolism'),tr('B　累積、過門檻、放電','B  Sum, threshold, fire'),tr('C　神經元的形狀','C  The shape'),tr('D　遺傳物質複製','D  DNA copying')]
        for i,s in enumerate(opts):
            x=70+i*305;cx=x+130
            if i==0:
                for j in range(4):
                    a=j*math.pi/2;P.ellipse(cx+65*math.cos(a),335+65*math.sin(a),19,fill='#d9edf9',stroke=BLUE,sw=2,z=3)
                P.line([(cx,270),(cx+65,335),(cx,400),(cx-65,335),(cx,270)],BLUE,4,arrow='end',z=4)
            elif i==1:
                for j in range(3):P.ellipse(cx-75+j*75,335,23,fill='#e9f3fb',stroke=BLUE,sw=3,z=3)
                arrow(cx-48,335,cx-25,335,BLUE,3);arrow(cx+28,335,cx+50,335,BLUE,3)
            elif i==2:neuron(cx=cx+10,cy=335,scale=.32,annotations=False)
            else:
                for j in range(5):
                    yy=270+j*34;dx=math.sin(j*1.2)*38
                    P.ellipse(cx+dx-25,yy,10,fill=BLUE,z=3);P.ellipse(cx-dx+25,yy,10,fill=PURPLE,z=3);P.line([(cx+dx-25,yy),(cx-dx+25,yy)],MUT,3,z=2)
            text(x,490,285,100,s,25,INK,True)
    elif kind=='quiz3':
        people=[('Rosenblatt','1958'),('McCulloch & Pitts','1943'),('Widrow & Hoff','1960'),('Minsky & Papert','1969')]
        for i,(who,year) in enumerate(people):
            cx=190+i*300
            P.ellipse(cx,270,38,fill='#e3edf6',stroke=BLUE,sw=3,z=3)
            P.poly([(cx-57,385),(cx-36,325),(cx+36,325),(cx+57,385)],fill='#d9e8f5',stroke=BLUE,sw=2,z=3)
            label(cx-135,420,270,85,chr(65+i)+'\n'+who,24,INK)
            label(cx-100,520,200,55,year,34,BLUE)
    elif kind=='quiz4':
        for i,(k,n) in enumerate([('sigmoid','A  Sigmoid'),('relu','B  ReLU'),('step',tr('C　二元階梯','C  Binary step')),('tanh','D  tanh')]):
            x=65+i*305;plotfun(x+18,240,230,240,k);text(x,520,285,70,n,26,INK,True)
            eq(x+235,390,45,'s',21);eq(x+85,190,45,'o',21)
        text(60,112,1150,42,tr('哪一張圖對應這顆神經元的輸出規則？','Which graph matches this neuron’s output rule?'),23,MUT)

def build(lang):
    global LANG,P
    LANG=lang
    title=tr('KP01 神經元與門檻','KP01 Neurons and thresholds')
    d=make(title,title,'kp01');sid='scene-kp01';sc=d['scenes'][sid]
    d['nodes']['frame-kp01']['name']=title
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();d['metadata'].update(createdAt=stamp,updatedAt=stamp)
    manifest=[]
    for i,item in enumerate(PAGES):
        kind,zh,en,tz,te,section,brief,maths=item
        P=Page(prefix=f'{lang}-p{i+1:02d}')
        P.rect(0,0,1280,720,fill='#ffffff',z=0)
        P.rect(0,0,1280,86,fill='#edf4fa',z=1)
        P.rect(0,0,13,86,fill=BLUE,z=2)
        text(48,20,1160,60,tr(zh,en),38 if lang=='zh' else 34,INK,True,name='page-title')
        text(1130,680,100,28,f'{i+1:02d} / {len(PAGES)}',17,MUT)
        start=len(P.nodes)
        draw(kind)
        graphic_ids=[n['id'] for n in P.nodes[start:]]
        # Each page has two beats: the visual argument, then its conclusion.
        footstart=len(P.nodes)
        P.rect(48,640,1184,56,fill='#eaf5ef',radius=12,z=8)
        text(70,650,1130,42,tr(tz,te),27 if lang=='zh' else 25,GREEN,True,name='takeaway')
        footer_ids=[n['id'] for n in P.nodes[footstart:]]
        notes=[tr(*TEACH[kind]),tr(tz,te)]
        if i==0:notes[0]=tr('教材程度：大學生。第二章 KP01。原生圖解審閱稿。','Level: undergraduate. Chapter 2 KP01. Native teaching-diagram review draft.')+' '+notes[0]
        P.check()
        if P.warnings:raise RuntimeError(P.warnings)
        page_nodes=[]
        for n in P.nodes:
            n['sceneId']=sid;n['transform']['x']+=i*1440
            n['initiallyHidden']=n['id'] in graphic_ids+footer_ids
            if n['type']=='text':n['style']['fontFamily']='Noto Sans CJK TC' if lang=='zh' else 'Arial'
            d['nodes'][n['id']]=n;sc['nodeIds'].append(n['id']);page_nodes.append(n['id'])
        for beat,ids in enumerate((graphic_ids,footer_ids)):
            order=len(d['steps']);d['steps'].append(dict(id=f'{lang}-p{i+1}-beat{beat+1}',order=order,label=tr(zh,en)+(' · '+str(beat+1)),durationMs=450,holdMs=500,easing='smooth',target=dict(kind='scene-point',sceneId=sid,point=dict(x=i*1440+640,y=360),zoom=1,referenceViewport=dict(width=1280,height=720)),reveal=dict(timing='after-camera',enter=[dict(nodeId=k,effect='fade',delayMs=0,durationMs=250) for k in ids],exitNodeIds=[]),speakerNotes=notes[beat]))
        manifest.append(dict(page=i+1,kind=kind,title=tr(zh,en),source_section=section,node_ids=page_nodes,final_step=len(d['steps'])-1,extension=i>=13))
    dest=ROOT/'output';dest.mkdir(exist_ok=True)
    name='lec02-kp01-biological-neuron-and-mp-unit'+('-en' if lang=='en' else '')
    path=dest/(name+'.jbd.json');path.write_text(json.dumps(d,ensure_ascii=False,indent=1),encoding='utf-8')
    (dest/(lang+'-pages.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'{lang}: {len(manifest)} pages, {len(d["steps"])} camera/reveal steps, {len(d["nodes"])} editable nodes -> {path}')

def nums():
    assert sum([1,0,1])==2
    th=[dict(theta=t,bias=-t,s=2-t,o=int(2>=t)) for t in (1,2,3)]
    assert [r['o'] for r in th]==[1,1,0]
    truth={name:[dict(x=[a,b],s=a+b-t,o=int(a+b>=t)) for a,b in [(0,0),(0,1),(1,0),(1,1)]] for name,t in [('AND',2),('OR',1)]}
    assert [r['o'] for r in truth['AND']]==[0,0,0,1]
    assert [r['o'] for r in truth['OR']]==[0,1,1,1]
    assert [int(-x+.5>=0) for x in (0,1)]==[1,0]
    (ROOT/'數值驗算.json').write_text(json.dumps(dict(threshold=th,gates=truth,not_gate=[dict(x=x,s=-x+.5,o=int(-x+.5>=0)) for x in (0,1)],convention='s >= 0 includes equality'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('All threshold, AND, OR and NOT assertions passed. No stochastic experiment.')

if __name__=='__main__':
    nums();build('zh');build('en')
