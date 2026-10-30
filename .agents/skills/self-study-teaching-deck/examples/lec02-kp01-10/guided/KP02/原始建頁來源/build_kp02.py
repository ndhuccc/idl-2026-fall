from pathlib import Path
import json,datetime,math
import pagekit as K
from specs import PAGES
from kp02_nums import ONE,TWO,DATA,save
ROOT=Path(__file__).parent
P=None;LANG='zh';SECOND=[]
INK,MUT,BLUE,PURPLE,GREEN,ORANGE,GREY,RED=K.INK,K.MUT,K.BLUE,K.PURPLE,K.GREEN,K.ORANGE,K.GREY,K.RED
text,label,eq,arrow,dot,lamp=K.text,K.label,K.eq,K.arrow,K.dot,K.lamp
def tr(a,b):return a if LANG=='zh' else b
def v(a):return '('+','.join(str(x) for x in a)+')'
def phase2(start):SECOND.extend(n['id'] for n in P.nodes[start:])
def badge(s):text(50,100,1180,42,s,22,MUT)
def node(cx,cy,s,c=BLUE,r=54):
    P.ellipse(cx,cy,r,fill='#eff5fa',stroke=c,sw=3,z=3);label(cx-r,cy-32,2*r,70,s,28,c)
def knob(cx,cy,s,c=PURPLE,r=42):
    P.ellipse(cx,cy,r,fill='#f5f1fb',stroke=c,sw=3,z=3)
    for a in (30,90,150,210,270,330):
        t=math.radians(a);P.line([(cx+(r-10)*math.cos(t),cy+(r-10)*math.sin(t)),(cx+(r-3)*math.cos(t),cy+(r-3)*math.sin(t))],c,2,z=4)
    P.line([(cx,cy),(cx+18,cy-20)],c,5,z=4);eq(cx-r-15,cy+r+6,2*r+30,s,24)
def small_input(cx,cy,x,c=BLUE):
    for j,a in enumerate(x):dot(cx+(j-1)*58,cy,a,c,20)
def triangle(cx,cy,r=12,c=GREEN):return P.poly([(cx,cy-r),(cx-r,cy+r),(cx+r,cy+r)],fill=c,stroke=c,sw=2,z=5)
def plot(w,x=120,y=225,size=280,mode='OR',miss=False):
    lo,hi=-.18,1.18
    fx=lambda a:x+(a-lo)/(hi-lo)*size;fy=lambda b:y+size-(b-lo)/(hi-lo)*size
    a,b,c=w if w is not None else (0,0,-1)
    poly=[(lo,lo),(hi,lo),(hi,hi),(lo,hi)];out=[]
    f=lambda p:a*p[0]+b*p[1]+c
    for aa,bb in zip(poly,poly[1:]+poly[:1]):
        ina,inb=f(aa)>=0,f(bb)>=0
        if ina:out.append(aa)
        if ina!=inb:
            t=-f(aa)/(f(bb)-f(aa));out.append((aa[0]+t*(bb[0]-aa[0]),aa[1]+t*(bb[1]-aa[1])))
    if len(out)>=3:P.poly([(fx(u),fy(vv)) for u,vv in out],fill='#e6f5ec',z=1)
    arrow(fx(lo),fy(0),fx(hi),fy(0),MUT,2);arrow(fx(0),fy(lo),fx(0),fy(hi),MUT,2)
    eq(fx(hi)-10,fy(0)+15,65,'x_1',20);eq(fx(0)-94,fy(hi)-20,65,'x_2',20)
    points=[]
    if b:
        for xx in (lo,hi):
            yy=-(a*xx+c)/b
            if lo-1e-8<=yy<=hi+1e-8:points.append((xx,yy))
    if a:
        for yy in (lo,hi):
            xx=-(b*yy+c)/a
            if lo-1e-8<=xx<=hi+1e-8:points.append((xx,yy))
    points=list(dict.fromkeys(points))
    if len(points)>=2:P.line([(fx(points[0][0]),fy(points[0][1])),(fx(points[-1][0]),fy(points[-1][1]))],ORANGE,4,z=4)
    for (xx,yy),tar in DATA:
        if mode=='XOR':tar=int(xx!=yy)
        if tar:triangle(fx(xx),fy(yy),11)
        else:P.ellipse(fx(xx),fy(yy),11,fill='#e6effb',stroke=BLUE,sw=3,z=5)
        eq(fx(xx)-44,fy(yy)+(12 if yy==0 else -50),88,f'({xx},{yy})',18)
        if miss and int(a*xx+b*yy+c>=0)!=tar:P.ellipse(fx(xx),fy(yy),20,fill='transparent',stroke=RED,sw=4,z=6)
    return fx,fy

def draw(kind):
    if kind=='cover':
        badge(tr('感知器｜沿著錯誤回饋學習','Perceptron | Learning from mistakes'))
        small_input(190,320,[1,0,1]);arrow(280,320,450,320,BLUE)
        knob(525,320,r'\mathbf{w}',PURPLE,65);arrow(610,320,770,320,PURPLE)
        lamp(850,320,True,42);arrow(900,320,1010,320,GREEN)
        P.poly([(1090,265),(1150,320),(1090,375),(1030,320)],fill='#fff1d9',stroke=ORANGE,sw=3,z=3)
        eq(1025,285,130,'y-o',27,c=ORANGE)
        start=len(P.nodes);P.line([(1090,390),(1090,515),(525,515),(525,410)],RED,5,arrow='end',head=15,z=5)
        text(650,525,480,60,tr('猜錯，回來修正','Mistake: update weights'),26,RED,True);phase2(start)
    elif kind=='supervised':
        badge(tr('Rosenblatt，1958｜標籤是提供的正確答案','Rosenblatt, 1958 | A label is a supplied correct answer'))
        knob(250,335,r'\mathbf{w}',PURPLE,68)
        arrow(330,335,510,335,PURPLE);lamp(590,335,True,45);eq(520,430,140,'o',30)
        lamp(955,230,False,45);eq(1030,195,150,'y',30,c=BLUE)
        arrow(660,335,850,335,GREEN);arrow(955,285,955,335,BLUE)
        P.poly([(935,300),(975,340),(935,380),(895,340)],fill='#fff0d8',stroke=ORANGE,sw=3,z=4)
        start=len(P.nodes);P.line([(935,400),(935,530),(250,530),(250,425)],RED,5,arrow='end',z=4)
        text(580,540,520,65,tr('誤差回饋，告訴權重怎麼改','Error drives the update'),26,RED,True);phase2(start)
    elif kind=='bias':
        for j,(xs,ws) in enumerate([('x_1','w_1'),('x_2','w_2'),('x_0=1','w_0')]):
            y=220+j*120;node(200,y,'$'+xs+'$',BLUE if j<2 else ORANGE,58);arrow(265,y,540,340,BLUE if j<2 else ORANGE,4)
            eq(315,y-48,170,ws,28,c=PURPLE if j<2 else ORANGE)
        node(610,340,r'$\sum$',PURPLE,62);arrow(680,340,850,340,PURPLE);node(940,340,'$s$',BLUE,65)
        badge(tr('固定輸入一，對應可學習的偏置','A constant input one carries the learnable bias'))
        start=len(P.nodes)
        eq(70,510,1140,r'\mathbf{x}=(x_1,x_2,1),\quad\mathbf{w}=(w_1,w_2,w_0)',28)
        eq(250,568,790,'s=w_1x_1+w_2x_2+w_0',29);phase2(start)
    elif kind=='loop':
        for cx,s,c in [(165,r'$\mathbf{x}_i$',BLUE),(435,'$s$',PURPLE),(695,'$o_i$',GREEN),(1035,'$y_i-o_i$',ORANGE)]:node(cx,315,s,c,56)
        arrow(230,315,370,315,BLUE);arrow(500,315,630,315,PURPLE);arrow(760,315,970,315,GREEN)
        node(1035,160,'$y_i$',BLUE,38);arrow(1035,202,1035,250,BLUE,3)
        eq(80,410,520,r's=\mathbf{w}^{\top}\mathbf{x}_i',28);eq(630,410,555,r'o_i=f(s)',29)
        start=len(P.nodes);P.line([(1035,380),(1035,535),(435,535),(435,385)],RED,4,arrow='end',z=4)
        text(475,545,570,40,tr('猜錯：更新權重，再看下一筆','Wrong? Update, then move on'),24,RED,True)
        text(65,103,1150,40,tr('正解與預測比較，得到誤差','Compare label and prediction to obtain the error'),23,MUT);phase2(start)
    elif kind=='errors':
        badge(tr('正解減預測：零、正一、負一','Label minus prediction: zero, plus one, minus one'))
        for i,(e,c,es,action) in enumerate([(0,MUT,'e=y-o=0',r'\mathbf{w}^{\prime}=\mathbf{w}'),(1,GREEN,'e=1-0=+1',r'\mathbf{w}^{\prime}=\mathbf{w}+\eta\mathbf{x}'),(-1,RED,'e=0-1=-1',r'\mathbf{w}^{\prime}=\mathbf{w}-\eta\mathbf{x}')]):
            y=225+i*132;eq(55,y-30,300,es,29,c=c)
            P.line([(420,y),(690,y)],GREY,3,z=3);P.ellipse(555,y,12,fill=PURPLE,z=4)
            if e:arrow(555,y,670 if e>0 else 440,y,c,7)
            eq(750,y-30,440,action,27,c=c)
        text(55,575,1160,40,tr('學習率為正，決定每次改多少；輸入決定改哪一路','A positive rate sets the step size; the input selects the affected weights'),22,MUT)
    elif kind=='rule':
        badge(tr('誤差＝正解減預測；學習率決定步長','Error = label minus prediction; learning rate sets step size'))
        for j,s in enumerate(('x_1','x_2','x_0=1')):
            x=245+j*390;node(x,275,'$'+s+'$',BLUE if j<2 else ORANGE,58);arrow(x,338,x,355,PURPLE,4);knob(x,405,'w_'+str(j+1) if j<2 else 'w_0',PURPLE if j<2 else ORANGE)
        eq(55,525,1170,r'\Delta w_j=\eta e x_j,\quad e=y-o,\quad\eta>0',29)
        start=len(P.nodes);eq(120,145,1050,r'\mathbf{w}^{\prime}=\mathbf{w}+\eta e\mathbf{x}',35);phase2(start)
    elif kind=='direction':
        badge(tr('只比較同一筆輸入在更新前後的總和','Compare this same sample’s score before and after the update'))
        eq(70,175,700,r's^{\prime}=\mathbf{w}^{\prime\top}\mathbf{x}',29)
        eq(70,260,700,r'=(\mathbf{w}+\eta e\mathbf{x})^{\top}\mathbf{x}',29)
        start=len(P.nodes);eq(70,345,700,r'=s+\eta e\|\mathbf{x}\|^2',32,c=PURPLE);eq(65,465,720,r'x_0=1\Rightarrow \|\mathbf{x}\|^2\ge1',28)
        for row,e,c in ((0,'+1',GREEN),(1,'-1',RED)):
            y=250+row*220;P.line([(845,y),(1190,y)],MUT,3,z=3);P.ellipse(1000,y,13,fill=PURPLE,z=4)
            arrow(1000,y,1150 if row==0 else 865,y,c,6);eq(885,y-80,260,'e='+e,29,c=c)
        text(840,555,355,45,tr('箭頭長度是方向示意','Arrows show direction only'),21,MUT);phase2(start)
    elif kind=='dataset':
        badge(tr('補充示例｜全零權重、固定資料順序、學習率一','Added example | Zero weights, fixed data order, rate one'))
        plot(None,x=160,y=230,size=300)
        for i,((a,b),target) in enumerate(DATA):
            yy=195+i*90;label(675,yy,80,55,str(i+1),32,PURPLE);eq(780,yy,390,rf'({a},{b})\to y={target}',30)
        text(90,520,510,35,tr('圓：正解零　三角：正解一','Circle: label zero. Triangle: label one.'),21,MUT)
        start=len(P.nodes);eq(90,555,1090,r'\mathbf{w}=(0,0,0),\quad \eta=1,\quad o=1\;(s\ge0)',29);phase2(start)
    elif kind.startswith('update'):
        idx=int(kind[6:])-1;r=ONE['updates'][idx];a,b=r['pair'];old=r['old'];new=r['new'];e=r['e']
        badge(tr(f'補充示例｜第{r["epoch"]}輪；權重順序：第一路、第二路、偏置',f'Added example | Pass {r["epoch"]}; weights: first input, second input, bias'))
        eq(65,155,1150,rf'\mathbf{{x}}={v(r["x"])},\quad y={r["y"]},\quad\eta=1',28)
        score=f'{old[0]}\\cdot {a}+{old[1]}\\cdot {b}+({old[2]})={r["s"]}'
        eq(80,225,1100,'s='+score+rf',\quad o={r["o"]},\quad e={r["y"]}-{r["o"]}={e}',27)
        eq(90,286,1080,r'o=1\;(s\ge0),\quad o=0\;(s<0)',22,h=36)
        for j in range(3):
            yy=340+j*78;eq(75,yy-14,180,['w_1','w_2','w_0'][j],25,c=PURPLE if j<2 else ORANGE)
            # A shared zero baseline makes positive and negative weights comparable.
            P.line([(350,yy+16),(650,yy+16)],GREY,2,z=2);P.ellipse(500+old[j]*80,yy+16,11,fill=BLUE,z=4)
            eq(270,yy-15,130,str(old[j]),25,c=BLUE)
        text(295,552,620,35,tr('藍實心：舊值　紫外圈：新值','Blue dot: old value. Purple ring: new value.'),20,BLUE)
        start=len(P.nodes)
        for j in range(3):
            yy=340+j*78;target=500+new[j]*80
            P.ellipse(target,yy+16,15,fill='transparent',stroke=PURPLE,sw=3,z=5)
            if r['dw'][j]:arrow(500+old[j]*80,yy+16,target,yy+16,RED if e<0 else GREEN,6)
            eq(705,yy-14,475,rf'{old[j]}+1\cdot({e})\cdot {r["x"][j]}={new[j]}',26)
        eq(65,580,1140,rf'\mathbf{{w}}^{{\prime}}={v(old)}+1\cdot({e})\cdot{v(r["x"])}={v(new)}',28,h=50)
        phase2(start)
    elif kind=='boundaries':
        badge(tr('補充示例｜形狀是正解；淺綠區是模型預測一','Added example | Shape: true label. Green region: model predicts one.'))
        for j,(w,formula) in enumerate([([0,1,0],'x_2=0'),([1,1,0],'x_1+x_2=0'),([1,1,-1],'x_1+x_2=1')]):
            x=110+j*400;plot(w,x=x,y=240,size=265,miss=True)
            label(x-20,150,300,50,tr(f'第{j+1}輪結束',f'End of pass {j+1}'),27,PURPLE)
            eq(x-35,530,350,formula,26,c=ORANGE)
        text(115,588,1100,35,tr('圓：標籤零　三角：標籤一　紅圈：錯分','Circle: label zero   Triangle: label one   Red ring: mistake'),21,MUT)
    elif kind=='stop':
        badge(tr('輪＝完整看過四筆；更新＝猜錯後修正一次','Pass: visit all four samples. Update: one correction after a mistake.'))
        arrow(170,300,1145,300,MUT,4)
        for j,n in enumerate((2,2,1,0)):
            cx=220+j*280;node(cx,300,str(n),GREEN if n==0 else PURPLE,58)
            label(cx-100,195,200,50,tr(f'第{j+1}輪',f'Pass {j+1}'),27,PURPLE)
            text(cx-90,390,180,70,tr('次更新','updates'),24,MUT,align='center')
        start=len(P.nodes);eq(125,485,1030,r'2+2+1=5,\qquad\mathbf{w}=(1,1,-1)',34)
        text(200,570,940,42,tr('第4輪四筆都猜對，所以沒有更新','All four predictions are correct in pass four: no update'),25,GREEN,True);phase2(start)
    elif kind=='interference':
        badge(tr('追蹤同一筆原點：正解是零，總和就是偏置','Track the origin: its true label is zero, and its score is the bias'))
        for j,(w,on) in enumerate([([0,0,0],True),([0,0,-1],False),([0,1,0],True)]):
            cx=235+j*400;lamp(cx,320,on,57)
            eq(cx-145,430,290,r'\mathbf{w}='+v(w),27);eq(cx-145,500,290,'s=w_0='+str(w[-1]),27)
            label(cx-155,190,310,60,tr(['起點','修正原點後','修正另一筆後'][j],['Start','After fixing the origin','After another update'][j]),25,PURPLE)
            if j<2:arrow(cx+85,320,cx+305,320,ORANGE)
    elif kind=='separable':
        badge(tr('補充｜兩類、數值向量；標準逐筆流程反覆看固定資料','Added | Two classes, numeric vectors, repeated single-sample passes'))
        plot([1,1,-1],x=170,y=235,size=280,mode='OR');plot(None,x=790,y=235,size=280,mode='XOR')
        label(150,150,330,60,tr('OR：一條線可分開','OR: one line separates'),27,GREEN)
        label(770,150,365,60,tr('XOR：一條線分不開','XOR: no separating line'),27,RED)
        text(90,555,530,65,tr('有限次更新後可停止','Finite convergence is guaranteed'),26,GREEN,True)
        text(725,555,530,65,tr('無法達成整輪零錯誤','A zero-error pass is impossible'),26,RED,True)
    elif kind=='rate':
        badge(tr('補充｜零初始化、固定正學習率、相同資料順序','Added | Zero start, constant positive rate, identical sample order'))
        for j,(w,eta) in enumerate([([1,1,-1],1),([2,2,-2],2)]):
            x=180+j*620;plot(w,x=x,y=225,size=270)
            eq(x-80,145,440,rf'\eta={eta},\quad\mathbf{{w}}={v(w)}',27)
        start=len(P.nodes);eq(120,530,1070,r'\mathbf{w}^{(2)}=2\mathbf{w}^{(1)},\quad s^{(2)}=2s^{(1)}',31)
        text(180,590,980,35,tr('圓：正解零　三角：正解一　淺綠：預測一','Circle: label zero. Triangle: label one. Green: predicted one.'),22,MUT);phase2(start)
    elif kind=='batch':
        badge(tr('同一組舊權重計算全部誤差；三筆為流程示意','All errors use the same old weights; three inputs illustrate the flow'))
        for j,y in enumerate((230,350,470)):
            small_input(185,y,[1,j%2,1]);eq(280,y-32,340,rf'(y_{j+1}-o_{j+1})\mathbf{{x}}_{j+1}',25)
            arrow(625,y,780,350,BLUE,3)
        node(850,350,r'$\sum$',PURPLE,61);arrow(922,350,1080,350,PURPLE);knob(1140,350,r'\mathbf{w}',ORANGE,44)
        start=len(P.nodes);eq(95,500,1090,r'\mathbf{w}^{\prime}=\mathbf{w}+\eta\sum_{i=1}^{b}(y_i-o_i)\mathbf{x}_i',30,h=110)
        eq(940,170,80,'b',25);text(1030,177,200,45,tr('批次筆數','batch size'),22,MUT);phase2(start)
    elif kind=='next':
        badge(tr('本節：零與一的標籤、猜錯才更新','This unit: zero/one labels and updates driven by mistakes'))
        for xx,val in ((200,'0'),(400,'1'),(900,'-1'),(1100,'+1')):node(xx,300,'$'+val+'$',BLUE if xx<500 else PURPLE,56)
        arrow(490,300,775,300,ORANGE);label(560,345,200,70,'?',52,ORANGE)
        eq(70,485,1140,r'\mathbf{w}^{\prime}=\mathbf{w}+\eta(y-o)\mathbf{x}',34)
        text(750,430,450,45,tr('KP03：幾何意義','KP03: geometric meaning'),28,PURPLE,True)
    elif kind=='practice1':
        badge(tr('輸出？要更新嗎？更新後權重是多少？','What is the output? Update or not? What are the new weights?'))
        eq(65,155,1150,r'\mathbf{w}=(1,1,-1),\quad\mathbf{x}=(1,0,1),\quad y=0,\quad\eta=1',27)
        for j,n in enumerate((1,1,-1)):knob(225+j*220,320,str(n),PURPLE,45)
        arrow(765,320,950,320,PURPLE);lamp(1060,320,None,55)
        eq(80,500,1120,r's=\mathbf{w}^{\top}\mathbf{x},\quad e=y-o,\quad\mathbf{w}^{\prime}=?',31)
    elif kind=='practice2':
        badge(tr('從全零權重開始，保持 OR 的正確標籤','Start from zero weights and keep the OR labels'))
        for j,(a,b,tar) in enumerate([(1,1,1),(1,0,1),(0,1,1),(0,0,0)]):
            cx=190+j*300;small_input(cx,300,[a,b],BLUE);eq(cx-125,390,250,'y='+str(tar),29)
            label(cx-80,185,160,60,str(j+1),33,PURPLE)
            if j<3:arrow(cx+90,300,cx+210,300,ORANGE,3)
        eq(170,505,940,r'\mathbf{w}=(0,0,0),\quad s=0\Rightarrow o=1',31)
        text(450,580,550,40,tr('第一個更新位置：？','First update position: ?'),26,PURPLE,True)
    elif kind=='practice3':
        badge(tr('相同資料順序、全零起點；比較每一次更新','Same data order and zero start: compare every update'))
        for j,eta in enumerate((1,2)):
            y=260+j*220;eq(80,y-35,230,r'\eta='+str(eta),33,c=ORANGE);node(425,y,r'$\mathbf{0}$',BLUE,58);arrow(495,y,805,y,PURPLE);knob(900,y,r'\mathbf{w}',PURPLE,50);label(1030,y-35,140,70,'?',48,PURPLE)
    elif kind=='quiz1':
        badge(tr('所有選項都在同一個歷史問題中比較','Compare all four options for the same historical question'))
        opts=[tr('A　Widrow 與 Hoff','A  Widrow & Hoff'),tr('B　Rosenblatt','B  Rosenblatt'),tr('C　McCulloch 與 Pitts','C  McCulloch & Pitts'),tr('D　Minsky 與 Papert','D  Minsky & Papert')]
        for j,s in enumerate(opts):
            cx=180+j*300;P.ellipse(cx,270,35,fill='#e5eff8',stroke=BLUE,sw=3,z=3);P.poly([(cx-50,390),(cx-35,322),(cx+35,322),(cx+50,390)],fill='#e5eff8',stroke=BLUE,sw=2,z=3)
            text(cx-135,435,285,65,s,24,INK,True,align='center')
            if j==1:text(cx-135,495,285,90,tr('康乃爾航空實驗室','Cornell Aeronautical\nLaboratory'),22,MUT,align='center')
    elif kind=='quiz2':
        badge(tr('Perceptron 的學習規則與古典 Hebbian 規則的關係？','How does the Perceptron rule relate to classical Hebbian learning?'))
        opts=[tr('A　完全相同','A  Identical'),tr('B　監督式，本質不同','B  Supervised, essentially different'),tr('C　完全不學習','C  No learning'),tr('D　Hebbian 是後繼','D  Hebbian came later')]
        for j,s in enumerate(opts):
            cx=180+j*300;node(cx-40,315,'',BLUE,27);node(cx+55,315,'',PURPLE,27);arrow(cx-8,315,cx+22,315,ORANGE,3)
            label(cx-135,450,285,110,s,25,INK)
    elif kind=='quiz3':
        badge(tr('條件：預測與正解相同','Condition: prediction equals the correct label'))
        eq(60,150,1160,r'o_i=y_i,\quad\mathbf{w}^{\prime}=\mathbf{w}+\eta(y_i-o_i)\mathbf{x}_i',29)
        opts=[tr('A　權重放大學習率倍','A  Scale weights by the rate'),tr('B　更新量零，不更新','B  Zero correction; no update'),tr('C　權重反轉','C  Flip the weights'),tr('D　學習率歸零','D  Set the rate to zero')]
        for j,s in enumerate(opts):
            cx=180+j*300;knob(cx,340,r'\mathbf{w}',PURPLE,48);label(cx-135,480,285,120,s,25,INK)
    elif kind=='quiz4':
        badge(tr('批次與單樣本更新規則有什麼關係？','How do batch and single-sample update rules relate?'))
        small_input(170,240,[1,0,1]);arrow(270,240,480,240,BLUE);node(550,240,r'$\sum$',PURPLE,48);arrow(608,240,850,240,PURPLE);eq(910,200,260,'b=1',34,c=ORANGE)
        opts=[tr('A　互相矛盾','A  Contradictory'),tr('B　批次加總誤差項\n單筆去掉求和','B  Sum error terms;\none sample omits sum'),tr('C　批次版不收斂','C  Batch never converges'),tr('D　只在分類時等價','D  Equal only for classification')]
        for j,s in enumerate(opts):label(45+j*305,395,285,140,s,23,INK)
        eq(60,535,1160,r'\mathbf{w}^{\prime}=\mathbf{w}+\eta\sum_{i=1}^{b}(y_i-o_i)\mathbf{x}_i',26,h=100)

NOTES={
'supervised':('Rosenblatt 在一九五八年的感知器加入由標籤驅動的回饋。正確標籤是一個外部提供的答案，模型先猜，再比較。這和古典Hebbian一起活躍就增強連結的規則不同。','Rosenblatt’s 1958 Perceptron uses feedback driven by a supplied label. Predict first, then compare with the answer. This differs from classical Hebbian co-activation strengthening.'),
'bias':('向量最後補上一個固定的一，對應的權重就是偏置。因此加總與偏置能用同一個內積表達，也能用同一條學習規則修正。這裡把偏置放在向量最後，整份都維持相同順序。','Append a constant one to the input; its weight is the bias. One inner product and one update rule can now handle both ordinary weights and the bias. We consistently place the bias last.'),
'direction':('把更新規則代入新權重的內積，最後多出學習率、誤差與輸入長度平方的乘積。學習率是正的，而偏置輸入確保長度平方至少一，所以誤差決定總和移動的方向；這不保證一次跨過零，也不保證其他資料改善。','Substitute the weight update into the new inner product. The added term is learning rate times error times squared input norm. The rate is positive and the constant input makes the squared norm at least one. The error therefore determines direction, not immediate correction or improvement on other samples.'),
'separable':('線性可分是存在一條直線把這些二維資料的兩類完全分開。標準逐筆感知器對可分資料有有限更新的收斂保證。不可分時，若持續循環且只以整輪零錯誤為停止條件，就不能達成停止；實務上仍可另設迭代上限。','Linear separability means a line can perfectly separate these two-dimensional classes. The standard single-sample Perceptron converges after finitely many updates on separable data. Repeated passes on nonseparable data cannot meet a zero-error stopping rule; an implementation may still impose a pass limit.'),
'rate':('這個比例結論限定零初始化、固定正學習率、相同資料順序與同樣的零點約定。程式逐一核對十六次樣本訪問：學習率二的每組權重都是學習率一的兩倍，每次預測也相同。不能把這個結論推廣成所有模型都不受學習率影響。','This scaling result requires zero initialization, a constant positive rate, identical sample order, and the same convention at zero. All sixteen visits were checked: rate two gives twice the weights and identical predictions. This is not a claim about learning rates in all models.'),
'batch':('批次版先用相同的舊權重，算出每一筆的輸出與誤差，把修正項加起來後更新一次。逐筆版每看完一筆就可能改權重，因此多筆批次一般不等於逐筆連做；批次只有一筆時才直接退化成單筆版。','A batch evaluates all outputs and errors under the same old weights, sums the correction terms, and updates once. Sequential processing can change weights between samples, so a multi-sample batch generally differs from sequential updates. A batch of one reduces directly to the single-sample rule.')}

def build(lang):
    global LANG,P,SECOND
    LANG=lang;K.LANG=lang
    title=tr('KP02 猜錯才改權重','KP02 Update only when wrong');d=K.make(title,title,'kp02');sid='scene-kp02';sc=d['scenes'][sid]
    d['nodes']['frame-kp02']['name']=title
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['metadata'].update(createdAt=now,updatedAt=now)
    manifest=[];warnings=[]
    for i,(kind,zh,en,tz,te,sec,brief) in enumerate(PAGES):
        P=K.Page(prefix=f'{lang}-p{i+1:02d}');K.P=P;SECOND=[]
        P.rect(0,0,1280,720,fill='#ffffff',z=0);P.rect(0,0,1280,86,fill='#edf4fa',z=1);P.rect(0,0,13,86,fill=BLUE,z=2)
        text(48,20,1160,60,tr(zh,en),36 if lang=='zh' else 31,INK,True,name='page-title');text(1130,684,100,28,f'{i+1:02d} / {len(PAGES)}',17,MUT)
        start=len(P.nodes);draw(kind);all_ids=[n['id'] for n in P.nodes[start:]]
        first=[k for k in all_ids if k not in SECOND];groups=[first]+([SECOND] if SECOND else [])
        footer=len(P.nodes);P.rect(48,640,1184,56,fill='#eaf5ef',radius=12,z=8);text(70,650,1130,42,tr(tz,te),26 if lang=='zh' else 24,GREEN,True,name='takeaway')
        groups.append([n['id'] for n in P.nodes[footer:]])
        notes=tr(*NOTES[kind]) if kind in NOTES else tr(tz,te)
        if kind.startswith('update'):
            r=ONE['updates'][int(kind[6:])-1]
            notes=tr(f'這是第{r["epoch"]}輪遇到的輸入{tuple(r["pair"])}。正確標籤是{r["y"]}，目前總和是{r["s"]}，預測{r["o"]}，所以誤差是{r["e"]}。',f'In pass {r["epoch"]}, input {tuple(r["pair"])} has label {r["y"]}. Its current score is {r["s"]}, prediction {r["o"]}, and error {r["e"]}.')
        P.check()
        if P.warnings:warnings.append((i+1,P.warnings))
        for n in P.nodes:
            n['sceneId']=sid;n['transform']['x']+=i*1440;n['initiallyHidden']=n['id'] in all_ids+[k for k in groups[-1]]
            if n['type']=='text':n['style']['fontFamily']='Noto Sans CJK TC' if lang=='zh' else 'Arial'
            d['nodes'][n['id']]=n;sc['nodeIds'].append(n['id'])
        for beat,ids in enumerate(groups):
            note=notes if beat==0 else tr(tz,te)
            if kind.startswith('update') and beat==1:
                r=ONE['updates'][int(kind[6:])-1];note=tr(f'用學習率一乘誤差，再逐路乘上輸入，修正量為{tuple(r["dw"])}。加回舊權重後，得到{tuple(r["new"])}。輸入為零的分量不改，固定輸入一讓偏置也一起改。',f'The correction is {tuple(r["dw"])}: multiply rate one by the error and each augmented input component. Adding it gives {tuple(r["new"])}. Zero inputs do not change their weights; the constant input updates the bias.')
            order=len(d['steps']);d['steps'].append(dict(id=f'{lang}-p{i+1}-beat{beat+1}',order=order,label=tr(zh,en)+' · '+str(beat+1),durationMs=450,holdMs=500,easing='smooth',target=dict(kind='scene-point',sceneId=sid,point=dict(x=i*1440+640,y=360),zoom=1,referenceViewport=dict(width=1280,height=720)),reveal=dict(timing='after-camera',enter=[dict(nodeId=k,effect='fade',delayMs=0,durationMs=250) for k in ids],exitNodeIds=[]),speakerNotes=note))
        manifest.append(dict(page=i+1,kind=kind,title=tr(zh,en),source_section=sec,first_step=len(d['steps'])-len(groups),final_step=len(d['steps'])-1,beats=len(groups),extension=i>=20))
    out=ROOT/'output';out.mkdir(exist_ok=True);stem='lec02-kp02-perceptron-learning-rule'+('-en' if lang=='en' else '')
    (out/(stem+'.jbd.json')).write_text(json.dumps(d,ensure_ascii=False,indent=1));(out/(lang+'-pages.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=1))
    print(lang,len(manifest),'pages;',len(d['steps']),'steps;',len(d['nodes']),'native nodes')
    if warnings:raise RuntimeError(warnings)
if __name__=='__main__':save();build('zh');build('en')
