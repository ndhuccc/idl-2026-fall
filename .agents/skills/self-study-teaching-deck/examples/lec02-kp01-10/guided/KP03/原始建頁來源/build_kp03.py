from pathlib import Path
import json,datetime,math
import pagekit as K
import visual_story
from specs import PAGES
from kp03_nums import DIST,SIGNED,COMB,TIE,save
ROOT=Path(__file__).parent
P=None;LANG='zh';SECOND=[]
INK,MUT,BLUE,PURPLE,GREEN,ORANGE,GREY,RED=K.INK,K.MUT,K.BLUE,K.PURPLE,K.GREEN,K.ORANGE,K.GREY,K.RED
text,label,eq,arrow,dot,lamp=K.text,K.label,K.eq,K.arrow,K.dot,K.lamp
def tr(a,b):return a if LANG=='zh' else b
def v(a):return '('+','.join(str(x) for x in a)+')'
def phase2(start):SECOND.extend(n['id'] for n in P.nodes[start:])
def badge(s):text(50,100,1180,42,s,22,MUT)
def node(cx,cy,s,c=BLUE,r=54):
    P.ellipse(cx,cy,r,fill='#eff5fa',stroke=c,sw=3,z=3)
    fs=(21 if r<60 else 24) if LANG=='en' and '$' not in s and len(s)>6 else 28
    label(cx-r,cy-32,2*r,70,s,fs,c)
def knob(cx,cy,s,c=PURPLE,r=42):
    P.ellipse(cx,cy,r,fill='#f5f1fb',stroke=c,sw=3,z=3)
    for a in (30,90,150,210,270,330):
        t=math.radians(a);P.line([(cx+(r-10)*math.cos(t),cy+(r-10)*math.sin(t)),(cx+(r-3)*math.cos(t),cy+(r-3)*math.sin(t))],c,2,z=4)
    P.line([(cx,cy),(cx+18,cy-20)],c,5,z=4)
    if s:eq(cx-r-15,cy+r+6,2*r+30,s,24)
def small_input(cx,cy,x,c=BLUE):
    for j,a in enumerate(x):dot(cx+(j-1)*58,cy,a,c,20)
def triangle(cx,cy,r=12,c=GREEN):return P.poly([(cx,cy-r),(cx-r,cy+r),(cx+r,cy+r)],fill=c,stroke=c,sw=2,z=5)

def plot(w=(1,1),b=-1.5,x=140,y=210,size=365,points=True,active=None,normal=False,projection=None):
    lo,hi=-.22,1.65
    fx=lambda q:x+(q-lo)/(hi-lo)*size;fy=lambda q:y+size-(q-lo)/(hi-lo)*size
    score=lambda a:w[0]*a[0]+w[1]*a[1]+b
    poly=[(lo,lo),(hi,lo),(hi,hi),(lo,hi)];out=[]
    for a,c in zip(poly,poly[1:]+poly[:1]):
        ia,ic=score(a)>=0,score(c)>=0
        if ia:out.append(a)
        if ia!=ic:
            t=-score(a)/(score(c)-score(a));out.append((a[0]+t*(c[0]-a[0]),a[1]+t*(c[1]-a[1])))
    if len(out)>=3:P.poly([(fx(u),fy(vv)) for u,vv in out],fill='#e8f5ed',z=1)
    arrow(fx(lo),fy(0),fx(hi),fy(0),MUT,2);arrow(fx(0),fy(lo),fx(0),fy(hi),MUT,2)
    eq(fx(hi)-5,fy(0)+10,65,'x_1',21);eq(fx(0)-85,fy(hi)-12,65,'x_2',21)
    cuts=[]
    for val in (lo,hi):
        if w[1]:
            yy=-(w[0]*val+b)/w[1]
            if lo<=yy<=hi:cuts.append((val,yy))
        if w[0]:
            xx=-(w[1]*val+b)/w[0]
            if lo<=xx<=hi:cuts.append((xx,val))
    cuts=list(dict.fromkeys(cuts))
    if len(cuts)>=2:P.line([(fx(cuts[0][0]),fy(cuts[0][1])),(fx(cuts[-1][0]),fy(cuts[-1][1]))],ORANGE,4,z=4)
    if points:
        if size>330:text(x-55,y-52,size+180,36,tr('圓：負類　三角：正類','Circle: negative. Triangle: positive.'),20,MUT)
        for j,r in enumerate(DIST):
            xx,yy=r['x'];c=GREY if active is not None and j not in active else (GREEN if r['y']==1 else BLUE)
            if r['y']==1:triangle(fx(xx),fy(yy),10,c)
            else:P.ellipse(fx(xx),fy(yy),10,fill='#e8f1fa',stroke=c,sw=3,z=5)
            eq(fx(xx)-48,fy(yy)+(10 if yy==0 else -45),96,f'({xx},{yy})',18)
            if int(score((xx,yy))>=0)*2-1!=r['y'] and (active is None or j in active):P.ellipse(fx(xx),fy(yy),19,fill='transparent',stroke=RED,sw=3,z=6)
    if projection is not None:
        xx,yy=projection;s=score((xx,yy));den=w[0]**2+w[1]**2;foot=(xx-s*w[0]/den,yy-s*w[1]/den)
        P.line([(fx(xx),fy(yy)),(fx(foot[0]),fy(foot[1]))],PURPLE,4,dash=True,z=5)
        P.ellipse(fx(foot[0]),fy(foot[1]),6,fill=PURPLE,z=6)
    if normal:
        base=(.75,.75);end=(1.35,1.35)
        arrow(fx(base[0]),fy(base[1]),fx(end[0]),fy(end[1]),PURPLE,5)
        eq(fx(end[0])-30,fy(end[1])-70,180,r'\mathbf{w}',30,c=PURPLE)
        u=.13/math.sqrt(2)
        P.line([(fx(.75+u),fy(.75-u)),(fx(.75+2*u),fy(.75)),(fx(.75+u),fy(.75+u))],PURPLE,2,z=5)
    return fx,fy

def legend(y=595):text(70,y,1160,34,tr('圓：負類　三角：正類　紅圈：錯分　淺綠：預測正類','Circle: negative label. Triangle: positive label. Red ring: mistake. Green: predicted positive.'),20,MUT)

def draw(kind):
    if kind=='cover':
        badge(tr('同一條規則，分別從反號差值與錯分損失來看','One rule viewed through opposite signs and a mistake loss'))
        for cx,c,s in [(260,BLUE,tr('代數','Algebra')),(1020,PURPLE,tr('幾何','Geometry'))]:node(cx,285,s,c,65)
        for cx,val in [(180,'+1'),(340,'-1')]:node(cx,465,'$'+val+'$',BLUE,36)
        P.line([(920,470),(1090,405)],ORANGE,4,z=4);triangle(990,420,12);P.ellipse(1000,485,12,fill='#e5eff8',stroke=BLUE,sw=3,z=5)
        start=len(P.nodes);arrow(340,285,510,355,BLUE);arrow(940,285,770,355,PURPLE)
        eq(390,355,520,r'\mathbf{w}^{\prime}=\mathbf{w}+2\eta y\mathbf{x}',30)
        label(440,475,400,55,tr('只在猜錯時','Only on a mistake'),28,RED);phase2(start)
    elif kind=='coding':
        badge(tr('標籤和輸出都用同一組正負值','Use the same signed values for labels and predictions'))
        for j,(old,new) in enumerate([('0','-1'),('1','+1')]):
            yy=260+j*210;node(180,yy,'$'+old+'$',BLUE,43);arrow(240,yy,405,yy,ORANGE,4);node(470,yy,'$'+new+'$',PURPLE,48)
        fx,fy=K.axis_plot(x=735,y=230,w=385,h=285,xmin=-2,xmax=2,ymin=-1.4,ymax=1.4)
        P.line([(fx(-2),fy(-1)),(fx(0),fy(-1))],BLUE,5,z=5);P.line([(fx(0),fy(1)),(fx(2),fy(1))],GREEN,5,z=5)
        P.ellipse(fx(0),fy(-1),9,fill='#fff',stroke=BLUE,sw=3,z=6);P.ellipse(fx(0),fy(1),9,fill=GREEN,z=6)
        eq(fx(0)-70,fy(-1)-25,60,'-1',24);eq(fx(0)-70,fy(1)-25,60,'+1',24);eq(1110,fy(0)+5,60,'s',24)
        start=len(P.nodes);eq(690,540,520,r'o=\operatorname{sign}(s),\quad\operatorname{sign}(0)=+1',24);phase2(start)
    elif kind=='four':
        badge(tr('正解是標籤，輸出是模型的預測','The label is the answer; the output is the prediction'))
        for j,r in enumerate(COMB):
            col=j%2;row=j//2;cx=355+col*570;cy=260+row*235;c=GREEN if not r['wrong'] else RED
            P.rect(cx-240,cy-70,480,165,fill='#f0f7f3' if not r['wrong'] else '#fdf1ef',stroke=c,sw=2,radius=18,z=1)
            node(cx-165,cy,'$'+str(r['y'])+'$',BLUE,35);node(cx-40,cy,'$'+str(r['o'])+'$',PURPLE,35)
            eq(cx+20,cy-34,210,f'{r["y"]}-({r["o"]})={r["error"]}',29,c=c)
            text(cx-210,cy+52,440,37,tr('猜錯：要更新','Wrong: update') if r['wrong'] else tr('猜對：不更新','Correct: no update'),24,c,True)
        text(90,151,1100,40,tr('每格先看正解，再看輸出，最後相減','In each panel: label, output, then subtraction'),23,MUT)
    elif kind=='algebra':
        badge(tr('前提：正解與輸出反號','Condition: label and prediction have opposite signs'))
        node(245,290,'$y$',BLUE,58);node(245,475,'$-y$',PURPLE,58);arrow(315,290,560,345,BLUE);arrow(315,475,560,400,PURPLE)
        eq(500,190,690,'o=-y',34);eq(500,315,690,'y-o=y-(-y)=2y',34)
        start=len(P.nodes);eq(410,450,790,r'\Delta\mathbf{w}=\eta(y-o)\mathbf{x}=2\eta y\mathbf{x}',32)
        label(520,550,600,50,tr('只在猜錯時使用','Use this only on a mistake'),28,RED);phase2(start)
    elif kind=='rule':
        badge(tr('本頁的更新向量包含偏置；最後一路輸入固定為一','Here the update vector includes the bias, with a constant input one'))
        eq(65,170,1150,r'\mathbf{x}=(x_1,x_2,1),\quad\mathbf{w}=(w_1,w_2,w_0)',28)
        for cx,c in ((305,BLUE),(975,PURPLE)):knob(cx,340,r'\mathbf{w}',c,61)
        arrow(420,340,850,340,ORANGE,5);eq(465,260,380,r'\eta^{\prime}=2\eta',35,c=ORANGE)
        eq(80,455,570,r'\mathbf{w}^{\prime}=\mathbf{w}+2\eta y\mathbf{x}',29);eq(700,455,500,r'\mathbf{w}^{\prime}=\mathbf{w}+\eta^{\prime}y\mathbf{x}',29)
        start=len(P.nodes);eq(155,545,980,r'w_0^{\prime}=w_0+2\eta y\quad(o\ne y)',30);phase2(start)
    elif kind=='normal':
        badge(tr('示例：幾何權重不含偏置；邊界上兩點都滿足同一等式','Example: geometric weights exclude bias; two boundary points satisfy the same equation'))
        fx,fy=plot(points=False,normal=True)
        for point,s in [((.5,1),'A'),((1,.5),'B')]:
            P.ellipse(fx(point[0]),fy(point[1]),8,fill=MUT,z=6);eq(fx(point[0])-40,fy(point[1])-52,80,s,25)
        arrow(fx(.5),fy(1),fx(1),fy(.5),BLUE,3)
        eq(640,180,560,r'\mathbf{w}^{\top}\mathbf{x}_A+w_0=0',28)
        eq(640,300,560,r'\mathbf{w}^{\top}\mathbf{x}_B+w_0=0',28)
        start=len(P.nodes);eq(610,440,610,r'\mathbf{w}^{\top}(\mathbf{x}_A-\mathbf{x}_B)=0',30)
        label(650,540,560,65,tr('相減後，沿線位移與權重垂直','Displacement is perpendicular to weights'),24,PURPLE);phase2(start)
    elif kind=='distance':
        badge(tr('幾何權重不可為零，偏置另外計算','The geometric weight vector must be nonzero; keep the bias separate'))
        plot(projection=(0,1),active=[1],normal=True)
        text(80,585,500,35,tr('虛線：點到邊界的垂線','Dashed segment: perpendicular projection'),21,MUT)
        eq(625,165,590,r's=\mathbf{w}^{\top}\mathbf{x}+w_0',29)
        start=len(P.nodes);eq(615,290,600,r'r=\frac{s}{\|\mathbf{w}\|}',36,h=115,c=PURPLE)
        eq(625,475,590,r'\|\mathbf{w}\|>0',29)
        label(650,540,530,50,tr('分數不等於距離','Score is not distance'),28,PURPLE);phase2(start)
    elif kind=='labeldistance':
        badge(tr('本頁先看不在邊界上的點','First consider points away from the decision boundary'))
        plot(normal=True)
        eq(625,180,590,r'd=yr=\frac{ys}{\|\mathbf{w}\|}',34,h=115)
        start=len(P.nodes);eq(650,350,530,r's\ne0,\quad d>0',30,c=GREEN);text(745,420,400,50,tr('在對的一側','On the correct side'),26,GREEN,True)
        eq(650,475,530,'d<0',30,c=RED);text(790,540,300,45,tr('在錯的一側','On the wrong side'),26,RED,True);phase2(start)
    elif kind=='tie':
        badge(tr('補充釐清｜沿用 sign 在零點取正一的約定','Clarification | Keep the convention that sign at zero is plus one'))
        plot(b=0,active=[0])
        eq(650,165,570,r'\mathbf{w}=(1,1),\quad w_0=0',28)
        eq(650,260,570,r'\mathbf{x}=(0,0),\quad y=-1',29)
        start=len(P.nodes);eq(650,365,570,r's=0,\quad o=+1\ne y',31,c=RED)
        eq(650,465,570,'d=0',36,c=PURPLE)
        label(620,550,600,48,tr('距離零，這一點仍然猜錯','Zero distance, yet this point is wrong'),25,RED);phase2(start)
    elif kind in ('distancesA','distancesB'):
        inds=[0,1] if kind=='distancesA' else [2,3]
        badge(tr('空間權重是一、一，偏置負一點五；以精確根號計算','Spatial weights: one, one. Bias: minus 1.5. Use the exact square root.'))
        plot(active=inds)
        for row,j in enumerate(inds):
            r=DIST[j];a,b=r['x'];y=190+row*230;c=RED if r['d']<0 else GREEN
            eq(615,y,615,rf's={a}+{b}-1.5={r["s"]:g}',26)
            eq(620,y+67,610,rf'd=\frac{{({r["y"]})({r["s"]:g})}}{{\sqrt{{2}}}}\approx {r["d"]:.2f}',30,h=110,c=c)
        eq(145,575,440,r'\|\mathbf{w}\|=\sqrt{2}',25)
    elif kind=='loss':
        badge(tr('錯分集合只收猜錯的點；這裡使用未正規化分數','The mistake set contains wrong predictions; this loss uses unnormalized scores'))
        for j,pair in enumerate(((0,1),(1,0))):
            yy=240+j*195;triangle(210,yy,25);P.ellipse(210,yy,40,fill='transparent',stroke=RED,sw=4,z=5)
            eq(270,yy-30,300,v(pair),28,c=RED);arrow(530,yy,760,335,RED,4)
        node(840,335,r'$\sum$',PURPLE,60)
        eq(590,160,630,r'E=-\sum_{i\in M}y_i s_i',31,h=110)
        start=len(P.nodes);eq(540,455,690,'E=-1(-0.5)-1(-0.5)=1',27)
        text(550,545,650,70,tr('邊界錯分可能貢獻零；零損失不等於零錯誤','A boundary mistake may contribute zero; zero loss is not zero mistakes'),23,RED,True);phase2(start)
    elif kind=='gradient':
        badge(tr('先固定錯分集合；在線上可能沒有唯一梯度','Hold the mistake set fixed; the boundary may not have a unique gradient'))
        # A genuine vector picture: x=(0,1) contributes in the positive vertical direction.
        fx,fy=K.axis_plot(x=145,y=220,w=330,h=310,xmin=-.2,xmax=1.6,ymin=-.2,ymax=1.6)
        arrow(fx(0),fy(0),fx(0),fy(1),GREEN,6);eq(210,200,300,r'y\mathbf{x}=(0,1)',25,c=GREEN)
        eq(610,155,610,r'\nabla_{\mathbf{w}} E=-\sum_{i\in M}y_i\mathbf{x}_i',29,h=115)
        eq(615,335,600,r'\frac{\partial E}{\partial w_0}=-\sum_{i\in M}y_i',28,h=115)
        start=len(P.nodes);label(560,535,660,70,tr('負梯度：往標籤乘輸入的方向','Negative gradient: label times input'),26,GREEN);phase2(start)
    elif kind=='descent':
        badge(tr('幾何權重與偏置可分開寫，也可用擴增向量一起寫','Write spatial weights and bias separately, or use one augmented vector'))
        for cx,c,s in [(230,BLUE,tr('代數','Algebra')),(1050,PURPLE,tr('梯度','Gradient'))]:node(cx,330,s,c,62)
        eq(330,160,620,r'\mathbf{w}^{\prime}=\mathbf{w}-\eta^{\prime}\nabla E',30)
        eq(340,285,600,r'=\mathbf{w}+\eta^{\prime}\sum_{i\in M}y_i\mathbf{x}_i',28,h=110)
        start=len(P.nodes);arrow(310,330,520,500,BLUE,3);arrow(970,330,780,500,PURPLE,3)
        eq(330,490,620,r'\mathbf{w}^{\prime}=\mathbf{w}+2\eta y_i\mathbf{x}_i',30)
        eq(410,565,490,r'\eta^{\prime}=2\eta,\quad o_i\ne y_i',27,c=ORANGE);phase2(start)
    elif kind=='numeric_update':
        badge(tr('輸入為零、一；正解正一；學習率零點五','Input: zero, one. Label: plus one. Learning rate: 0.5.'))
        eq(65,145,1150,r'\mathbf{x}_{\rm aug}=(0,1,1),\quad\mathbf{w}_{\rm aug}=(1,1,-1.5)',28)
        old=[1,1,-1.5];new=[1,2,-.5]
        for j in range(3):
            yy=285+j*100;eq(70,yy-25,190,['w_1','w_2','w_0'][j],27)
            P.line([(340,yy),(680,yy)],GREY,2,z=2);P.ellipse(480+old[j]*70,yy,10,fill=BLUE,z=4)
            eq(255,yy-29,110,str(old[j]),25,c=BLUE)
        start=len(P.nodes)
        for j,xx in enumerate((0,1,1)):
            yy=285+j*100;P.ellipse(480+new[j]*70,yy,15,fill='transparent',stroke=PURPLE,sw=3,z=5)
            if xx:arrow(480+old[j]*70,yy,480+new[j]*70,yy,GREEN,5)
            eq(730,yy-30,450,rf'{old[j]:g}+2(0.5)(+1)({xx})={new[j]:g}',25)
        eq(70,555,1140,r'\mathbf{w}_{\rm aug}^{\prime}=(1,1,-1.5)+2(0.5)(+1)(0,1,1)=(1,2,-0.5)',26,h=65);phase2(start)
    elif kind=='numeric_score':
        badge(tr('同一個點在更新前後，從錯的一側移到對的一側','The same point changes from the wrong side to the correct side'))
        plot(x=140,y=215,size=300,active=[1]);fx,fy=plot(w=(1,2),b=-.5,x=825,y=215,size=300,active=[1])
        P.ellipse(fx(0),fy(1),19,fill='transparent',stroke=GREEN,sw=3,z=6)
        eq(70,150,500,'x_1+x_2=1.5',27,c=ORANGE);eq(740,150,500,'x_1+2x_2=0.5',27,c=ORANGE)
        eq(50,525,570,'s=1(0)+1(1)-1.5=-0.5',25,c=RED);eq(665,525,590,'s=1(0)+2(1)-0.5=1.5',25,c=GREEN)
        start=len(P.nodes);eq(120,582,1040,'1.5-(-0.5)=2=2(0.5)(1+1)',27,h=45);phase2(start)
    elif kind=='trace':
        badge(tr('程式計算｜資料順序固定、零起點、零分輸出正一','Computed | Fixed sample order, zero start, plus-one output at zero'))
        eq(80,150,1120,r'\eta=0.5,\quad 2\eta=1,\quad\mathbf{w}^{\prime}=\mathbf{w}+y\mathbf{x}',29)
        for j,r in enumerate(SIGNED['updates']):
            col=j if j<3 else j-3;cx=220+col*410 if j<3 else 425+col*410;cy=305 if j<3 else 490
            node(cx,cy,str(j+1),PURPLE,34);eq(cx-145,cy+45,290,v([int(a) for a in r['new']]),27)
            if j<2:arrow(cx+45,cy,cx+360,cy,ORANGE,3)
            if j==3:arrow(cx+45,cy,cx+360,cy,ORANGE,3)
        start=len(P.nodes);P.line([(1110,335),(1110,390),(425,390),(425,450)],ORANGE,3,arrow='end',z=4);phase2(start)
    elif kind=='gate':
        badge(tr('同號與反號，是更新規則不可少的開關','Matching or opposite signs form the essential update gate'))
        node(205,345,'$y,o$',BLUE,68);arrow(285,345,490,345,BLUE)
        P.poly([(590,255),(680,345),(590,435),(500,345)],fill='#fff0d8',stroke=ORANGE,sw=3,z=3);eq(515,305,150,r'o\ne y',26,c=ORANGE)
        P.line([(680,330),(820,230),(1080,230)],GREEN,4,arrow='end',z=4);eq(810,150,410,r'\Delta\mathbf{w}=2\eta y\mathbf{x}',30,c=GREEN)
        P.line([(680,365),(820,475),(1080,475)],MUT,4,arrow='end',z=4);eq(840,490,360,r'\Delta\mathbf{w}=\mathbf0',30)
        text(810,565,385,45,tr('猜對時不更新','No update when correct'),25,MUT,True)
    elif kind=='next':
        badge(tr('代數：反號差兩倍；幾何：錯分損失下降','Algebra: opposite signs differ by two. Geometry: descend the mistake loss.'))
        node(210,255,tr('代數','Algebra'),BLUE,56);node(210,470,tr('幾何','Geometry'),PURPLE,56)
        arrow(280,255,500,340,BLUE);arrow(280,470,500,395,PURPLE)
        eq(450,320,740,r'\mathbf{w}^{\prime}=\mathbf{w}+2\eta y\mathbf{x}',34)
        start=len(P.nodes);label(540,440,590,60,tr('只在猜錯時','Only on a mistake'),28,RED)
        for xx,yy,on in [(950,540,True),(1020,540,False),(950,600,False),(1020,600,True)]:
            if on:triangle(xx,yy,10)
            else:P.ellipse(xx,yy,10,fill='#e5eff8',stroke=BLUE,sw=3,z=5)
        label(1040,530,150,90,'?',48,PURPLE);phase2(start)
    elif kind=='practice1':
        badge(tr('先算是否猜錯，再算更新後的分數','First check the mistake, then compute the updated score'))
        eq(50,160,1180,r'\mathbf{w}=(1,1),\quad w_0=-1.5,\quad\mathbf{x}=(1,0),\quad y=+1,\quad\eta=0.5',26)
        small_input(230,340,[1,0,1]);arrow(335,340,560,340,BLUE);knob(640,340,r'\mathbf{w}^{\prime}',PURPLE,59);arrow(710,340,950,340,PURPLE);lamp(1060,340,None,55)
        eq(150,520,980,r'\mathbf{w}^{\prime}=?,\quad w_0^{\prime}=?,\quad s^{\prime}=?',32)
    elif kind=='practice2':
        badge(tr('問的是無符號幾何距離，不需要乘標籤','Find the unsigned geometric distance; no label is needed'))
        fx,fy=plot(w=(3,4),b=-5,points=False,projection=(0,0));P.ellipse(fx(0),fy(0),10,fill='#e5eff8',stroke=BLUE,sw=3,z=5)
        eq(625,175,580,'3x_1+4x_2-5=0',29);eq(625,300,580,r'\mathbf{w}=(3,4),\quad w_0=-5',28)
        eq(625,430,580,r'\operatorname{distance}=\frac{|w_0|}{\|\mathbf{w}\|}=?',29,h=110)
    elif kind=='practice3':
        badge(tr('正解與輸出各有正、負兩種可能','Both label and output can be plus or minus one'))
        for j,r in enumerate(COMB):
            cx=220+(j%2)*800;cy=270+(j//2)*220
            node(cx-60,cy,'$'+str(r['y'])+'$',BLUE,34);node(cx+55,cy,'$'+str(r['o'])+'$',PURPLE,34);eq(cx-150,cy+65,310,'e=y-o=?',29)
        label(520,310,240,160,'?',80,ORANGE)
    elif kind=='quiz1':
        badge(tr('Signum 感知器的輸出可以取哪些值？','Which values can a sign-activated Perceptron output?'))
        vals=['0','+1',r'\pm1',r'\mathbb R'];texts=[tr('A　只有零','A  Only zero'),tr('B　只有正一','B  Only plus one'),tr('C　正一或負一','C  Plus or minus one'),tr('D　任何實數','D  Any real number')]
        for j,(val,s) in enumerate(zip(vals,texts)):
            cx=190+j*300;node(cx,320,'$'+val+'$',BLUE,56);text(cx-130,455,270,90,s,25,INK,True,align='center')
    elif kind=='quiz2':
        badge(tr('反號時，原更新項可以化簡成哪一項？','On a mistake, which expression equals the original correction?'))
        eq(100,165,1080,r'y,o\in\{-1,+1\},\quad o\ne y,\quad\eta(y-o)\mathbf{x}=?',29)
        for j,val in enumerate([r'\eta y\mathbf{x}',r'2\eta y\mathbf{x}',r'-2\eta\mathbf{x}','0']):
            cx=190+j*300;knob(cx,365,'',PURPLE,45);label(cx-50,270,100,60,chr(65+j),27,PURPLE);eq(cx-130,460,260,val,31)
    elif kind=='quiz3':
        badge(tr('原題釐清｜不同函數，不是相等改寫','Clarified question | Different functions, not an equality'))
        for col,absolute in enumerate((True,False)):
            x=160+col*600;fx,fy=K.axis_plot(x=x,y=195,w=360,h=235,xmin=-1.5,xmax=1.5,ymin=-1.3,ymax=1.7)
            if absolute:P.line([(fx(-1.5),fy(1.5)),(fx(0),fy(0)),(fx(1.5),fy(1.5))],BLUE,5,z=4)
            else:P.line([(fx(-1.3),fy(-1.3)),(fx(1.5),fy(1.5))],PURPLE,5,z=4)
            eq(x+60,440,270,'|s|' if absolute else r'ys\quad(y=+1)',26)
        opts=[tr('A　讓距離變大','A  Make distance larger'),tr('B　絕對值在零點不可微','B  Absolute value has a corner at zero'),tr('C　讓權重成單位向量','C  Make weights a unit vector'),tr('D　讓誤差變小','D  Reduce the error')]
        for j,s in enumerate(opts):text(50+j*305,520,280,100,s,22,INK,True,align='center')
    elif kind=='quiz4':
        badge(tr('原題釐清｜比較代數與錯分損失推導','Clarified question | Compare algebra with the mistake-loss derivation'))
        node(285,290,tr('代數','Algebra'),BLUE,58);node(990,290,tr('梯度','Gradient'),PURPLE,58)
        arrow(355,290,565,350,BLUE);arrow(920,290,710,350,PURPLE);label(565,340,140,110,'?',65,ORANGE)
        opts=[tr('A　對齊學習率後相同','A  Same after matching rates'),tr('B　方向相反','B  Opposite direction'),tr('C　差一個負號','C  Differ by a sign'),tr('D　無法比較','D  Not comparable')]
        for j,s in enumerate(opts):text(50+j*305,500,280,105,s,24,INK,True,align='center')

NOTES={
'tie':('這裡特別檢查邊界：負類原點的分數是零，而符號函數在零點輸出正一，所以猜錯。帶標籤的距離仍是零。因此「猜錯一定負距離」必須加上不在邊界的前提。','Check the boundary carefully. This negative origin has score zero, but the sign function outputs plus one at zero, so it is wrong. Its label-weighted distance is still zero. A strictly negative distance characterizes a mistake only away from the boundary.'),
'distance':('分數是加權和加偏置。把分數除以空間權重的長度，才是沿法向量的有方向距離。幾何權重不含偏置，長度不能是零；不要把更新時的擴增向量範數拿來當這個分母。','The score is the weighted sum plus bias. Divide by the norm of the spatial weights to obtain oriented distance along the normal. Exclude the bias from this norm, and require it to be nonzero. The augmented update norm is not this denominator.'),
'loss':('兩個錯分的正類點各有負零點五的分數，乘上負標籤後各貢獻零點五，總損失是一。這是未正規化分數的損失，不是帶有權重長度分母的幾何距離。負類邊界點雖然猜錯，這一項仍可能為零，不能只靠損失為零停止。','The two misclassified positive points each have score minus one half and contribute one half to the loss, totaling one. This uses unnormalized scores, not geometric distances with a weight-norm denominator. A negative boundary mistake can contribute zero, so zero loss alone is not the stopping test.'),
'gradient':('在不落於決策邊界的局部區域，錯分集合固定，直接對線性分數求導。到了零分邊界，損失有折角，梯度不一定唯一；標準感知器按符號輸出約定選擇一個有效次梯度。不要宣稱整個損失處處可微。','Away from decision boundaries, the mistake set is locally fixed and we differentiate the linear scores. At a zero-score boundary the loss has a corner and may lack a unique gradient. The Perceptron convention selects a valid subgradient; the full loss is not differentiable everywhere.'),
'numeric_update':('這一筆是零、一，擴增後最後再補一。舊完整權重是一、一、負一點五，正確標籤正一，學習率零點五。第一路輸入零所以不改；第二路和偏置都增加一。','The input is zero, one, augmented with a final one. The old complete weights are one, one, minus 1.5, the label is plus one, and the rate is 0.5. The first input is zero, so that weight stays; the second weight and bias each increase by one.'),
'trace':('用固定順序從全零開始，符號版的二倍學習率等於一，剛好和上一單元的零一版每次修正一致。程式逐一比對全部十六次訪問的新權重，不只比對最後結果。五次更新之後，第四輪完整檢查沒有錯誤才停止。','With fixed order and zero initialization, twice the signed-label rate equals one, matching every binary-label correction from KP02. The program compares the new weights at all sixteen visits, not just the final result. Five updates are followed by a fourth pass confirming no mistakes.'),
'quiz3':('原題把絕對值與帶標籤分數稱作相等改寫，這並不成立。因此改問為什麼採用帶標籤的分數表達誤差方向。這兩個小圖是不同函數，題目選項不提供答案。','The original question presents an absolute value and a label-weighted score as an equal rewrite, which is not valid. We instead ask why the score representation is useful for class direction. The two plots represent different functions; no answer is supplied.'),
'quiz4':('這裡比較的是已經推導的代數路徑與錯分損失的梯度路徑。原題把最大化所有正確點距離當成同一個目標，敘述過廣，因此釐清題幹；先讓學生比對兩個學習率的定義。','Compare the established algebraic update and the gradient of the mistake loss. The original wording was too broad in treating maximization of all correct distances as the same objective. The clarified question asks students to compare the two learning-rate definitions.')}

def build(lang):
    global LANG,P,SECOND
    LANG=lang;K.LANG=lang;title=tr('KP03 正負標籤與更新推導','KP03 Signed labels and the update rule')
    d=K.make(title,title,'kp03');sid='scene-kp03';sc=d['scenes'][sid];d['nodes']['frame-kp03']['name']=title
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['metadata'].update(createdAt=now,updatedAt=now)
    manifest=[];warnings=[]
    for i,(kind,zh,en,tz,te,sec,brief) in enumerate(PAGES):
        P=K.Page(prefix=f'{lang}-p{i+1:02d}');K.P=P;SECOND=[]
        P.rect(0,0,1280,720,fill='#ffffff',z=0);P.rect(0,0,1280,86,fill='#edf4fa',z=1);P.rect(0,0,13,86,fill=BLUE,z=2)
        en=en.replace('Self-check: why use a label-weighted score rather than an absolute value?','Self-check: why use a label-weighted score?')
        text(48,20,1160,60,tr(zh,en),35 if lang=='zh' else 29,INK,True,name='page-title');text(1130,684,100,28,f'{i+1:02d} / {len(PAGES)}',17,MUT)
        start=len(P.nodes);draw(kind)
        for n in P.nodes[start:]:
            n['transform']['x']=20+n['transform']['x']*.8
            n['transform']['y']=170+(n['transform']['y']-150)*.8
            n['width']*=.8;n['height']*=.8
            if n['type']=='text':n['style']['fontSize']=max(16,n['style']['fontSize']*.8)
            for key in ('radius','borderWidth'):
                if key in n['style']:n['style'][key]*=.8
        actorstart=len(P.nodes);visual_story.draw(K,kind);phase2(actorstart)
        main_ids=[n['id'] for n in P.nodes[start:]];groups=[[k for k in main_ids if k not in SECOND]]+([SECOND] if SECOND else [])
        ft=len(P.nodes);P.rect(48,640,1184,56,fill='#eaf5ef',radius=12,z=8);text(70,650,1130,42,tr(tz,te),26 if lang=='zh' else 24,GREEN,True,name='takeaway');groups.append([n['id'] for n in P.nodes[ft:]])
        P.check()
        if P.warnings:warnings.append((i+1,P.warnings))
        for n in P.nodes:
            n['sceneId']=sid;n['transform']['x']+=i*1440;n['initiallyHidden']=n['id'] in main_ids+groups[-1]
            if n['type']=='text':n['style']['fontFamily']='Noto Sans CJK TC' if lang=='zh' else 'Arial'
            d['nodes'][n['id']]=n;sc['nodeIds'].append(n['id'])
        first_step=len(d['steps'])
        for beat,ids in enumerate(groups):
            note=tr(*NOTES[kind]) if beat==0 and kind in NOTES else tr(tz,te)
            if beat==0 and kind not in NOTES:note=tr('先看圖中的條件與對象。'+tz,'Identify the conditions and objects in the figure. '+te)
            if kind=='numeric_update' and beat==1:note=tr('修正係數是二乘零點五乘正一，也就是一。因此三路修正量是零、一、一，加回舊完整權重後，得到一、二、負零點五。','The correction coefficient is two times 0.5 times plus one, which is one. The three corrections are zero, one, one. Adding them yields one, two, minus 0.5.')
            order=len(d['steps']);d['steps'].append(dict(id=f'{lang}-p{i+1}-beat{beat+1}',order=order,label=tr(zh,en)+' · '+str(beat+1),durationMs=450,holdMs=500,easing='smooth',target=dict(kind='scene-point',sceneId=sid,point=dict(x=i*1440+640,y=360),zoom=1,referenceViewport=dict(width=1280,height=720)),reveal=dict(timing='after-camera',enter=[dict(nodeId=k,effect='fade',delayMs=0,durationMs=250) for k in ids],exitNodeIds=[]),speakerNotes=note))
        manifest.append(dict(page=i+1,kind=kind,title=tr(zh,en),source_section=sec,first_step=first_step,final_step=len(d['steps'])-1,beats=len(groups),extension=i>=19))
    out=ROOT/'output';out.mkdir(exist_ok=True);stem='lec02-kp03-signum-update-derivations'+('-en' if lang=='en' else '')
    (out/(stem+'.jbd.json')).write_text(json.dumps(d,ensure_ascii=False,indent=1));(out/(lang+'-pages.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=1))
    print(lang,len(manifest),'pages;',len(d['steps']),'steps;',len(d['nodes']),'native nodes')
    if warnings:raise RuntimeError(warnings)
if __name__=='__main__':save();build('zh');build('en')
