from pathlib import Path
import json,datetime,math
import pagekit as K
import mascot
from specs import PAGES,MATH
from kp08_nums import *
ROOT=Path(__file__).parent;P=None;LANG='zh';SECOND=[];TEMP=[]
INK,MUT,BLUE,PURPLE,GREEN,ORANGE,GREY,RED=K.INK,K.MUT,K.BLUE,K.PURPLE,K.GREEN,K.ORANGE,K.GREY,K.RED
text,label=K.text,K.label
def eq(x,y,w,s,size=27,h=62,c=INK):
 if any(t in s for t in [r'\frac',r'\sum',r'\prod',r'\sqrt']):h=max(h,110)
 return K.eq(x,y,w,s,size,h,c)
tr=lambda a,b:a if LANG=='zh' else b

def phase2(start):SECOND.extend(n['id'] for n in P.nodes[start:])
def badge(s):text(50,110,950,46,s,21,MUT)
def shape(kind,x,y,w,h,c=BLUE,fill=None,z=3,**style):
 n=P.rect(x,y,w,h,fill=fill or '#edf5fc',stroke=c,sw=2,radius=12 if kind=='rectangle' else 0,z=z,**style);n['shapeKind']=kind;return n

def box(x,y,w,h,s,c=BLUE,kind='rectangle',fs=25):
 n=shape(kind,x,y,w,h,c);lh=h-16;label(x+8,y+15,w-16,lh,s,fs,c);return n

def conn(a,b,c=BLUE,sh=4,th=12,curve=False,lab=None,dash=False):
 n=P._add('connector',0,0,1,1,{'fill':'transparent','stroke':c,'textColor':INK,'fontSize':20,'borderWidth':3,'borderStyle':'dashed' if dash else 'solid','radius':0},z=25,name='native-flow')
 n['connector']={'sourceNodeId':a['id'],'sourceHook':sh,'targetNodeId':b['id'],'targetHook':th,'curve':curve,'direction':'forward','arrowSize':'medium'}
 if lab:n['connector']['label']={'text':lab,'t':.5,'offset':'above'}
 return n

def star(x,y,r=16,c=PURPLE):return shape('star',x-r,y-r,2*r,2*r,c,c,z=8)
def point(x,y,lab,r=10,c=None):
 c=c or (GREEN if lab else BLUE)
 return shape('triangle' if lab else 'ellipse',x-r,y-r,2*r,2*r,c,c if lab else '#edf5fc',z=6)
def tick(x,y,ok=True):
 if ok:P.line([(x-8,y),(x-2,y+7),(x+11,y-9)],GREEN,3,z=8)
 else:P.line([(x-7,y-7),(x+7,y+7)],RED,3,z=8);P.line([(x-7,y+7),(x+7,y-7)],RED,3,z=8)

def stamp(x,y,s):
 n=P.rect(x,y,190,50,fill='#fff2d6',stroke=ORANGE,sw=2,radius=8,z=9);n['transform']['rotation']=-7
 t=label(x,y+4,190,42,s,23,ORANGE);t['transform']['rotation']=-7

def curve(fn,x=105,y=245,w=440,h=295,xmin=-3,xmax=3,ymin=0,ymax=1.05,c=BLUE,n=300,dash=False,axes=True):
 fx=lambda v:x+(v-xmin)/(xmax-xmin)*w;fy=lambda v:y+h-(v-ymin)/(ymax-ymin)*h
 if axes:
  P.rect(x,y,w,h,fill='#eff5fa',stroke=GREY,sw=1,fillOpacity=.55,z=1)
  if ymin<=0<=ymax:P.line([(x,fy(0)),(x+w,fy(0))],GREY,2,z=2)
  if xmin<0<xmax:P.line([(fx(0),y),(fx(0),y+h)],GREY,2,z=2)
  for v in sorted(set([xmin,xmax]+([0] if xmin<0<xmax else []))):eq(fx(v)-30,y+h+4,60,f'{v:g}',18,h=35)
  for v in [ymin,ymax]:eq(x-65,fy(v)-15,55,f'{v:g}',18,h=35)
 pts=[]
 for k in range(n):
  a=xmin+(xmax-xmin)*k/(n-1);v=fn(a);assert ymin-1e-6<=v<=ymax+1e-6,(a,v,ymin,ymax);pts.append((fx(a),fy(v)))
 for start in range(0,n-1,180):P.line(pts[start:start+181],c,4,dash=dash,z=4)
 return fx,fy

def ordots(x=150,y=245,size=320,boundary=None):
 lo=-.25;hi=1.25;fx=lambda v:x+(v-lo)/(hi-lo)*size;fy=lambda v:y+size-(v-lo)/(hi-lo)*size
 P.rect(x,y,size,size,fill='#f1f5fb',fillOpacity=.6,stroke=GREY,sw=1,z=1)
 P.line([(x,fy(0)),(x+size,fy(0))],GREY,2,z=2);P.line([(fx(0),y),(fx(0),y+size)],GREY,2,z=2)
 if boundary:
  for total,c,dash in boundary:
   a=max(lo,total-hi);b=min(hi,total-lo);P.line([(fx(a),fy(total-a)),(fx(b),fy(total-b))],c,4,dash=dash,z=3)
 for j,(a,b) in enumerate(OR_X):
  point(fx(a),fy(b),bool(OR_Y[j]),11,c=BLUE if OR_Y[j] else PURPLE)
  eq(fx(a)-50,fy(b)+(14 if b==0 else -48),100,f'({a:g},{b:g})',18,h=35)
 eq(x+size-25,y+size+30,70,'x_1',22,h=40);eq(x-55,y-35,70,'x_2',22,h=40)
 return fx,fy

def logdata(x=120,y=260,w=780,h=250):
 fx=lambda v:x+(v+.5)/4*w;fy=lambda v:y+h-(v+.08)/1.16*h
 P.rect(x,y,w,h,fill='#f1f5fb',stroke=GREY,sw=1,fillOpacity=.55,z=1)
 P.line([(x,fy(.5)),(x+w,fy(.5))],ORANGE,2,dash=True,z=2)
 for a,b in zip(LX,LY):
  point(fx(a),fy(b),bool(b),11,c=BLUE if b else PURPLE);eq(fx(a)-30,y+h+8,60,f'{a:g}',20,h=35)
 return fx,fy

def draw(kind):
 if kind=='cover':
  badge(tr('承接 KP07：選好損失後，如何更新單一神經元？','From KP07: with a loss chosen, how do we update one unit?'))
  a=box(65,315,160,100,tr('輸入','Input'),BLUE,'ellipse',31)
  for j,(s,c,t) in enumerate([('ADALINE',GREEN,r'$\frac12\sum(y-\hat y)^2$'),('Logistic',ORANGE,r'$-\ell$')]):
   yy=200+j*250;b=box(390,yy,230,105,s,c,'ellipse',30);e=box(795,yy,205,105,t,c,'ellipse',30);conn(a,b,c,sh=2 if j==0 else 6,th=12);conn(b,e,c)
  stamp(725,580,tr('連續目標','CONTINUOUS'))
 elif kind=='train':
  badge(tr('概念示意｜訓練損失直接看連續值，不先做硬分類','Concept sketch | Train on the continuous value before a hard decision'))
  a=box(60,315,160,105,r'$\mathbf x$',BLUE,'ellipse',34);b=box(390,315,220,105,r'$\hat y$',PURPLE,'ellipse',36);conn(a,b)
  c=box(790,200,210,110,'LMS',GREEN,'diamond',32);conn(b,c,GREEN,sh=2,th=12)
  d=box(790,455,210,110,tr('門檻','Threshold'),ORANGE,'diamond',29);conn(b,d,ORANGE,sh=6,th=12)
  eq(80,165,620,MATH[kind][0],31,h=110);label(665,185,125,45,tr('訓練','Train'),25,GREEN);label(665,445,125,45,tr('推論','Infer'),25,ORANGE)
  st=len(P.nodes);eq(275,525,460,MATH[kind][1],28,h=85);phase2(st)
 elif kind=='lms':
  badge(tr('記號釐清｜以 $E$ 表損失，避免與 KP07 殘差混用','Notation | E denotes loss, distinct from the residual in KP07'))
  curve(FUN['sq'],y=280,w=350,h=260,ymax=5,c=GREEN);eq(120,195,380,MATH[kind][0],28,h=110,c=GREEN)
  eq(525,190,490,MATH[kind][1],27,h=110)
  st=len(P.nodes);eq(525,355,490,MATH[kind][2],27,h=110,c=PURPLE);eq(525,510,490,MATH[kind][3],29,h=95,c=GREEN);phase2(st)
 elif kind=='ordata':
  badge(tr('原始四筆 OR 資料｜正類三角、負類圓點','Source OR data | Positive triangles, negative circles'))
  ordots()
  eq(510,190,500,MATH[kind][0],27,h=100)
  eq(515,330,495,MATH[kind][1],25,h=110)
  st=len(P.nodes);eq(515,490,495,MATH[kind][2],26,h=110,c=GREEN);eq(515,585,495,r'E_0=\frac12(0^2+1^2+1^2+1^2)=1.5',23,h=110,c=ORANGE);phase2(st)
 elif kind=='orgrad':
  badge(tr('逐筆配對｜左側是四筆資料；右側算三個參數的梯度','Match samples on the left; calculate three parameter gradients on the right'))
  eq(55,205,65,'r_i',22,h=45,c=BLUE)
  for j in range(4):box(145+j*75,195,55,65,'$'+str(int(OR_Y[j]))+'$',BLUE,'ellipse',25)
  lastcells=[]
  for k in range(3):
   yy=330+k*100;eq(55,yy,80,['x_1','x_2','1'][k],22,h=45,c=PURPLE);last=None
   for j in range(4):
    v=[OR_X[j,0],OR_X[j,1],1][k];last=shape('rectangle',145+j*75,yy,50,45,PURPLE);eq(145+j*75,yy,50,str(int(v)),23,h=45,c=PURPLE)
   lastcells.append(last)
  st=len(P.nodes)
  for k,f in enumerate(MATH[kind]):
   result=eq(470,305+k*100,540,f,24,h=90,c=[GREEN,PURPLE,ORANGE][k]);conn(lastcells[k],result,[GREEN,PURPLE,ORANGE][k])
  phase2(st)
 elif kind=='orupdate':
  badge(tr('批次梯度下降｜同時更新兩個權重與一個偏置','Batch gradient descent | Update two weights and one bias together'))
  for j,(name,g,v) in enumerate(zip(['w_1','w_2','w_0'],[-2,-2,-3],[.2,.2,.3])):
   yy=200+j*125;a=box(80,yy,180,80,'$'+name+'=0$',BLUE,'ellipse',29);b=box(775,yy,200,80,'$'+name+'='+str(v)+'$',GREEN,'ellipse',29);conn(a,b,GREEN)
   eq(300,yy+37,400,rf'0-0.1({g})={v}',30,h=65,c=GREEN)
  eq(280,560,700,MATH[kind][0],26,h=65,c=MUT)
 elif kind=='orloss':
  badge(tr('更新後重新前饋｜使用新參數，不沿用舊預測','Forward again | Use new parameters rather than stale predictions'))
  eq(65,165,945,MATH[kind][0],27,h=110)
  for j,(p,y) in enumerate(zip(OR_TRACE[1]['prediction'],OR_Y)):
   xx=140+j*220;hh=210*p;P.rect(xx,520-hh,60,hh,fill=BLUE,z=3);P.line([(xx-15,520-y*210),(xx+75,520-y*210)],PURPLE,3,dash=True,z=4);eq(xx-20,535,110,f'{p:.1f}',25,h=45,c=BLUE)
  st=len(P.nodes);eq(70,580,940,MATH[kind][2],26,h=110,c=GREEN);phase2(st)
 elif kind=='orlimit':
  badge(tr('閉式最小平方解｜本例以零點五為分類門檻','Closed-form least squares | Classification threshold 0.5 in this example'))
  eq(70,165,940,MATH[kind][0],31,h=95)
  for j,(p,y) in enumerate(zip(YSTAR,OR_Y)):
   xx=130+j*225;P.rect(xx,535-p*190,70,p*190,fill=BLUE,z=3);P.line([(xx-15,440),(xx+85,440)],ORANGE,3,dash=True,z=4);eq(xx-35,550,140,f'{p:.2f}',25,h=45,c=BLUE)
   shape('ellipse',xx+18,270,35,35,GREEN);eq(xx+18,270,35,str(int(y)),22,h=40,c=GREEN)
  st=len(P.nodes);eq(65,590,945,MATH[kind][2],25,h=110,c=GREEN);phase2(st)
 elif kind=='orboundary':
  badge(tr('精確幾何｜實線是此 ADALINE 解，虛線是 KP02 的特例','Exact geometry | Solid: this ADALINE fit; dashed: the KP02 example'))
  ordots(x=150,y=230,size=340,boundary=[(.5,ORANGE,False),(1,BLUE,True)])
  eq(565,230,440,r'x_1+x_2=1',31,h=80,c=BLUE)
  st=len(P.nodes);eq(545,375,470,r'0.5x_1+0.5x_2+0.25=0.5',27,h=110,c=ORANGE);eq(570,520,430,r'x_1+x_2=0.5',31,h=80,c=ORANGE);phase2(st)
 elif kind=='ortrace':
  badge(tr('原程式重現｜四筆全批次、初始零、學習率零點一','Source reproduction | Full batch of four; zero start; learning rate 0.1'))
  x,y,w,h=115,255,825,275;fx=lambda k:x+k/12*w;fy=lambda v:y+h-v/1.6*h
  P.rect(x,y,w,h,fill='#eff5fa',stroke=GREY,sw=1,fillOpacity=.55,z=1)
  P.line([(x,fy(.125)),(x+w,fy(.125))],GREEN,3,dash=True,z=3)
  pts=[(fx(r['k']),fy(r['loss'])) for r in OR_TRACE];P.line(pts,BLUE,4,z=4)
  for a,b in pts:P.ellipse(a,b,5,fill=BLUE,z=5)
  for k in [0,1,4,8,12]:eq(fx(k)-30,535,60,str(k),19,h=35)
  for k in [0,1]:eq(fx(k)+12,fy(OR_TRACE[k]['loss'])-35,140,str(OR_TRACE[k]['loss']),23,h=40,c=BLUE)
  eq(640,450,300,r'E^\ast=0.125',26,h=65,c=GREEN)
  eq(70,160,940,MATH[kind][0],27,h=110);eq(75,575,940,MATH[kind][1],25,h=110)
 elif kind=='madaline':
  badge(tr('沿革依講義｜示意圖只標訓練範圍，不是完整電路','Handout chronology | Sketches show training scope, not full circuits'))
  anchors=[]
  for j,(year,name,zh,en) in enumerate([(1962,'MRI','輸入到隱藏','Input to hidden'),(1988,'MRII','兩段連接','Both connections'),(1990,'MRIII','更深多層','Deeper layers')]):
   xx=80+j*330;a=shape('ellipse',xx+90,210,40,40,K.PAL[j],K.PAL[j]);anchors.append(a);label(xx,160,240,45,str(year),29,K.PAL[j]);label(xx,270,240,55,name,32,K.PAL[j])
   ns=[shape('ellipse',xx+35+k*70,390,27,27,K.PAL[j]) for k in range(3)]
   for k in range(2):conn(ns[k],ns[k+1],K.PAL[j] if j>0 or k==0 else GREY,dash=j==0 and k==1)
   label(xx-15,465,280,80,tr(zh,en),25,K.PAL[j])
  for a,b in zip(anchors,anchors[1:]):conn(a,b,ORANGE)
 elif kind=='logistic':
  badge(tr(r'判別式模型｜直接描述 $P(y\mid\mathbf x)$，不是 $P(\mathbf x\mid y)$',r'Discriminative model | Model $P(y\mid\mathbf x)$, not $P(\mathbf x\mid y)$'))
  fx,fy=curve(FUN['sigmoid']);P.line([(105,fy(.5)),(545,fy(.5))],ORANGE,2,dash=True,z=3);star(fx(0),fy(.5),11,ORANGE)
  eq(110,185,430,r'p=\sigma(s)',28,h=65,c=BLUE)
  eq(570,170,440,MATH[kind][0],24,h=110)
  eq(585,325,425,MATH[kind][1],29,h=110,c=BLUE)
  st=len(P.nodes);eq(580,480,430,MATH[kind][2],26,h=90,c=PURPLE);eq(590,560,410,MATH[kind][3],26,h=70,c=GREEN);phase2(st)
 elif kind=='likelihood':
  badge(tr('條件獨立的 Bernoulli 觀測｜固定輸入，似然視為參數的函數','Conditionally independent Bernoulli observations | Inputs are fixed'))
  a=box(80,210,185,100,r'$p_i$',BLUE,'ellipse',32);b=box(80,425,185,100,r'$1-p_i$',PURPLE,'ellipse',32);c=box(385,315,125,115,r'$\ln$',GREEN,'diamond',32);conn(a,c);conn(b,c,PURPLE)
  eq(550,165,460,MATH[kind][0],24,h=110)
  eq(540,315,475,MATH[kind][1],25,h=110)
  st=len(P.nodes);eq(535,455,480,MATH[kind][2],25,h=110,c=GREEN);eq(90,550,380,MATH[kind][3],32,h=65,c=GREEN);phase2(st)
 elif kind=='compare':
  badge(tr('補充釐清｜左看負損失梯度，右看正似然梯度','Clarification | Compare the negative loss gradient and positive likelihood gradient'))
  for j,(name,fn,c) in enumerate([('ADALINE',FUN['convex'],GREEN),('Logistic',FUN['concave'],ORANGE)]):
   xx=105+j*500;curve(fn,x=xx,y=255,w=350,h=230,xmin=-1,xmax=1,ymin=0,ymax=1.1,c=c);label(xx,175,350,50,name,29,c)
  st=len(P.nodes);eq(60,535,480,MATH[kind][0],26,h=110,c=GREEN);eq(540,535,475,MATH[kind][1],26,h=110,c=ORANGE);phase2(st)
 elif kind=='hessian':
  badge(tr('欄向量約定｜第一列全為一，偏置已併入參數','Column convention | First row is all ones; bias is included'))
  for col in range(4):
   for row in range(3):P.rect(140+col*65,275+row*65,52,52,fill='#dcecf8' if row else '#e2d8f2',stroke=BLUE,sw=1,z=3)
  eq(90,175,410,MATH[kind][0],27,h=80)
  eq(85,505,430,r'\mathbf g=\mathbf X(\mathbf y-\mathbf p)',29,h=100,c=BLUE)
  eq(535,185,480,MATH[kind][1],26,h=110)
  st=len(P.nodes);eq(535,350,480,MATH[kind][3],30,h=110,c=ORANGE);eq(575,510,400,r'\mathbf H\preceq0',31,h=80,c=ORANGE);phase2(st)
 elif kind=='newton':
  badge(tr('前提：所用 Hessian 可逆；一般式同時用於尋找極小或極大駐點','Assume an invertible Hessian; the generic step targets either type of stationary point'))
  eq(200,170,810,MATH[kind][0],34,h=110)
  eq(200,330,810,MATH[kind][1],31,h=110,c=PURPLE)
  st=len(P.nodes);eq(190,490,820,MATH[kind][2],31,h=110,c=GREEN);phase2(st)
  ns=[box(75,185+j*160,70,70,str(j+1),[BLUE,PURPLE,GREEN][j],'ellipse',28) for j in range(3)];conn(ns[0],ns[1],PURPLE,sh=8,th=0);conn(ns[1],ns[2],GREEN,sh=8,th=0)
 elif kind=='newtondata':
  badge(tr('原資料｜四筆一維資料交錯排列，沒有改標籤','Source data | Four interleaved labels; no labels changed'))
  logdata(x=115,y=200,w=825,h=180)
  eq(60,410,520,MATH[kind][0],31,h=130)
  st=len(P.nodes);eq(615,410,385,MATH[kind][1],28,h=90);eq(90,560,900,MATH[kind][2],27,h=85,c=BLUE);phase2(st)
 elif kind=='newtongrad':
  badge(tr(r'起點 $\boldsymbol\beta=\mathbf0$｜四個預測機率都是零點五',r'Start at $\boldsymbol\beta=\mathbf0$ | All four probabilities are 0.5'))
  eq(60,160,950,MATH[kind][0],28,h=110)
  P.line([(100,370),(500,370)],GREY,2,z=2)
  for j,v in enumerate([-.5,.5,-.5,.5]):P.rect(130+j*95,370-90 if v>0 else 370,48,90,fill=BLUE if v>0 else PURPLE,z=3)
  eq(525,260,490,MATH[kind][1],31,h=130,c=BLUE)
  st=len(P.nodes);eq(90,495,910,MATH[kind][2],31,h=130,c=GREEN);phase2(st)
 elif kind=='newtonstep':
  badge(tr('解線性方程更實用；逆矩陣供手算，左圖顯示參數移動','Solve the linear system in practice; the inverse permits hand checking'))
  xx,yy,ww,hh=115,280,305,265
  fx=lambda v:xx+(v+1.5)/2*ww;fy=lambda v:yy+hh-(v+.2)/1.3*hh
  P.rect(xx,yy,ww,hh,fill='#eff5fa',stroke=GREY,sw=1,fillOpacity=.55,z=1)
  P.line([(xx,fy(0)),(xx+ww,fy(0))],GREY,2,z=2);P.line([(fx(0),yy),(fx(0),yy+hh)],GREY,2,z=2)
  a=shape('ellipse',fx(0)-7,fy(0)-7,14,14,BLUE,BLUE);b=shape('ellipse',fx(-1.2)-7,fy(.8)-7,14,14,GREEN,GREEN);conn(a,b,GREEN,sh=14,th=6)
  P.line([(fx(-1.2),fy(0)),(fx(-1.2),fy(.8)),(fx(0),fy(.8))],GREEN,2,dash=True,z=3)
  eq(xx+ww-20,fy(0)+10,70,r'\beta_0',22,h=40);eq(fx(0)-30,yy-40,70,r'\beta_1',22,h=40)
  eq(xx-20,fy(.8)-20,60,'0.8',20,h=40,c=GREEN);eq(fx(-1.2)-55,yy+hh+5,110,'-1.2',20,h=45,c=GREEN)
  eq(505,165,500,MATH[kind][0],27,h=100)
  eq(505,305,500,MATH[kind][1],29,h=130,c=PURPLE)
  st=len(P.nodes);eq(485,475,525,MATH[kind][2],28,h=130,c=GREEN);phase2(st)
 elif kind=='newtontrace':
  badge(tr('原程式重現｜同一份四筆資料；曲線網格三百點','Source reproduction | Same four observations; 300 curve samples'))
  fx,fy=logdata(x=105,y=245,w=480,h=300)
  for k,c in enumerate([GREY,BLUE,ORANGE]):
   st=len(P.nodes);bb=np.array(NEWTON[k]['beta']);curve(lambda a:float(sigmoid(bb[0]+bb[1]*a)),x=105,y=245,w=480,h=300,xmin=-.5,xmax=3.5,ymin=-.08,ymax=1.08,c=c,n=300,dash=k==0,axes=False)
   if k>0:phase2(st)
  for k in range(3):
   bb=NEWTON[k]['beta'];yy=190+k*135;eq(605,yy,400,rf'\boldsymbol\beta_{k}\approx({bb[0]:.3f},{bb[1]:.3f})',25,h=65,c=[GREY,BLUE,ORANGE][k]);eq(605,yy+56,400,rf'\ell_{k}\approx{NEWTON[k]["loglikelihood"]:.4f}',26,h=65,c=[GREY,BLUE,ORANGE][k])
  eq(90,165,500,MATH[kind][0],24,h=110,c=BLUE);eq(65,575,945,r'\ell_k=\sum_{i=1}^{4}[y_i\ln p_k(x_i)+(1-y_i)\ln(1-p_k(x_i))]',25,h=110,c=ORANGE)
 elif kind=='outcome':
  badge(tr('同一批訓練資料｜比較最大似然與 0–1 錯誤數兩個目標','Same training data | Likelihood and 0–1 error count are different objectives'))
  eq(65,160,945,MATH[kind][0],29,h=85)
  fx,fy=logdata(x=120,y=275,w=820,h=220)
  for j,(x,p,y) in enumerate(zip(LX,PSTAR,LY)):
   P.ellipse(fx(x),fy(p),9,fill='white',stroke=GREEN,sw=3,z=5);P.line([(fx(x),fy(y)),(fx(x),fy(p))],GREY,2,dash=True,z=3);eq(fx(x)-55,530,110,f'{p:.3f}',23,h=40,c=GREEN)
  st=len(P.nodes)
  for cutoff,c in [(.5,ORANGE),(1.5,GREEN)]:
   P.line([(fx(cutoff),275),(fx(cutoff),495)],c,2,dash=True,z=3);eq(fx(cutoff)-90,225,180,rf'x_0={cutoff}',23,h=45,c=c)
  eq(60,565,950,r'\mathrm{MLE}:\ (0,0,1,1),\quad 2/4',24,h=40,c=GREEN)
  eq(60,603,950,r'x_0=0.5:\ (0,1,1,1),\quad 1/4',24,h=35,c=ORANGE);phase2(st)
 elif kind=='limits':
  badge(tr('選讀補充｜凹性不等於牛頓法每一步都保證改善','Optional clarification | Concavity does not guarantee improvement of every Newton step'))
  for j,k in enumerate([1,3,8]):
   curve(lambda a:float(sigmoid(k*a)),x=110,y=250,w=480,h=300,c=[GREY,BLUE,ORANGE][j],axes=j==0)
  eq(130,180,450,r'p=\sigma(ks),\quad k=1,3,8',28,h=70)
  eq(625,215,375,r'\mathbf H\preceq0',34,h=85,c=ORANGE)
  st=len(P.nodes);eq(615,375,395,r'\|\boldsymbol\beta\|\to\infty',32,h=100,c=RED);label(620,530,380,75,tr('可分時：留意正則化','Separable data: consider regularization'),24,GREEN);phase2(st)
 elif kind=='next':
  badge(tr('概念銜接｜固定基函數是下一節的特定設計選擇','Concept bridge | Fixed basis functions are a specific design choice next'))
  a=box(70,315,220,110,tr('第一層怎麼做','Design the first layer'),BLUE,'diamond',27)
  for j,(name,c) in enumerate([('RBF',GREEN),('SOM',PURPLE)]):
   b=box(735,200+j*265,240,95,name,c,'ellipse',34);conn(a,b,c,sh=2 if j==0 else 6,th=12)
   if j==0:
    for x in [430,500,570]:P.ellipse(x,245,30,fill='transparent',stroke=c,sw=3,z=4)
   else:
    for k in range(3):P.line([(420,445+k*35),(590,445+k*35)],c,2,z=4);P.line([(420+k*85,445),(420+k*85,515)],c,2,z=4)
 elif kind=='practice1':
  badge(tr('練習｜本頁與講者備註只給提示','Practice | This page and its notes provide hints only'))
  ordots();eq(530,195,480,MATH[kind][0],29,h=110)
  st=len(P.nodes);eq(540,410,460,MATH[kind][1],40,h=110,c=GREEN);phase2(st)
 elif kind=='practice2':
  badge(tr('練習｜不顯示中間分數或最終機率','Practice | Neither the intermediate score nor the final probability is supplied'))
  eq(70,170,940,MATH[kind][0],31,h=100)
  ns=[box(80+j*345,350,220,105,s,K.PAL[j],'ellipse',32) for j,s in enumerate(['$x=2$','$s=?$','$p=?$'])];conn(ns[0],ns[1]);conn(ns[1],ns[2],PURPLE)
 elif kind=='practice3':
  badge(tr('練習｜分清模型輸出、分類規則與所用門檻','Practice | Distinguish model output, class rule and chosen threshold'))
  a=box(75,315,210,110,tr('連續輸出','Continuous score'),BLUE,'ellipse',28);b=box(435,310,175,125,'?',ORANGE,'diamond',40);conn(a,b)
  for j in [0,1]:c=box(820,210+j*255,130,100,'$'+str(j)+'$',GREEN,'ellipse',34);conn(b,c,GREEN,sh=2 if j==0 else 6,th=12)
 elif kind.startswith('quiz'):draw_quiz(kind)
 else:raise ValueError(kind)

def draw_quiz(kind):
 q=int(kind[4:]);opts={
 1:[('硬門檻後的二元值','Hard-threshold output'),('活化前的加權和','Pre-activation weighted sum'),('Sigmoid 機率','Sigmoid probability'),('權重的絕對值','Absolute weight values')],
 2:[('交叉熵','Cross-entropy'),('LMS 二次損失','LMS quadratic loss'),('Hinge','Hinge'),('KL 散度','KL divergence')],
 3:[(r'$w_j+\eta\frac{\partial E}{\partial w_j}$',)*2,(r'$w_j-\eta\frac{\partial E}{\partial w_j}$',)*2,(r'$\eta\frac{\partial E}{\partial w_j}$',)*2,(r'$w_j/\eta$',)*2],
 4:[('MRI（1962）：輸入至隱藏\nMRII（1988）：兩段連接','MRI (1962): input–hidden\nMRII (1988): both links'),('MRI 已支援所有連接','MRI updates all connections'),('MRIII 是最早的規則','MRIII was the first rule'),('MRII 不需要隱藏層','MRII needs no hidden layer')],
 5:[(r'$P(\mathbf x\mid y)$',)*2,(r'$P(y\mid\mathbf x)$',)*2,(r'$P(y)$',)*2,(r'$P(\mathbf x,y)$',)*2],
 6:[('人口成長（講義列\nVerhulst，1845）','Population growth\n(Verhulst, 1845)'),('影像分類','Image classification'),('矩陣分解','Matrix factorization'),('最短路徑','Shortest paths')],
 7:[(r'$\boldsymbol\beta+(\mathbf X\mathbf W\mathbf X^\top)^{-1}\mathbf X(\mathbf y-\mathbf p)$',)*2,(r'$\boldsymbol\beta+\mathbf X\mathbf W\mathbf X^\top$',)*2,(r'$\boldsymbol\beta-\mathbf W^{-1}\mathbf y$',)*2,(r'$\boldsymbol\beta+(\mathbf y-\mathbf p)$',)*2],
 8:[('梯度是否為零','Is the gradient zero?'),('機率是否至少零點五','Is the probability at least 0.5?'),('取權重總和','Sum the weights'),('隨機猜測','Guess randomly')],
 }
 badge(tr('八題依講義保留選項；四項同樣式，不附答案','Eight source checks; equal styling for all choices; no answers'))
 if q==7:
  for j,(zh,en) in enumerate(opts[q]):
   yy=185+j*108;shape('ellipse',70,yy+5,50,50,PURPLE);label(70,yy+9,50,40,chr(65+j),25,PURPLE);eq(140,yy,850,tr(zh,en).strip('$'),27,h=100,c=INK)
  return
 for j,(zh,en) in enumerate(opts[q]):
  xx=70+j%2*500;yy=205+j//2*210;shape('ellipse',xx,yy,50,50,PURPLE);label(xx,yy+4,50,40,chr(65+j),25,PURPLE)
  if q in [3,5]:eq(xx+65,yy,350,tr(zh,en).strip('$'),28 if q==5 else 25,h=110)
  else:text(xx+65,yy,350,110,tr(zh,en),24 if LANG=='zh' else 22,INK,True)
  xp=xx+110;yp=yy+135
  P.line([(xp,yp+35),(xp+240,yp+35)],GREY,1,z=3)
  if q in [1,2,8]:
   if q==1:
    if j==0:
     P.line([(xp,yp+35),(xp+120,yp+35)],PURPLE,3,z=4);P.line([(xp+120,yp-15),(xp+240,yp-15)],PURPLE,3,z=4)
    elif j==1:P.line([(xp,yp+35),(xp+240,yp-15)],PURPLE,3,z=4)
    elif j==2:P.line([(xp+3*k,yp+35-50*float(sigmoid(-3+6*k/80))) for k in range(81)],PURPLE,3,z=4)
    else:
     P.line([(xp,yp-10),(xp+120,yp+35)],PURPLE,3,z=4);P.line([(xp+120,yp+35),(xp+240,yp-10)],PURPLE,3,z=4);P.line([(xp+120,yp-15),(xp+120,yp+35)],GREY,1,z=3)
   elif q==2:
    if j==0:
     P.line([(xp+3*k,yp+35+17*math.log(.08+.92*k/80)) for k in range(81)],PURPLE,3,z=4);P.line([(xp,yp-15),(xp,yp+35)],GREY,1,z=3)
    elif j==1:P.line([(xp+3*k,yp+35-45*(-1+2*k/80)**2) for k in range(81)],PURPLE,3,z=4)
    elif j==2:P.line([(xp,yp-10),(xp+120,yp+35),(xp+240,yp+35)],PURPLE,3,z=4)
    else:
     for k in range(3):P.rect(xp+k*80,yp+5+k*5,45,30-k*5,fill='#e6dbf4',stroke=PURPLE,sw=2,z=4)
   else:
    if j==0:P.line([(xp,yp+20),(xp+240,yp+20)],PURPLE,3,z=4);P.line([(xp+120,yp+15),(xp+120,yp+25)],PURPLE,2,z=4)
    elif j==1:P.line([(xp+3*k,yp+35-50*float(sigmoid(-3+6*k/80))) for k in range(81)],PURPLE,3,z=4)
    elif j==2:P.line([(xp,yp+35),(xp+70,yp+35),(xp+70,yp+20),(xp+150,yp+20),(xp+150,yp+5),(xp+240,yp+5)],PURPLE,3,z=4)
    else:
     for k,v in enumerate([20,-5,32]):P.ellipse(xp+35+k*80,yp+v,6,fill=PURPLE,z=4)
  elif q==4:
   for k in range(3):shape('ellipse',xp+k*85,yp,28,28,PURPLE)
  elif q==6:
   if j==0:P.line([(xp,yp+30),(xp+80,yp+25),(xp+160,yp+10),(xp+240,yp-10)],PURPLE,3,smooth=True,z=4)
   elif j==1:
    for k in range(3):P.rect(xp+k*70,yp,45,35,fill='#e6dbf4',stroke=PURPLE,sw=2,z=4)
   elif j==2:P.rect(xp+70,yp-10,90,55,fill='#e6dbf4',stroke=PURPLE,sw=2,z=4)
   else:
    a=shape('ellipse',xp,yp+10,18,18,PURPLE);b=shape('ellipse',xp+200,yp-10,18,18,PURPLE);conn(a,b,PURPLE)
  else:
   for k in range(3):P.ellipse(xp+k*85,yp+15,12,fill='#e6dbf4',stroke=PURPLE,sw=2,z=4)

NOTES={
 'train':('ADALINE 是 Widrow 與 Hoff 在1960年提出的自適應線性單元，更新亦稱 LMS 或 Widrow–Hoff 規則。本圖比較訓練與推論兩條路：訓練看連續和，推論才套用本例的0.5門檻。','Widrow and Hoff introduced ADALINE in 1960; its update is also called LMS or the Widrow–Hoff rule. Training uses a continuous score, while inference applies the example’s 0.5 threshold.'),
 'cover':('承接KP07，以連續可微的目標來引導更新。一般Perceptron更新不是對0–1錯誤數做普通梯度下降，不能說Perceptron不能學。ADALINE採平方損失配線性輸出；邏輯回歸採條件Bernoulli模型配最大似然。','Continuous differentiable objectives guide updates. Standard Perceptron updates are not ordinary gradient descent on the 0–1 count, so this does not mean Perceptrons cannot learn. ADALINE uses squared loss and linear outputs; logistic regression uses a conditional Bernoulli model and maximum likelihood.'),
 'lms':('講義用小寫e代表損失，與前單元殘差e容易混淆，這裡改E，殘差定義r等於目標減預測。LMS此處是半平方誤差總和，不是mean。偏置的輸入等效常數一，因此其梯度是殘差負加總。普通批次梯度下降要用適當學習率；一般LMS也常指逐筆或隨機更新，本例明確全批次。','We use E for loss and r=target-prediction for residuals. This LMS objective is a half squared-error sum, not a mean. The bias has a constant input of one. The example uses full-batch gradient descent with a suitable rate; LMS also commonly denotes online/stochastic updates.'),
 'orloss':('新參數0.2、0.2、0.3給四個預測0.3、0.5、0.5、0.7。初始損失半乘零平方加三個一平方等於1.5；更新後半乘負0.3平方、0.5平方、0.5平方、0.3平方之和為0.34。這個特定步長下降不保證任意學習率都下降。','The new parameters give predictions 0.3,0.5,0.5,0.7. The initial half squared sum is 1.5 and the updated sum 0.34. Decrease for this particular step size does not guarantee decrease for arbitrary learning rates.'),
 'orlimit':('這是四筆資料的唯一最小平方解，輸出0.25、0.75、0.75、1.25。0.5門檻正確分類所有四點，但每筆連續值仍差四分之一，因此半平方和為0.125。1.25不是合法機率；ADALINE沒有機率輸出的約束。門檻取決於標籤編碼，本例0/1取0.5，不是所有ADALINE都用這個數。','The unique least-squares solution outputs 0.25,0.75,0.75,1.25. A 0.5 threshold classifies all four correctly, yet residual magnitudes remain one quarter and loss 0.125. These scores are not probabilities. The threshold depends on label coding; 0.5 is specific to this 0/1 example.'),
 'orboundary':('兩條線分別是本例ADALINE與KP02那次Perceptron的結果。Perceptron的解會隨順序、初始化與平手約定改變，不能說它必停在這條線。兩類之間的邊界位置不構成較好泛化能力的實驗證據。','These are this ADALINE solution and the particular KP02 Perceptron result. Perceptron solutions depend on order, initialization and tie conventions. The displayed location between classes is not experimental evidence of better generalization.'),
 'ortrace':('完全重現原程式十二次全批次更新，另以最小平方求解得到下限。更新曲線只連接真實迭代點，不插造中間損失。學習率0.1小於這個二次問題Hessian最大特徵值的二倒數，因此本例收斂；不是所有步長都能得到相同下降圖。','The thirteen points reproduce twelve full-batch source updates. A separate least-squares solve gives the minimum. Lines join actual iterations, not invented intermediate losses. A rate of 0.1 is below two over the largest Hessian eigenvalue here; arbitrary rates need not converge.'),
 'madaline':('沿用講義所列簡化沿革：1962 MRI、1988 MRII、1990 MRIII及其訓練範圍。這不是完整架構或歷史文獻考證。只有早期嘗試可合理放在反向傳播普及之前；不能把1988、1990的改進全部說成發生在反向傳播普及之前。','We retain the handout’s simplified MRI/MRII/MRIII chronology and training scope, not a full architectural or historical survey. Early attempts predate widespread backpropagation; the later 1988/1990 refinements should not all be described that way.'),
 'logistic':('邏輯回歸直接建模給定輸入後的標籤條件機率；LDA通常先建模各類別的輸入分布和先驗，再用貝氏法則取得後驗。本單元不展開LDA假設。logistic模型輸出連續機率，最後才做分類，名稱中的回歸不代表輸出不能連續。講義以Verhulst1845人口成長文獻作歷史小記；不把此年解釋為所有logistic形式第一次出現。','Logistic regression directly models labels conditional on inputs. LDA usually models class-conditional input distributions and priors, then derives posteriors by Bayes’ rule. Logistic outputs are continuous probabilities before classification. The handout cites Verhulst’s 1845 population-growth work; this is not asserted as the first appearance of every logistic formulation.'),
 'likelihood':('固定觀測輸入、假設給定參數後標籤條件獨立，樣本條件機率乘積是似然，對數後成加總。這不是對參數的貝氏後驗，除非另加參數先驗。代入sigmoid後得到y乘線性分數減softplus的式子，負值等於二元交叉熵；實作使用logaddexp避免溢位。','For fixed inputs and conditionally independent labels, the product of label probabilities is a likelihood as a function of parameters; taking logs yields a sum. It is not a Bayesian parameter posterior without a parameter prior. Substituting Sigmoid gives y times score minus Softplus. Its negative is BCE; logaddexp provides stable computation.'),
 'compare':('講義直接比較ADALINE的損失梯度與loglikelihood的梯度，少了一個符號差別。本頁明確比較ADALINE下降方向負梯度與loglikelihood上升方向正梯度，才同為目標減預測乘輸入。偏置併入常數一時，ADALINE的增廣輸入順序要與參數順序配對。','The loss gradient and log-likelihood gradient have opposite sign conventions. We compare ADALINE’s negative loss gradient with logistic’s positive likelihood gradient, both residual times input. Match the augmented feature ordering to the parameter ordering, including the bias.'),
 'hessian':('X採每欄一筆資料，因此尺寸(d+1)乘n，與常見每列一筆的軟體習慣相反。W是n乘n對角矩陣，對角p乘1-p非負，所以H等於負XWX轉置是負半定。這給凹性，不自動給嚴格凹性、有限MLE或Hessian可逆；滿列秩與有限分數時A=-H才正定。','Samples are columns of a (d+1)-by-n matrix, unlike common row-sample software notation. W is diagonal with nonnegative p(1-p), making H negative semidefinite. Concavity alone does not imply strict concavity, a finite MLE or an invertible Hessian. Full row rank and finite logits make A=-H positive definite.'),
 'newton':('一般牛頓式是beta減H逆乘g，無論尋找極大或極小的駐點都採這個形式。此處H是loglikelihood的負半定Hessian，令A=-H才出現加A逆g。不可說最大化的一般牛頓法都直接用加H逆g。實作應解A乘delta等於g，不真的形成逆矩陣；必要時阻尼或線搜尋確保改善。','The generic Newton stationary-point step is beta minus inverse H times g for minima and maxima alike. Here H is the log-likelihood Hessian, so A=-H yields plus inverse A times g. Do not replace this with a universal maximization plus sign. Solve A delta=g in practice; damping or line search may be needed.'),
 'newtontrace':('由原資料與初值重現300個網格點的三條曲線，與每一步精確beta、對數似然一致。起點likelihood為負四ln二，第一步約負2.3526、第二步約負2.3475。所有數字用完整中間精度。這個例子前兩步改善，不宣稱一般牛頓法每一步都上升。','The source data and initialization reproduce three curves on 300 grid points, using each exact beta and log-likelihood. The start is -4 ln 2, then approximately -2.3526 and -2.3475. Intermediate computations retain full precision. This example improves, not every plain Newton step in general.'),
 'outcome':('本例滿秩且無完全或準完全分離，有限最大似然解約負1.362、0.908，邊界1.5，四個預測機率約0.204、0.388、0.612、0.796，硬分類0011造成兩錯。交錯資料確實無法由單一門檻全對，但補充反例門檻0.5給0111只錯一筆。因此兩錯不是資料所迫的最低錯誤數；最大似然與最少0-1錯誤是不同目標。','This full-rank example has neither complete nor quasi-complete separation, giving a finite MLE near (-1.362,0.908). Boundary 1.5 and probabilities (0.204,0.388,0.612,0.796) give 0011, with two errors. A threshold at 0.5 instead gives 0111 with one error. Interleaving prevents zero errors, but two is not the minimum attainable 0–1 count.'),
 'limits':('無正則化loglikelihood凹，不代表有有限最大值或每一步牛頓都改善。完全或準完全分離可能使部分參數往無窮增長；單說不可完全分就保證有限MLE也不足。可加適當正則化，檢查秩、穩定性並用阻尼或線搜尋。此圖的三條sigmoid是明示補充概念例子，不是另一項來源訓練實驗。','Unregularized concavity does not guarantee a finite optimum or improving Newton steps. Complete or quasi-complete separation may send parameters to infinity, so absence of complete separation alone is insufficient. Consider appropriate regularization, rank and stability checks, and damping/line search. The three steepening curves are an added concept example, not another training experiment.'),
 'next':('下一KP以RBF和SOM比較其他建模方式。固定局部基函數、只學輸出層是本課例子的特定設計；一般RBF網路也可以調中心與寬度，不能說所有RBF第一層永遠固定。SOM則以無標籤方式組織輸入。','The next KP considers RBF and SOM. Fixed local basis functions with learned output weights are a specific course example; general RBF networks may also learn centers and widths. SOM organizes inputs without labels.'),
}

def build(lang):
 global LANG,P,SECOND,TEMP
 LANG=lang;K.LANG=lang;title=tr('KP08 ADALINE 與邏輯回歸','KP08 ADALINE and logistic regression')
 d=K.make(title,title,'kp08');sid='scene-kp08';d['nodes']['frame-kp08']['name']=title
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['metadata'].update(createdAt=now,updatedAt=now)
 manifest=[];mascot.STORY={p[0]:p[7:] for p in PAGES}
 for i,(kind,zh,en,tz,te,sec,*_) in enumerate(PAGES):
  P=K.Page(prefix=f'{lang}-p{i+1:02d}');K.P=P;SECOND=[];TEMP=[]
  P.rect(0,0,1280,720,fill='#ffffff',z=0);P.rect(0,0,1280,86,fill='#edf4fa',gradientTo='#f6f8fc',gradientAngle=0,gradientStops=[{'offset':.45,'color':'#e9effb'}],z=1);P.rect(0,0,13,86,fill=BLUE,z=2)
  text(48,20,1180,60,tr(zh,en),34 if lang=='zh' else 28,INK,True,name='page-title');text(1130,684,100,28,f'{i+1:02d} / {len(PAGES)}',17,MUT)
  start=len(P.nodes);draw(kind)
  st=len(P.nodes);mascot.draw(K,kind);phase2(st)
  main_ids=[n['id'] for n in P.nodes[start:]];first=[nid for nid in main_ids if nid not in SECOND]
  if kind=='newton':
   t=text(140,580,800,45,tr('停一下：H 與 A 是同一個矩陣嗎？','Pause: are H and A the same matrix?'),21,ORANGE,True);TEMP=[t['id']];first+=TEMP;main_ids+=TEMP
  ft=len(P.nodes);P.rect(48,640,1184,56,fill='#eaf5ef',radius=12,z=8)
  take=tr(tz,te)
  if kind=='outcome':take=tr('最大似然解{color=#b24b52}不等於{/color}分類錯誤數最少。','Best likelihood {color=#b24b52}need not{/color} mean fewest errors.')
  text(70,650,1130,42,take,26 if lang=='zh' else 23,GREEN,True,name='takeaway');groups=[first,SECOND,[n['id'] for n in P.nodes[ft:]]]
  P.check();assert not P.warnings,P.warnings
  for n in P.nodes:
   n['sceneId']=sid;n['transform']['x']+=i*1440;n['initiallyHidden']=n['id'] in main_ids+groups[-1]
   if n['type']=='text':n['style']['fontFamily']='Noto Sans CJK TC' if lang=='zh' else 'Arial'
   d['nodes'][n['id']]=n;d['scenes'][sid]['nodeIds'].append(n['id'])
  fs=len(d['steps'])
  for beat,ids in enumerate(groups):
   note=tr(*NOTES[kind]) if kind in NOTES and beat==0 else tr(tz,te)
   if beat==0 and kind not in NOTES:note=tr('先確認圖中的輸入、方向與條件。'+tz,'Identify the input, direction and conditions. '+te)
   if beat==1:note=tr('配合新出現的圖解與角色提示：'+tz,'Explain the newly revealed evidence and mascot prompt: '+te)
   if i>=23:note=tr('延伸練習或自我檢核，只給提示不揭答案。'+tz,'Practice or self-check: hints only, no answer. '+te)
   point_=dict(x=i*1440+640,y=360);zoom=1
   if kind=='newton' and beat==1:point_['y']=385;zoom=1.07
   d['steps'].append(dict(id=f'{lang}-p{i+1}-beat{beat+1}',order=len(d['steps']),label=tr(zh,en)+f' · {beat+1}',durationMs=650,holdMs=600,easing='smooth',target=dict(kind='scene-point',sceneId=sid,point=point_,zoom=zoom,referenceViewport=dict(width=1280,height=720)),reveal=dict(timing='after-camera',enter=[dict(nodeId=nid,effect='scale' if beat==1 else 'fade',delayMs=0,durationMs=300) for nid in ids],exitNodeIds=TEMP if beat==1 else []),speakerNotes=note))
  manifest.append(dict(page=i+1,kind=kind,title=tr(zh,en),source_section=sec,first_step=fs,final_step=len(d['steps'])-1,beats=3,extension=i>=23))
 out=ROOT/'output';out.mkdir(exist_ok=True);stem='lec02-kp08-adaline-and-logistic-regression'+('-en' if lang=='en' else '')
 (out/(stem+'.jbd.json')).write_text(json.dumps(d,ensure_ascii=False,indent=1));(out/(lang+'-pages.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=1));print(lang,len(manifest),'pages;',len(d['steps']),'steps;',len(d['nodes']),'nodes')
if __name__=='__main__':save();build('zh');build('en')
