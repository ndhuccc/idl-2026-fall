from pathlib import Path
import json,datetime,math
import pagekit as K
import mascot
from specs import PAGES,MATH
from kp07_nums import *
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

COL=[RED,BLUE,GREEN];NAMES=['Squared','Absolute','Huber']
def graph(keys,x=105,y=240,w=450,h=300,xmin=-3,xmax=3,ymin=0,ymax=9.5,colors=None):
 fx=lambda v:x+(v-xmin)/(xmax-xmin)*w;fy=lambda v:y+h-(v-ymin)/(ymax-ymin)*h
 P.rect(x,y,w,h,fill='#f0f5fa',stroke=GREY,sw=1,fillOpacity=.6,z=1)
 if w>=300:
  yl=r'\ell(e)' if len(keys)>1 else {'huber':r'h_1(e)','ce':r'-\ln p','hinge':r'\ell(yf_o)'}.get(keys[0] if isinstance(keys[0],str) else '',r'\ell')
  eq(x-25,y-42,170,yl,22,h=35)
  if keys==['huber']:eq(x+w-45,y+h+40,70,'e',22,h=35)
 if ymin<=0<=ymax:P.line([(x,fy(0)),(x+w,fy(0))],GREY,2,z=2)
 if xmin<0<xmax:P.line([(fx(0),y),(fx(0),y+h)],GREY,2,z=2)
 for v in sorted(set([xmin,xmax]+([0] if xmin<0<xmax else []))):eq(fx(v)-30,y+h+5,60,f'{v:g}',18,h=32)
 for v in [ymin,ymax]:eq(x-65,fy(v)-16,55,f'{v:g}',18,h=32)
 for j,key in enumerate(keys):
  fn=FUN[key] if isinstance(key,str) else key;pts=[]
  for z in range(600):
   a=xmin+(xmax-xmin)*z/599;v=fn(a);assert ymin-.00001<=v<=ymax+.00001,(key,a,v)
   pts.append((fx(a),fy(v)))
  for k in range(0,599,180):P.line(pts[k:k+181],(colors or COL)[j%len(colors or COL)],4,dash=key=='huber',z=4)
 return fx,fy

def signedbars(vals,x=125,y=280,w=780,h=230,scale=10,labels=None,colors=None):
 P.line([(x,y+h),(x+w,y+h)],GREY,2,z=2)
 for j,v in enumerate(vals):
  cx=x+(j+.5)*w/len(vals);bh=v/scale*h
  if bh>=0:P.rect(cx-36,y+h-bh,72,bh,fill=(colors or [BLUE]*len(vals))[j],z=4)
  else:P.rect(cx-36,y+h,72,-bh,fill=(colors or [PURPLE]*len(vals))[j],z=4)
  if labels:eq(cx-75,y+h+12,150,labels[j],24,h=50)
  eq(cx-65,(y+h-bh-44 if bh>=0 else y+h-bh+5),130,f'{v:g}',24,h=45)

def draw(kind):
 if kind=='cover':
  badge(tr('承接 KP06：輸出只是預測；我們還要知道錯多少','From KP06: an output is a prediction; now measure its error'))
  a=box(80,210,210,100,tr('預測','Prediction'),BLUE,'ellipse',29);b=box(80,455,210,100,tr('目標','Target'),PURPLE,'ellipse',29);c=box(445,320,210,130,tr('損失','Loss'),GREEN,'diamond',32);conn(a,c);conn(b,c,PURPLE)
  st=len(P.nodes);n=box(800,330,160,105,'$L$',ORANGE,'ellipse',36);conn(c,n,GREEN);phase2(st)
  for rr in [60,37,14]:P.ellipse(850,205,rr,fill='transparent',stroke=ORANGE,sw=2,z=3)
  stamp(750,550,tr('錯多少？','HOW WRONG?'));star(850,205,8,ORANGE);shape('triangle',900,175,22,22,BLUE,BLUE,z=6)
 elif kind=='residual':
  badge(tr('講義例子｜四筆殘差，不是四個類別','Handout example | Four residuals, not four classes'))
  eq(75,165,930,MATH[kind][0],31,h=80)
  for j,e in enumerate(RES):
   yy=285+j*78;P.line([(135,yy),(905,yy)],GREY,2,z=2)
   P.ellipse(155,yy,8,fill=PURPLE,z=5);P.ellipse(155+e*70,yy,9,fill=BLUE if j<3 else RED,z=5)
   eq(70,yy-20,65,f'y_{j+1}',22,h=40);eq(180+e*70,yy-20,100,f'e_{j+1}={e}',24,h=45,c=BLUE if j<3 else RED)
  st=len(P.nodes);label(170,580,740,45,tr('補充畫法：每條尺以該筆目標為零點','Added view: each target is the zero of its own ruler'),22,MUT);phase2(st)
 elif kind in ['squared','absolute','huberexample']:
  idx=['squared','absolute','huberexample'].index(kind);c=COL[idx]
  badge(tr('講義補充算例｜四筆殘差固定為 $(1,1,1,10)$','Handout’s added example | Residuals $(1,1,1,10)$'))
  vals=VALS[idx]
  if kind=='squared':
   for j,e in enumerate(RES):
    x=120+j*180;side=23*e;P.rect(x,495-side,side,side,fill=c,fillOpacity=.72,z=3);eq(x-25,505,120,f'e_{j+1}^2={e*e}',23,h=55,c=c)
   eq(65,165,930,MATH[kind][0],31,h=110)
   st=len(P.nodes);eq(70,565,930,MATH[kind][1],32,h=65,c=c);phase2(st)
  else:
   signedbars(vals,x=110,y=300,w=850,h=205,scale=10,labels=[f'e_{j+1}' for j in range(4)],colors=[c]*4)
   eq(80,165,920,MATH[kind][0 if kind=='absolute' else 1],31,h=110)
   st=len(P.nodes);eq(80,565,920,MATH[kind][-1],31,h=60,c=c);phase2(st)
 elif kind=='huber':
  badge(tr(r'門檻條件｜$\delta>0$；圖中取 $\delta=1$',r'Threshold condition | $\delta>0$; plot uses $\delta=1$'))
  fx,fy=graph(['huber'],ymax=2.7,colors=[GREEN]);P.rect(fx(-1),240,fx(1)-fx(-1),300,fill='#f3cf79',fillOpacity=.18,z=2)
  for e in [-1,1]:P.line([(fx(e),240),(fx(e),540)],ORANGE,2,dash=True,z=3);eq(fx(e)-35,580,70,str(e),22,h=38)
  eq(580,200,430,MATH[kind][0],27,h=110,c=GREEN)
  st=len(P.nodes);eq(580,375,430,MATH[kind][1],24,h=110,c=GREEN);phase2(st)
  eq(580,545,430,r'L_{\rm H,sum}=\sum_i h_\delta(e_i)',27,h=90,c=GREEN)
 elif kind=='summean':
  badge(tr('釐清：講義表列加總；MSE、MAE 的 mean 通常代表平均','Clarification: the handout uses sums; “mean” normally denotes an average'))
  for j,(a,b) in enumerate(zip(TOTAL,[25.75,3.25,2.75])):
   yy=205+j*140;n=box(80,yy,190,90,'$'+str(a)+'$',COL[j],'ellipse',31);out=box(745,yy,220,90,'$'+str(b)+'$',COL[j],'ellipse',31);conn(n,out,COL[j]);label(300,yy-30,350,40,tr(['平方誤差','絕對誤差','Huber'][j],NAMES[j]),24,COL[j]);eq(305,yy+65,335,MATH[kind][j],26,h=75,c=COL[j])
 elif kind=='outlier':
  badge(tr('每條長條各自正規化；紅色都是殘差十那一筆','Each bar is normalized separately; red marks the residual of ten'))
  for j,(vals,share) in enumerate(zip(VALS,SHARE)):
   yy=215+j*135;label(65,yy+8,180,45,tr(['平方誤差','絕對誤差','Huber'][j],NAMES[j]),25,COL[j]);xx=265
   for k,v in enumerate(vals):
    ww=600*v/sum(vals);P.rect(xx,yy,ww,52,fill=RED if k==3 else ['#bdd6ed','#a6c7e3','#8eb7d8'][k],z=4);xx+=ww
   eq(295,yy+57,560,MATH[kind][j],24,h=65,c=COL[j])
 elif kind=='curves':
  badge(tr('依來源公式重畫｜保留全段拋物線，不沿用原圖裁切','Recomputed from source formulas | Show the entire parabola'))
  graph(['sq','abs','huber'],x=125,y=235,w=830,h=330,ymax=9.5)
  for j,f in enumerate(MATH[kind]):eq(130+j*280,165,250,f,28,h=60,c=COL[j])
  eq(885,585,80,'e',22,h=35)
 elif kind=='crossentropy':
  badge(tr(r'互斥多類別｜$\ln$ 是自然對數；圖只畫單筆 one-hot 情況','Exclusive classes | Natural logarithm; plot shows one one-hot target'))
  graph(['ce'],xmin=.02,xmax=1,ymax=4,colors=[BLUE]);eq(475,585,80,'p',22,h=35)
  eq(570,190,440,MATH[kind][0],25,h=110)
  st=len(P.nodes);eq(590,365,410,MATH[kind][1],32,h=110,c=BLUE);eq(590,510,410,r'\sum_l p_l=1,\quad p_l\ge0',25,h=100,c=PURPLE);phase2(st)
 elif kind=='cecalc':
  badge(tr('程式重算｜以 $(2,1,0)$ 的完整 Softmax 值計算','Computed | Full-precision Softmax of $(2,1,0)$'))
  eq(80,165,920,r'S=e^2+e+1,\quad\mathbf p=(e^2,e,1)/S',28,h=110)
  xx=120
  for j,p in enumerate(PRED):P.rect(xx,315,780*p,62,fill=K.PAL[j],z=4);label(xx,330,780*p,40,str(j+1),25,'#ffffff');xx+=780*p
  st=len(P.nodes)
  for j in range(3):eq(75+j*310,435,305,MATH[kind][j+2],27,h=110,c=K.PAL[j])
  phase2(st)
 elif kind=='binary':
  badge(tr('補充釐清｜二元機率不等同整組互斥類別向量','Added clarification | A binary scalar differs from a categorical vector'))
  ns=[box(90,200,190,95,'$p$',BLUE,'ellipse',32),box(715,200,240,95,'$1-p$',PURPLE,'ellipse',32)]
  P.rect(125,345,780*.65,65,fill=BLUE,z=3);P.rect(125+780*.65,345,780*.35,65,fill=PURPLE,z=3)
  label(125,415,780,40,tr('長條為互補機率的概念示意','Concept sketch: complementary probabilities'),22,MUT)
  st=len(P.nodes);eq(65,480,950,MATH[kind][0],34,h=110,c=GREEN);eq(210,565,650,MATH[kind][1],25,h=65,c=MUT);phase2(st)
 elif kind=='kl':
  badge(tr(r'KL 的方向｜目標 $\mathbf y$ 對預測 $\mathbf p$',r'KL direction | Target $\mathbf y$ relative to prediction $\mathbf p$'))
  x=115;y=240;baseline=525
  P.line([(x,baseline),(x+465,baseline)],GREY,2,z=2)
  for j,(yv,pv) in enumerate(zip(Y,PRED)):
   for k,v in enumerate([yv,pv]):P.rect(135+j*155+k*50,baseline-v*350,40,v*350,fill=PURPLE if k==0 else BLUE,z=4)
   eq(145+j*155,540,95,str(j+1),24,h=40)
  label(100,175,470,40,tr('紫：目標　藍：預測','Purple: target   Blue: prediction'),23,MUT)
  eq(585,210,430,MATH[kind][0],27,h=110)
  st=len(P.nodes);eq(585,380,430,MATH[kind][1],25,h=110,c=GREEN);eq(585,520,430,MATH[kind][2],25,h=95,c=ORANGE);phase2(st)
 elif kind=='klcalc':
  badge(tr('完整 Softmax 計算｜正、負項相加；最後才取位數','Full-precision Softmax | Add signed terms, then round'))
  eq(70,155,940,r'S=e^2+e+1,\quad\mathbf p=(e^2,e,1)/S,\quad\mathbf y=(0.7,0.2,0.1)',25,h=110)
  zero=435
  P.line([(110,zero),(960,zero)],GREY,2,z=2)
  for j,v in enumerate(TERMS):
   xx=220+j*285;bh=abs(v)*2300
   P.rect(xx-40,zero-bh if v>=0 else zero,80,bh,fill=PURPLE if v>=0 else BLUE,z=4)
   eq(xx-125,285,250,[r'0.7\ln\frac{0.7}{e^2/S}',r'0.2\ln\frac{0.2}{e/S}',r'0.1\ln\frac{0.1}{1/S}'][j],25,h=110,c=PURPLE if v>=0 else BLUE)
   eq(xx-110,540,220,rf'\approx{v:.6f}',25,h=65,c=PURPLE if v>=0 else BLUE)
  st=len(P.nodes);eq(100,590,890,r'D_{\rm KL}=\sum_l y_l\ln(y_l/p_l)\approx0.005787',27,h=55,c=GREEN);phase2(st)
 elif kind=='relation':
  badge(tr('補充公式｜固定目標分布，最小化 CE 等價於最小化 KL','Added identity | For fixed targets, minimizing CE also minimizes KL'))
  eq(60,175,950,MATH[kind][0],30,h=110)
  for j,(v,c,s) in enumerate([(H,PURPLE,r'$H(\mathbf y)$'),(KL,GREEN,r'$D_{\rm KL}$')]):
   xx=110 if j==0 else 110+830*H/SOFTCE;ww=830*v/SOFTCE;piece=P.rect(xx,340,ww,65,fill=c,z=4)
   if j==0:label(xx+15,350,ww-30,45,s,29,'#ffffff')
   else:
    tag=box(800,430,180,65,r'$D_{\rm KL}$',GREEN,'ellipse',25);conn(piece,tag,GREEN,sh=8,th=0)
  eq(70,430,650,MATH[kind][1],29,h=110)
  st=len(P.nodes);eq(80,565,920,MATH[kind][2],29,h=65,c=GREEN);phase2(st)
 elif kind in ['hinge','hingecalc']:
  badge(tr(r'二分類分數｜$y\in\{-1,+1\}$；門檻 $m=1$',r'Binary scores | $y\in\{-1,+1\}$; margin $m=1$'))
  fx,fy=graph(['hinge'],xmin=-2,xmax=3,ymin=-.1,ymax=3.1,colors=[ORANGE]);eq(450,585,120,'yf_o',23,h=40)
  if kind=='hinge':
   P.rect(fx(0),240,fx(1)-fx(0),300,fill='#f3cc73',fillOpacity=.25,z=2)
   # Discontinuous 0-1 curve is drawn in separate segments, without a spurious ramp.
   for aa,bb,v in [(-2,0,1),(0,3,0)]:P.line([(fx(aa),fy(v)),(fx(bb),fy(v))],GREY,3,dash=True,z=4)
   P.ellipse(fx(0),fy(1),6,fill='white',stroke=GREY,sw=2,z=5);P.ellipse(fx(0),fy(0),6,fill=GREY,z=5)
   eq(580,220,430,MATH[kind][0],28,h=110)
   st=len(P.nodes);eq(590,390,420,r'yf_o\ge1\Rightarrow\ell=0',29,h=110,c=GREEN);label(595,540,400,60,tr('灰虛線：0–1 損失','Gray dashed: 0–1 loss'),23,MUT);phase2(st)
  else:
   for v in [1.5,.3,-.5]:P.ellipse(fx(v),fy(max(0,1-v)),8,fill='white',stroke=ORANGE,sw=3,z=6)
   eq(580,170,430,r'y=+1,\quad m=1',26,h=75)
   st=len(P.nodes)
   for j,f in enumerate(MATH[kind][1:]):eq(575,275+j*110,440,f,27,h=90,c=[GREEN,ORANGE,RED][j])
   phase2(st)
 elif kind=='margin':
  badge(tr('補充釐清｜線性分數，非零權重；畫面是概念示意','Added clarification | Linear score, nonzero weights; concept sketch'))
  P.line([(310,195),(310,560)],PURPLE,4,z=3);P.ellipse(525,360,12,fill=BLUE,z=5)
  a=shape('ellipse',318,350,14,14,PURPLE,PURPLE);b=shape('ellipse',510,350,14,14,BLUE,BLUE);conn(a,b,ORANGE,lab='$d$')
  eq(95,570,480,r'd=\frac{s}{\|\mathbf w\|_2}',29,h=70)
  eq(590,200,425,MATH[kind][0],29,h=90)
  st=len(P.nodes);eq(595,350,420,r's\mapsto2s',32,h=80,c=BLUE);eq(590,460,430,r'\frac{2s}{\|2\mathbf w\|_2}=\frac{s}{\|\mathbf w\|_2}',29,h=110,c=GREEN);phase2(st)
 elif kind=='join':
  badge(tr(r'選讀｜$\delta>0$；圖用 $\delta=1$，兩端都核對',r'Optional | $\delta>0$; plot uses $\delta=1$; check both joins'))
  fx,fy=graph(['huber'],ymax=2.7,colors=[GREEN])
  for a in [-1,1]:star(fx(a),fy(.5),12,ORANGE)
  eq(590,190,420,r'\frac12(\pm\delta)^2=\delta(\delta-\frac12\delta)',27,h=110)
  eq(590,335,420,MATH[kind][0],29,h=110,c=GREEN)
  st=len(P.nodes)
  eq(590,475,420,r'h_\delta^{\prime}(-\delta)=-\delta',28,h=65,c=PURPLE);eq(590,550,420,r'h_\delta^{\prime}(+\delta)=+\delta',28,h=65,c=PURPLE);phase2(st)
 elif kind=='choose':
  badge(tr('常用選擇｜不是所有任務都只有一種合法損失','Common choices | Tasks need not have only one valid loss'))
  src=box(60,315,150,120,tr('任務','Task'),ORANGE,'diamond',30)
  for j,(zh,en,s) in enumerate([('實數回歸','Real targets','MSE · MAE · Huber'),('互斥分類','Exclusive classes','Cross-entropy'),('軟分布對齊','Soft distributions','KL'),('分數間隔','Score margin','Hinge')]):
   yy=180+j*110;a=box(335,yy,245,80,tr(zh,en),K.PAL[j],'ellipse',24);b=box(750,yy,245,80,s,K.PAL[j],'ellipse',24);conn(src,a,K.PAL[j],sh=[1,3,5,7][j],th=12);conn(a,b,K.PAL[j])
 elif kind=='myths':
  badge(tr('概念示意｜損失也能用在驗證／測試資料，不限訓練集','Concept sketch | Loss can also be measured on validation/test data'))
  for j,(zh,en) in enumerate([('訓練資料','Training data'),('未見資料','Unseen data')]):
   xx=90+j*480;P.rect(xx,225,190,220,fill='#f3f7fc',stroke=K.PAL[j],sw=3,radius=10,z=3)
   for k in range(4):P.line([(xx+25,265+k*40),(xx+165,265+k*40)],K.PAL[j],2,z=4)
   label(xx-40,465,270,60,tr(zh,en),26,K.PAL[j]);shape('triangle',xx+225,300,50,50,ORANGE,'#fff1d0');label(xx+225,308,50,40,'?',26,ORANGE)
  st=len(P.nodes);label(150,570,760,45,tr('同一把尺、不同資料，必須分開報告','Same ruler, different data: report separately'),25,GREEN);phase2(st)
 elif kind=='next':
  badge(tr('承接講義的 0–1 對照；不把 Perceptron 更新誤當對 0–1 求梯度','From the handout’s 0–1 comparison; Perceptron updates are not its gradient'))
  a=box(65,260,220,105,tr('線性輸出','Linear output'),BLUE,'ellipse',29);b=box(410,260,220,105,r'$(f_o-y)^2$',GREEN,'ellipse',30);c=box(775,260,220,105,tr('更新權重','Update weights'),PURPLE,'ellipse',28);conn(a,b);conn(b,c,GREEN)
  st=len(P.nodes);graph(['sq'],x=415,y=440,w=210,h=145,xmin=-1,xmax=1,ymax=1.1,colors=[GREEN]);label(65,490,230,80,'ADALINE',32,BLUE);phase2(st)
 elif kind=='practice1':
  badge(tr('練習｜不在本頁或講者備註放答案','Practice | No answers on this page or in its speaker notes'))
  signedbars([2,-1,.5,6],x=100,y=210,w=850,h=200,scale=6,colors=[BLUE,PURPLE,BLUE,RED])
  eq(60,505,940,MATH[kind][0],29,h=65)
  st=len(P.nodes);eq(60,575,940,r'L_{\rm sq,sum}=?,\quad L_{\rm abs,sum}=?,\quad L_{\rm H,sum}=?',27,h=55,c=GREEN);phase2(st)
 elif kind=='practice2':
  badge(tr('練習｜三個機率全部留空，請自己正規化','Practice | All probabilities are blank; normalize them yourself'))
  for j in range(3):
   yy=205+j*125;a=box(80,yy,150,80,'$0$',BLUE,'ellipse',30);b=box(435,yy,150,80,r'$e^0$',PURPLE,'ellipse',30);c=box(825,yy,120,80,'?',GREEN,'ellipse',30);conn(a,b);conn(b,c,PURPLE)
 elif kind=='practice3':
  badge(tr('練習｜保留符號，先算乘積，不給數值答案','Practice | Keep the signs; compute the product without a supplied answer'))
  a=box(85,220,230,110,r'$y=-1$',PURPLE,'ellipse',32);b=box(85,450,230,110,r'$f_o=0.8$',BLUE,'ellipse',32);c=box(445,320,170,125,r'$yf_o$',ORANGE,'diamond',32);conn(a,c,PURPLE);conn(b,c)
  st=len(P.nodes);n=box(810,330,140,105,'?',GREEN,'ellipse',36);conn(c,n,GREEN);eq(510,515,490,r'\ell=\max(0,1-yf_o)=?',29,h=95);phase2(st)
 elif kind.startswith('quiz'):
  opts={
  'quiz1':[('零點不可微','Not differentiable at zero'),('平方放大離群值','Squaring amplifies outliers'),('不能用於回歸','Cannot be used for regression'),('輸出不在零與一之間','Outputs lie outside zero to one')],
  'quiz2':[('學習率','Learning rate'),('二次與線性區的切換','Quadratic-to-linear transition'),('類別數','Number of classes'),('分類分數間隔','Classification score margin')],
  'quiz3':[('Hinge','Hinge'),('MSE','MSE'),('KL divergence','KL divergence'),('MAE','MAE')],
  'quiz4':[('線性實數直接當機率','Use any linear real output as a probability'),('多類別用機率向量；二元用機率與補數','Class probabilities; binary probability and complement'),('一定要用 Maxout','Always use Maxout'),('有偏置就一定符合','A bias alone ensures validity')]}
  badge(tr('原題釐清｜區分二元與互斥分類；選項均不標答案','Clarified source question | Distinguish binary and categorical forms') if kind=='quiz4' else tr('四個選項等樣式；只給思考提示','Equal styling for every choice; hints only'))
  for j,(zh,en) in enumerate(opts[kind]):
   xx=70+j%2*500;yy=205+j//2*210
   shape('ellipse',xx,yy,50,50,PURPLE);label(xx,yy+4,50,40,chr(65+j),25,PURPLE);text(xx+65,yy,350,110,tr(zh,en),24 if LANG=='zh' else 22,INK,True)
   # Neutral, candidate-related mini diagrams: the same visual weight for every option.
   xp=xx+110;yp=yy+130
   P.line([(xp,yp+45),(xp+240,yp+45)],GREY,1,z=3)
   if kind=='quiz3':
    fn=[FUN['hinge'],FUN['sq'],None,FUN['abs']][j]
    if fn:
     pts=[(xp+z*3,yp+45-fn(-1+2*z/80)*20) for z in range(81)];P.line(pts,PURPLE,3,z=4)
    else:
     for z,v in enumerate([.5,.3,.2]):P.rect(xp+z*80,yp+45-v*70,50,v*70,fill='#e5dbf4',stroke=PURPLE,sw=2,z=4)
   elif kind=='quiz2':
    if j==0:
     for z in range(3):P.ellipse(xp+40+z*70,yp+20,16,fill='#f0eafb',stroke=PURPLE,sw=2,z=4);P.line([(xp+40+z*70,yp+20),(xp+50+z*70,yp+9)],PURPLE,2,z=5)
    elif j==1:P.line([(xp,yp+45),(xp+80,yp+30),(xp+160,yp+5),(xp+240,yp-20)],PURPLE,3,smooth=True,z=4)
    elif j==2:
     for z in range(3):shape('ellipse',xp+z*80,yp+5,35,35,PURPLE)
    else:P.line([(xp+90,yp-10),(xp+90,yp+40)],PURPLE,3,dash=True,z=4);P.line([(xp+160,yp-10),(xp+160,yp+40)],PURPLE,3,dash=True,z=4)
   elif kind=='quiz1':
    if j<2:P.line([(xp,yp),(xp+120,yp+40),(xp+240,yp)],PURPLE,3,smooth=j==1,z=4)
    elif j==2:
     for z in range(3):P.ellipse(xp+40+z*80,yp+15-z*8,6,fill=PURPLE,z=4)
    else:P.line([(xp,yp+35),(xp+240,yp-10)],PURPLE,3,z=4)
   else:
    if j in [0,2]:P.line([(xp,yp+45),(xp+120,yp+20 if j==2 else yp+10),(xp+240,yp-10)],PURPLE,3,z=4)
    elif j==1:P.rect(xp,yp+5,150,35,fill='#d5c5e9',stroke=PURPLE,sw=2,z=4);P.rect(xp+150,yp+5,90,35,fill='#f2ecf9',stroke=PURPLE,sw=2,z=4)
    else:shape('ellipse',xp+90,yp,60,60,PURPLE);eq(xp+90,yp,60,'b',26,h=60,c=PURPLE)
 else:raise ValueError(kind)

NOTES={
'absolute':('單筆絕對誤差在零點不可微；在正負兩側的導數分別是一和負一。離群值增大時損失線性增長，較不會像平方誤差那樣放大。穩健是相對比較，不代表對任何離群模式都免疫。','Absolute error is not differentiable at zero; its slopes are one and minus one on the two sides. Linear growth makes it less sensitive to large residuals than squared error, not immune to every outlier pattern.'),
'cover':('損失比較預測與目標，提供要最小化的訓練目標。靶心與三角形是教學比喻，不是量測實驗。本單元保留六種損失，先講實數回歸，再講機率分布和分類分數。','A loss compares predictions with targets and supplies an objective to minimize. The target icon is a metaphor, not an experiment. We cover six losses, from real-valued regression to distributions and classification scores.'),
'squared':('單筆平方誤差處處可微，對殘差的導數是兩倍殘差。來源把批次平方誤差總和稱作MSE，本稿明確寫平方誤差總和。四個正方形邊長按殘差絕對值，面積按損失，不能把邊長誤讀為平方損失。真正取mean要除以批次大小，下一頁組將單獨釐清。','Squared error is differentiable everywhere, with residual derivative twice the residual. The handout calls a sum MSE; here it is explicitly the squared-error sum. Square side lengths encode absolute residuals, while areas encode losses. A true mean requires division by batch size, addressed separately.'),
'huber':('Huber門檻delta必須為正。中央半平方、外側delta乘絕對值減半delta；當delta一時，負正一之間為二次區。不要把Huber恆低於MAE當成任意門檻下的結論，delta會改變外側斜率與尺度。','Huber requires a positive delta. The center is half the squared residual; outside is delta times absolute residual minus half delta. With delta one, the quadratic region is [-1,1]. Huber need not remain below MAE for arbitrary thresholds because delta changes the outer slope and scale.'),
'summean':('講義103、13、11都是總和，不能直接稱平均。四筆的平均分別25.75、3.25、2.75；原例的離群占比不受共同除四影響。固定批次時兩者最小化位置相同，但梯度尺度不同，學習率和其他正則項的相對尺度會受影響。','The source values 103,13,11 are sums. Their four-sample means are 25.75,3.25,2.75. Shares are unchanged by dividing every term by four. For a fixed batch, minimizers agree but gradient scaling changes, including relative scaling against regularization.'),
'outlier':('長條分別除各自總和。前三筆用三種淺藍段，最後一筆紅色。占比依完整精度計算：100/103、10/13、9.5/11。此例Huber占比在平方與絕對值之間，並不是保證所有資料或delta都具有同樣次序。不同損失總分不直接用來排名模型。','Each bar is normalized by its own total. Pale blue marks the first three observations and red the last. Shares use full precision. Huber falls between the other shares in this example, not for every dataset or delta. Do not rank models by unlike loss totals.'),
'curves':('依原程式在負三到三取六百點。原圖縱軸上限六會裁掉平方曲線在兩端的九，本稿改上限九點五以完整保留曲線。平方紅、絕對值藍、Huber綠虛線。這是不同懲罰規則的形狀比較，不是驗證集表現實驗。','We use the source’s 600 samples on [-3,3]. Its y-limit of six clips the squared loss of nine at the ends; this deck uses 9.5 to preserve the entire curve. Red is squared, blue absolute, green dashed Huber. These are penalty shapes, not empirical validation results.'),
'crossentropy':('此頁先假設互斥類別的目標是one-hot，交叉熵才簡化成正解類別負對數。一般軟分布目標仍需保留全部類別和。ln採自然對數，單位nats。機率可由Softmax產生，但不限定只有Softmax一種方式；實作常直接把logits交給內建交叉熵，由程式做穩定的log-softmax。','The negative-log simplification assumes a one-hot categorical target. Soft targets require the full class sum. We use natural logs, hence nats. Softmax is a common probability parameterization, not the only one. APIs often accept logits and compute stable log-softmax internally.'),
'cecalc':('計算沿用KP06未取位數的Softmax(2,1,0)，得到負對數0.4076059644、1.4076059644、2.4076059644。講義用三位顯示機率再列這組結果，其實若真的拿0.245重算會是1.406497。本稿用完整中間值並只在最後取位數，避免混用兩種輸入。','The probabilities are the full-precision Softmax of (2,1,0), yielding 0.4076059644,1.4076059644,2.4076059644. Recomputing from the displayed 0.245 instead would give 1.406497. We retain full precision and round only the final display.'),
'binary':('二元標籤零或一，預測p是正類機率，負類機率為一減p。一般二元交叉熵必須保留兩項，只有代入某個硬標籤後才剩一項；可推廣軟二元目標。補充的互補長條只示意分配，不標數值避免被當作來源算例。常用BCEWithLogits直接接未活化分數，比手動Sigmoid再log穩定。','For a binary target, p is the positive probability and 1-p the negative probability. Keep both terms in the general BCE; a hard target removes one only after substitution. Soft binary targets are also possible. The unlabeled bar is conceptual. A logits-based API can compute this more stably than manual Sigmoid and logs.'),
'kl':('目標y與預測p皆為正規化機率分布。y正的位置若p為零，KL為無限；y零的項按極限約定為零。方向y對p一般不可交換。KL可以用於蒸餾等軟分布對齊，不限於分類。它不是一般的對稱距離。','Targets and predictions must be normalized distributions. A positive target with zero predicted probability gives infinite KL. Zero-target terms use the limiting zero convention. KL is generally asymmetric, supports distribution matching such as distillation, and is not an ordinary symmetric metric.'),
'klcalc':('三個分量逐項可正可負，但正規化分布的KL總和非負。以完整Softmax計算總和0.005787411901，而若硬用講義三位機率計算則為0.005853188838。本稿不把後者也寫成0.0058，來源對照說明列出兩個版本。','Individual contributions may be positive or negative; the total KL between normalized distributions is nonnegative. Full-precision Softmax gives 0.005787411901, whereas the handout’s three-digit vector gives 0.005853188838. The review documents both rather than conflating them.'),
'relation':('交叉熵等於目標熵加KL，這是補充的完整關係。固定目標時熵與預測參數無關，所以最小化目標等價。軟目標即使預測完全相同，交叉熵仍等於目標熵，通常不是零；只有one-hot目標熵零才與KL數值相同。堆疊長條依實算熵與KL比例，KL段很細正是本例分布很接近。','Cross-entropy is target entropy plus KL. For fixed targets the entropy is independent of prediction parameters, so the minimizers agree. Even a perfect soft-target match has cross-entropy equal to target entropy, usually positive. One-hot targets have zero entropy. The thin KL bar accurately reflects this example’s close match.'),
'hinge':('Hinge作用在分數而非機率，二元標籤取負一或正一。零損失的條件是標籤乘分數至少margin，不是分數絕對值夠大。0與1之間的signed score符號正確但還未達margin，因此仍罰。零分數時的0-1曲線本頁採不罰的畫圖約定，不是指定模型的平手分類規則。','Hinge acts on scores, not probabilities, with labels -1 or +1. Zero loss requires y times score to reach the margin, not merely a large absolute score. Scores between zero and one have the correct sign but still incur a penalty. The 0-1 plot convention at zero does not prescribe a classifier tie rule.'),
'margin':('線性分數s等於w轉置x加b，非零w時除其範數才是帶符號幾何距離。把權重和偏置同乘正二，不改零邊界，s變兩倍，但距離不變，因此本單元m是分數尺度上的門檻。圖是補充概念示意，不宣稱相同模型訓練結果。','For nonzero w, signed geometric distance is (w transpose x+b)/norm(w). Multiplying weights and bias by positive two leaves the zero boundary and geometric distance unchanged but doubles the score. The Hinge margin is on the score scale. This is an added concept sketch, not a training comparison.'),
'join':('正接點delta：二次值半delta平方、線性值delta乘delta減半delta，相同；左、右一階導數都delta。負接點同值，兩側導數皆負delta。因此Huber在正門檻時一次可微且一階導數連續，但二階導數從一變零，不是處處二次光滑。','At positive delta both values are half delta squared and both first derivatives delta. At negative delta the values again agree and both slopes are minus delta. Huber is continuously once differentiable for positive thresholds, but the second derivative changes from one to zero.'),
'myths':('損失可以量在訓練、驗證或測試資料上，不是只能在訓練資料上計算。低訓練損失不保證低未見資料損失。比較模型需固定資料集、目標、損失定義和加總或平均約定；平方103與絕對值13的大小不能直接證明何者模型較好。','Loss can be measured on training, validation or test data. Low training loss does not guarantee unseen-data performance. Comparisons require the same dataset, target, loss and reduction convention. A squared sum of 103 versus an absolute sum of 13 does not establish which model is better.'),
'next':('銜接KP08的ADALINE：線性單元搭配平方誤差，提供連續可微的目標。來源把Perceptron損失概括成0-1計數，但一般Perceptron更新不是對0-1做普通梯度下降；本頁不沿用那個概括。此處只指出0-1對比的局部指引不足，不把連續與可微混為一談。','ADALINE combines a linear unit with a squared-error objective. The handout calls the Perceptron loss a 0-1 count, but standard Perceptron updates are not ordinary gradient descent on 0-1 loss. We avoid that conflation, and distinguish continuity from differentiability.'),
'quiz4':('原題把Softmax或Sigmoid都描述成輸出和一，對單一Sigmoid不成立；本題明示修訂，區分互斥分類機率向量與二元機率及補數，不提供選項答案。實作可以直接接logits由損失函數內部處理。','The original option incorrectly makes scalar Sigmoid sum to one. The revised question distinguishes a categorical vector from a binary probability and complement, without giving an answer. Implementations may accept logits and perform the transformation internally.'),
}

def build(lang):
 global LANG,P,SECOND,TEMP
 LANG=lang;K.LANG=lang;title=tr('KP07 損失函數','KP07 Loss functions')
 d=K.make(title,title,'kp07');sid='scene-kp07';d['nodes']['frame-kp07']['name']=title
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['metadata'].update(createdAt=now,updatedAt=now)
 manifest=[];mascot.STORY={p[0]:p[7:] for p in PAGES}
 for i,(kind,zh,en,tz,te,sec,*_) in enumerate(PAGES):
  P=K.Page(prefix=f'{lang}-p{i+1:02d}');K.P=P;SECOND=[];TEMP=[]
  P.rect(0,0,1280,720,fill='#ffffff',z=0);P.rect(0,0,1280,86,fill='#edf4fa',gradientTo='#f6f8fc',gradientAngle=0,gradientStops=[{'offset':.45,'color':'#e9effb'}],z=1);P.rect(0,0,13,86,fill=BLUE,z=2)
  text(48,20,1180,60,tr(zh,en),34 if lang=='zh' else 28,INK,True,name='page-title');text(1130,684,100,28,f'{i+1:02d} / {len(PAGES)}',17,MUT)
  start=len(P.nodes);draw(kind)
  st=len(P.nodes);mascot.draw(K,kind);phase2(st)
  main_ids=[n['id'] for n in P.nodes[start:]];first=[nid for nid in main_ids if nid not in SECOND]
  if kind=='klcalc':
   t=text(140,580,800,45,tr('停一下：每一項都一定非負嗎？','Pause: must every term be nonnegative?'),21,ORANGE,True);TEMP=[t['id']];first+=TEMP;main_ids+=TEMP
  ft=len(P.nodes);P.rect(48,640,1184,56,fill='#eaf5ef',radius=12,z=8)
  take=tr(tz,te)
  if kind=='myths':take=tr('訓練低損失{color=#b24b52}不保證{/color}泛化；不同尺不直接比高低。','Low training loss {color=#b24b52}does not ensure{/color} generalization.')
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
   if i>=22:note=tr('延伸練習或自我檢核，只給提示不揭答案。'+tz,'Practice or self-check: hints only, no answer. '+te)
   point_=dict(x=i*1440+640,y=360);zoom=1
   if kind=='klcalc' and beat==1:point_['y']=385;zoom=1.07
   d['steps'].append(dict(id=f'{lang}-p{i+1}-beat{beat+1}',order=len(d['steps']),label=tr(zh,en)+f' · {beat+1}',durationMs=650,holdMs=600,easing='smooth',target=dict(kind='scene-point',sceneId=sid,point=point_,zoom=zoom,referenceViewport=dict(width=1280,height=720)),reveal=dict(timing='after-camera',enter=[dict(nodeId=nid,effect='scale' if beat==1 else 'fade',delayMs=0,durationMs=300) for nid in ids],exitNodeIds=TEMP if beat==1 else []),speakerNotes=note))
  manifest.append(dict(page=i+1,kind=kind,title=tr(zh,en),source_section=sec,first_step=fs,final_step=len(d['steps'])-1,beats=3,extension=i>=22))
 out=ROOT/'output';out.mkdir(exist_ok=True);stem='lec02-kp07-loss-functions'+('-en' if lang=='en' else '')
 (out/(stem+'.jbd.json')).write_text(json.dumps(d,ensure_ascii=False,indent=1));(out/(lang+'-pages.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=1));print(lang,len(manifest),'pages;',len(d['steps']),'steps;',len(d['nodes']),'nodes')
if __name__=='__main__':save();build('zh');build('en')
