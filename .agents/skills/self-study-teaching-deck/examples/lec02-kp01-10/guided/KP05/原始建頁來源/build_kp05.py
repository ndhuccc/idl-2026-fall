from pathlib import Path
import json,datetime,math
import pagekit as K
import mascot
from specs import PAGES
from kp05_nums import *
ROOT=Path(__file__).parent;P=None;LANG='zh';SECOND=[];TEMP=[]
INK,MUT,BLUE,PURPLE,GREEN,ORANGE,GREY,RED=K.INK,K.MUT,K.BLUE,K.PURPLE,K.GREEN,K.ORANGE,K.GREY,K.RED
text,label,eq=K.text,K.label,K.eq
tr=lambda a,b:a if LANG=='zh' else b

def phase2(start):SECOND.extend(n['id'] for n in P.nodes[start:])
def badge(s):text(50,110,950,46,s,21,MUT)
def shape(kind,x,y,w,h,c=BLUE,fill=None,z=3,**style):
 n=P.rect(x,y,w,h,fill=fill or '#edf5fc',stroke=c,sw=2,radius=12 if kind=='rectangle' else 0,z=z,**style);n['shapeKind']=kind;return n

def box(x,y,w,h,s,c=BLUE,kind='rectangle',fs=25):
 n=shape(kind,x,y,w,h,c);label(x+8,y+8,w-16,h-16,s,fs,c);return n

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

def unit(cx,cy,title,c=BLUE):
 return box(cx-58,cy-50,116,100,title,c,'ellipse',26)
def network(weighted=False,roles=False):
 ns=[unit(145,260,'$x_1$'),unit(145,470,'$x_2$'),unit(510,245,'OR' if weighted else '$h_1$',PURPLE),unit(510,480,'NAND' if weighted else '$h_2$',PURPLE),unit(880,360,'AND' if weighted else '$o$',GREEN)]
 for idx,(a,b) in enumerate([(0,2),(1,2),(0,3),(1,3),(2,4),(3,4)]):
  c=BLUE if a<2 else PURPLE
  n=conn(ns[a],ns[b],c,sh=4,th=12,lab=('$'+('1' if b!=3 else '-1')+'$') if weighted else None)
  if weighted:n['connector']['label']['t']=[.28,.28,.72,.28,.60,.60][idx]
 if weighted:
  eq(405,305,210,'b_1=-1',22,h=38,c=PURPLE);eq(405,540,210,'b_2=1.5',22,h=38,c=PURPLE);eq(780,425,220,'b_3=-2',22,h=38,c=GREEN)
 if roles:
  for x,zh,en in [(55,'輸入層','Input'),(405,'隱藏層','Hidden'),(780,'輸出層','Output')]:label(x,160,210,50,tr(zh,en),25,BLUE)
 return ns

def dots_plot(x,y,size,transformed=False):
 lo,hi=-.2,1.2;fx=lambda v:x+(v-lo)/(hi-lo)*size;fy=lambda v:y+size-(v-lo)/(hi-lo)*size
 P.line([(x,y),(x,y+size),(x+size,y+size)],GREY,2,z=2)
 eq(x+size-15,y+size+10,70,'h_1' if transformed else 'x_1',19,h=30);eq(x-60,y-30,65,'h_2' if transformed else 'x_2',19,h=30)
 if transformed:
  # The positive class is on the boundary h1+h2=2 under the >=0 convention.
  cuts=[(.8,1.2),(1.2,.8)];P.line([(fx(a),fy(b)) for a,b in cuts],ORANGE,4,z=4)
  vals=[(0,1,0),(1,0,0),(1,1,1)]
 else:vals=[(r['x'][0],r['x'][1],r['o']) for r in ROWS]
 for a,b,v in vals:
  point(fx(a),fy(b),v,11);eq(fx(a)-48,fy(b)+13,96,f'({a},{b})',17,h=32)
 if transformed:P.ellipse(fx(1),fy(1),21,fill='transparent',stroke=PURPLE,sw=2,z=7);eq(fx(1)-83,fy(1)-53,100,r'\times2',20,h=40,c=PURPLE)
 return fx,fy

def experiment_plot(x,y,size,p):
 fx=lambda v:x+(v+3)/6*size;fy=lambda v:y+size-(v+3)/6*size
 P.rect(x,y,size,size,fill='#eff4fb',stroke=GREY,sw=1,fillOpacity=.6,z=1)
 P.line([(x,fy(0)),(x+size,fy(0))],GREY,1,dash=True,z=2);P.line([(fx(0),y),(fx(0),y+size)],GREY,1,dash=True,z=2)
 for seg in p['segments']:
  for start in range(0,len(seg)-1,180):
   chunk=seg[start:start+181]
   if len(chunk)>1:P.line([(fx(a),fy(b)) for a,b in chunk],PURPLE if p['name']=='relu' else BLUE,4,z=4)
 for a in (-3,0,3):eq(fx(a)-24,y+size+5,48,str(a),16,h=27)
 eq(x+size,y+size+8,60,'x_1',17,h=28);eq(x-55,y-20,60,'x_2',17,h=28)

def draw(kind):
 if kind=='cover':
  badge(tr('承接 KP04：一條線分不開 XOR，接成多層呢？','From KP04: one line cannot separate XOR. What about layers?'))
  network(roles=False)
  stamp(730,555,tr('關鍵：非線性','NONLINEARITY'))
 elif kind=='roles':
  badge(tr('概念示意｜隱藏層指中間的角色，不是資料被遮住','Concept sketch | Hidden names the intermediate role, not blindness to data'))
  network(roles=True)
 elif kind=='dense':
  badge(tr('本圖未畫偏置連線；偏置另計','Bias connections are omitted here and counted separately'))
  network(roles=True)
  eq(295,575,590,r'2\times2+2\times1=6',29,h=45)
 elif kind=='neuron':
  badge(tr('本頁把偏置明寫出來，不藏在權重向量內','Here the bias is explicit rather than folded into a weight vector'))
  a=box(70,275,170,100,r'$\mathbf z$',BLUE,'ellipse',32);b=box(420,275,180,100,r'$\sum$',PURPLE,'ellipse',40);c=box(810,275,170,100,r'$\sigma$',GREEN,'diamond',34)
  conn(a,b,BLUE);conn(b,c,PURPLE)
  eq(80,470,550,r'a_i=\sum_\ell w_{i\ell}z_\ell+b_i',31,h=110)
  st=len(P.nodes);eq(655,480,345,r'z_i=\sigma(a_i)',31,h=100,c=GREEN);phase2(st)
 elif kind=='layer':
  badge(tr('約定釐清｜本稿 A 是講義轉置後的權重矩陣','Convention | A is the transposed weight matrix used in the handout'))
  eq(70,165,930,r'\mathbf A_k=\mathbf W_k^\top,\quad\mathbf z^{(k)}=\sigma(\mathbf A_k\mathbf z^{(k-1)}+\mathbf b_k)',27,h=95)
  # Matrix dimensions are a visual multiplication, not arbitrary card facts.
  for row in range(3):
   for col in range(2):P.rect(190+col*55,330+row*55,45,45,fill='#d9eaf8',stroke=BLUE,sw=1,z=3)
  for row in range(2):P.rect(420,358+row*55,45,45,fill='#e6daf7',stroke=PURPLE,sw=1,z=3)
  for row in range(3):P.rect(650,330+row*55,45,45,fill='#dbefe7',stroke=GREEN,sw=1,z=3);P.rect(850,330+row*55,45,45,fill='#fff0d9',stroke=ORANGE,sw=1,z=3)
  for xx,s in [(155,r'3\times2'),(390,r'2\times1'),(610,r'3\times1'),(805,r'3\times1')]:eq(xx,520,150,s,25,h=48)
  eq(500,390,75,'+',35);eq(745,390,75,'=',35)
  # Correct equation layout: matrix*input + bias = output.
  eq(330,390,65,r'\times',35)
 elif kind=='gates':
  badge(tr('補充｜手工設定權重；每顆採零分也輸出一的階梯','Added | Hand-set weights; each step unit outputs one at zero'))
  network(weighted=True)
  eq(60,580,950,r'f(s)=\mathbf1[s\ge0]',25,h=45)
 elif kind.startswith('case'):
  idx={'case00':0,'case01':1,'case10':2,'case11':3}[kind];r=ROWS[idx];a,b=r['x']
  badge(tr('補充｜同一組手工權重，逐步代入與活化','Added | One set of hand-set weights; substitute and activate'))
  eq(70,165,920,rf'\mathbf x=({a},{b}),\quad f(s)=\mathbf1[s\ge0]',27)
  strings=[rf's_1={a}+{b}-1={r["s1"]:g},\quad h_1=f({r["s1"]:g})={r["h1"]}',rf's_2=-{a}-{b}+1.5={r["s2"]:g},\quad h_2=f({r["s2"]:g})={r["h2"]}',rf's_3={r["h1"]}+{r["h2"]}-2={r["s3"]:g},\quad o=f({r["s3"]:g})={r["o"]}']
  ns=[]
  for j,(ss,val,name) in enumerate(zip(strings,[r['h1'],r['h2'],r['o']],['OR','NAND','AND'])):
   rowstart=len(P.nodes)
   yy=265+j*115;n=shape('ellipse',70,yy,70,70,GREEN if val else GREY,fill='#dcefe5' if val else '#f1f2f4');label(66,yy+18,78,40,name,18,GREEN if val else MUT);eq(170,yy+5,830,ss,28,h=80);ns.append(n)
   if j>0:phase2(rowstart)
  st=len(P.nodes);conn(ns[0],ns[2],PURPLE,sh=12,th=12,curve=True);conn(ns[1],ns[2],PURPLE,sh=8,th=0);phase2(st)
 elif kind=='features':
  badge(tr('補充｜隱藏層輸出作為新座標；右圖只有三個位置','Added | Hidden outputs are new coordinates; only three positions remain'))
  dots_plot(100,240,290);dots_plot(675,240,290,True)
  a=box(420,330,165,105,tr('隱藏轉換','Hidden map'),PURPLE,'diamond',23)
  l=shape('ellipse',375,370,2,2,BLUE,'transparent');rr=shape('ellipse',650,370,2,2,PURPLE,'transparent');conn(l,a,BLUE);conn(a,rr,PURPLE)
  label(95,165,300,50,tr('原始輸入','Original input'),26,BLUE);label(650,150,350,45,tr('隱藏特徵','Hidden features'),26,PURPLE)
  eq(650,195,350,'h_1+h_2=2',24,h=40)
  st=len(P.nodes);eq(40,575,960,r'(0,0)\mapsto(0,1),\quad(0,1),(1,0)\mapsto(1,1),\quad(1,1)\mapsto(1,0)',21,h=45);phase2(st)
 elif kind=='constant':
  badge(tr('補充｜同一組 XOR 權重，活化全部改成恆等','Added | The XOR weights are fixed; all activations become identity'))
  eq(55,175,950,r'y=(x_1+x_2-1)+(-x_1-x_2+1.5)-2',27,h=80)
  # Equal opposite contributions are drawn as two coincident directed lengths with cancellation marks.
  P.rect(160,300,280,36,fill=BLUE,z=3);P.rect(160,385,280,36,fill=PURPLE,z=3)
  eq(155,258,290,'x_1+x_2',24,h=40,c=BLUE);eq(155,342,290,'-x_1-x_2',24,h=40,c=PURPLE)
  eq(485,290,480,r'\mathbf A_2\mathbf A_1=(1,1)\begin{pmatrix}1&1\\-1&-1\end{pmatrix}=(0,0)',24,h=120)
  P.line([(150,285),(450,435)],RED,3,z=7)
  eq(65,465,450,'-1+1.5-2=-1.5',28,h=80)
  st=len(P.nodes);eq(590,480,380,'y=-1.5',37,h=80,c=RED);phase2(st)
 elif kind=='affine':
  badge(tr('本頁沿用欄向量約定；活化都是恆等','Column-vector convention; all activations are identity'))
  a=box(80,235,200,100,r'$\mathbf A_1,\mathbf b_1$',BLUE);b=box(740,235,230,100,r'$\mathbf A_2,\mathbf b_2$',PURPLE);conn(a,b,PURPLE,lab=r'$\mathbf h$')
  eq(65,370,930,r'\mathbf h=\mathbf A_1\mathbf x+\mathbf b_1',29,h=70)
  eq(65,450,930,r'\mathbf y=\mathbf A_2\mathbf h+\mathbf b_2',29,h=70)
  st=len(P.nodes);eq(55,535,950,r'\mathbf y=(\mathbf A_2\mathbf A_1)\mathbf x+(\mathbf A_2\mathbf b_1+\mathbf b_2)',27,h=80,c=GREEN);phase2(st)
 elif kind=='depth':
  badge(tr('本頁不含偏置；含偏置的合併見前頁','No biases here; their composition is shown on the preceding page'))
  ns=[box(70+j*335,240,200,100,r'$\mathbf A_'+str(j+1)+'$',BLUE if j==0 else PURPLE) for j in range(3)]
  conn(ns[0],ns[1],BLUE);conn(ns[1],ns[2],PURPLE)
  eq(60,415,940,r'\mathbf y=\mathbf A_3\mathbf A_2\mathbf A_1\mathbf x=\mathbf V\mathbf x',31,h=80)
  st=len(P.nodes);eq(70,515,920,r'\mathbf V=\mathbf A_3\mathbf A_2\mathbf A_1',31,h=70,c=GREEN);phase2(st)
 elif kind=='experiment':
  badge(tr('程式重現｜種子 2；無偏置；只改兩個隱藏層的活化','Reproduced | Seed 2; no biases; only hidden activations change'))
  for j,p in enumerate(PLOTS):
   plotstart=len(P.nodes)
   xx=105+j*535;experiment_plot(xx,255,300,p);label(xx-35,175,370,55,'Identity' if j==0 else 'ReLU',28,BLUE if j==0 else PURPLE)
   if j==1:phase2(plotstart)
  eq(270,575,700,r'2\to4\to3\to1,\quad f(\mathbf x)=10^{-9}',23,h=45)
 elif kind=='learning':
  badge(tr('概念示意｜手工設計與自動訓練是不同工作','Concept sketch | Manual design and automatic learning are distinct tasks'))
  a=box(80,240,240,100,tr('手工權重','Hand-set weights'),BLUE);b=box(730,240,240,100,'XOR',GREEN);conn(a,b,BLUE,lab=tr('這次展示','Shown here'))
  c=box(80,470,240,100,tr('未知權重','Unknown weights'),PURPLE);d=box(790,455,125,125,'?',ORANGE,'diamond',42);conn(c,d,PURPLE,lab=tr('怎麼學','How to learn?'))
 elif kind=='next':
  badge(tr('階梯非線性可表示 XOR；學習還需要可用的梯度','A step enables XOR representation; learning needs a useful gradient'))
  x,y,w,h=140,240,370,300;fx=lambda v:x+(v+1.5)/3*w;fy=lambda v:y+h-(v+.3)/1.6*h
  P.line([(x,fy(0)),(x+w,fy(0))],GREY,2,z=2);P.line([(fx(0),y+h),(fx(0),y)],GREY,2,z=2)
  P.line([(fx(-1.5),fy(0)),(fx(0),fy(0))],BLUE,4,z=4);P.line([(fx(0),fy(1)),(fx(1.5),fy(1))],BLUE,4,z=4)
  P.ellipse(fx(0),fy(0),7,fill='white',stroke=BLUE,sw=2,z=5);P.ellipse(fx(0),fy(1),7,fill=BLUE,z=5)
  eq(x+w-5,fy(0)+15,55,'s',20,h=35);eq(fx(0)-65,y-30,110,'f(s)',20,h=35);eq(fx(0)-65,fy(0)+5,45,'0',18,h=30);eq(fx(0)-65,fy(1)-15,45,'1',18,h=30)
  eq(650,220,345,r's\ne0:\quad f^{\prime}(s)=0',28,h=90)
  label(645,360,355,60,tr('零點不可微','Not differentiable at zero'),24,RED)
  q=box(770,470,120,90,'?',PURPLE,'diamond',35)
 elif kind=='practice1':
  badge(tr('重算主線例子；這頁與備註只給提示','Recompute the main example; this page and its notes give only hints'))
  eq(80,160,900,r'\mathbf x=(1,0),\quad f(s)=\mathbf1[s\ge0]',27)
  for j,s in enumerate([r's_1=x_1+x_2-1,\quad h_1=f(s_1)',r's_2=-x_1-x_2+1.5,\quad h_2=f(s_2)',r's_3=h_1+h_2-2,\quad o=f(s_3)']):
   eq(65,260+j*110,760,s,28,h=70);label(855,260+j*110,120,70,'?',38,PURPLE)
 elif kind=='practice2':
  badge(tr('無偏置、恆等活化；矩陣採輸出乘輸入的約定','No biases; identity activation; matrices map input columns to outputs'))
  eq(60,170,950,r'\mathbf A_1=\begin{pmatrix}1&0\\1&1\end{pmatrix},\quad \mathbf A_2=(1,2)',29,h=125)
  a=box(100,385,210,100,r'$\mathbf A_1$',BLUE);b=box(740,385,210,100,r'$\mathbf A_2$',PURPLE);conn(a,b,PURPLE)
  eq(80,525,900,r'\mathbf y=\mathbf A_2\mathbf A_1\mathbf x,\quad\mathbf A_2\mathbf A_1=?',30,h=85)
 elif kind=='practice3':
  badge(tr('概念示意｜省略號代表中間各層，不是只有三層','Concept sketch | The ellipsis stands for the intermediate layers'))
  ns=[box(80+j*335,270,200,110,s,PURPLE) for j,s in enumerate([r'$\mathbf A_1$',r'$\cdots$',r'$\mathbf A_{10}$'])];conn(ns[0],ns[1],PURPLE);conn(ns[1],ns[2],PURPLE)
  eq(65,480,940,r'\mathbf A_{10}\cdots\mathbf A_2\mathbf A_1=?',32,h=100)
 elif kind.startswith('quiz'):
  q=int(kind[-1]);opts={1:[('資訊回饋到前層','Information feeds back'),('只向前傳，不回頭','Information only moves forward'),('只有一層神經元','Only one neuron layer'),('可隨意連線','Arbitrary connections')],2:[('是否把輸入層計入總層數','Whether input counts in the total'),('輸入層就是隱藏層','Input is a hidden layer'),('所有文獻數法相同','All authors count alike'),('輸入層一定在最後','Input is always last')],3:[('非線性模型','A nonlinear model'),('一次線性映射','One linear map'),('指數級複雜模型','Exponentially complex'),('無法訓練','Impossible to train')],4:[('更多隱藏層','More hidden layers'),('非線性活化','Nonlinear activation'),('更大學習率','A larger learning rate'),('更小批次','A smaller batch')]}
  badge(tr('原題釐清｜層的角色與計數不同；不附答案','Clarified question | Layer roles and counts differ; no answer shown') if q==2 else tr('四個選項同樣呈現；只給提示','Equal styling for all four choices; hints only'))
  for j,(zh,en) in enumerate(opts[q]):
   xx=70+(j%2)*520;yy=220+(j//2)*215
   shape('ellipse',xx,yy+5,56,56,PURPLE);label(xx,yy+14,56,40,chr(65+j),25,PURPLE)
   text(xx+80,yy,330,95,tr(zh,en),25 if LANG=='zh' else 23,INK,True)
   # Matched neutral visual symbols: all options are illustrated, none is marked correct.
   if q==1:
    if j==2:shape('ellipse',xx+175,yy+112,46,46,PURPLE)
    else:
     a=shape('ellipse',xx+110,yy+110,28,28,BLUE);b=shape('ellipse',xx+260,yy+110,28,28,PURPLE)
     cc=conn(a,b,PURPLE)
     if j==0:cc['connector']['direction']='backward'
     if j==3:cc['connector']['direction']='both'
   elif q==2:
    for k in range(3):shape('ellipse',xx+110+k*65,yy+110,26,26,PURPLE)
   elif q==3:
    if j in (0,1):P.line([(xx+100,yy+155),(xx+180,yy+120 if j==0 else yy+125),(xx+290,yy+155 if j==0 else yy+85)],PURPLE,3,smooth=j==0,z=4)
    elif j==2:
     a=shape('ellipse',xx+100,yy+120,18,18,PURPLE)
     for k in range(3):b=shape('ellipse',xx+270,yy+100+k*25,18,18,PURPLE);conn(a,b,PURPLE)
    else:
     P.line([(xx+140,yy+110),(xx+220,yy+160)],RED,3,z=4);P.line([(xx+140,yy+160),(xx+220,yy+110)],RED,3,z=4)
   elif q==4:
    if j==0:
     ns=[shape('ellipse',xx+95+k*85,yy+120,26,26,PURPLE) for k in range(3)];conn(ns[0],ns[1],PURPLE);conn(ns[1],ns[2],PURPLE)
    elif j==1:P.line([(xx+100,yy+155),(xx+155,yy+150),(xx+210,yy+110),(xx+290,yy+105)],PURPLE,3,smooth=True,z=4)
    elif j==2:
     P.line([(xx+100,yy+135),(xx+290,yy+135)],PURPLE,3,z=4);P.ellipse(xx+250,yy+135,12,fill='#ede5f8',stroke=PURPLE,sw=2,z=5)
    else:
     for k in range(4):P.rect(xx+150+(k%2)*30,yy+110+(k//2)*30,20,20,fill='#e5d8f7',stroke=PURPLE,sw=1,z=4)

NOTES={
'roles':('隱藏層是輸入與輸出之間的運算角色，並不是把原始資料遮起來。第一個隱藏層直接接收輸入；更深的隱藏層接收前層輸出。監督式訓練仍可由目標影響權重更新，不能說隱藏層完全看不到答案訊號。','Hidden names intermediate computation. The first hidden layer directly receives input data; deeper layers receive preceding outputs. In supervised training the targets can influence updates, so hidden layers are not isolated from all supervision.'),
'layer':('為避免講義混用轉置與不轉置的記號，本稿統一A為輸出維度乘輸入維度，使用欄向量。A等於講義W的轉置。偏置明寫，三個輸出需要三個偏置。矩陣示意應讀為A乘輸入再加偏置，經活化成輸出。','Here A has output-by-input shape and acts on column vectors; it equals the handout W transpose. Biases are explicit. Three outputs require three biases. Read the visual as matrix times input, plus bias, followed by activation.'),
'gates':('這是講義明示補充的手工權重例子，不是訓練結果。每顆神經元用分數非負輸出一的階梯，OR偏置負一、NAND偏置一點五、AND偏置負二。六條連線的權重與三個偏置都必須一起看。','This is the handout’s added, hand-set example, not a training result. Each unit fires for a nonnegative score. OR bias is minus one, NAND plus 1.5 and AND minus two. Read all six weights together with the three biases.'),
'features':('兩個正類輸入零一與一零都映到隱藏座標一一，圖上同一位置標兩筆疊合。原始零零映零一，原始一一映一零。在新座標，AND把一一與其餘位置分開；邊界上的點按零分判一。','The positive inputs zero-one and one-zero both map to hidden coordinates one-one; the ring denotes two overlapping samples. Zero-zero maps to zero-one and one-one maps to one-zero. An AND gate separates the positive hidden position, using the stated convention at zero.'),
'constant':('這組特殊權重的線性部分相消，所以恆等活化的輸出是負一點五。這不代表一般全線性網路都輸出常數。來源「連一半XOR都做不到」也不當作精確結論；若再用零為門檻判類零，就會對四點中的兩點。','These particular weights cancel, giving minus 1.5 under identity activations. General all-linear networks need not be constant. The source’s claim of less than half of XOR is not retained: thresholding this constant at zero would classify two of the four points correctly.'),
'affine':('代回隱藏輸出並展開括號，得到A二乘A一的線性部分，以及A二乘b一再加b二的偏置。合併後仍是仿射映射，並非一定是常數。','Substitution gives the product A2 A1 and the bias A2 b1 plus b2. This is another affine map, not necessarily a constant.'),
'depth':('本頁只談沒有偏置的線性映射。矩陣乘積仍是一個矩陣，因此深度沒有增加可表示的函數類別。參數化與最佳化過程仍可能不同，不宣稱所有訓練行為完全相同。','This page concerns bias-free linear maps. Their product is one matrix, so depth does not enlarge the represented function class. Parameterization and optimization can still differ; training behavior is not asserted identical.'),
'experiment':('依原始圖程式種子二重現二、四、三、一網路，所有權重固定且沒有偏置，僅切換兩個隱藏層的恆等或ReLU，輸出層保持線性。畫面畫的是原程式一乘十的負九次方數值等值線；原網格有大片恰零輸出，不能把它當一般的精確零分界。沒有採用講義「分成四塊」的保證。這也不是訓練或泛化比較。','The source seed-two 2-4-3-1 network has fixed weights and no biases. Only the two hidden activations change; output stays linear. We draw the source numerical contour at 1e-9. Large exactly-zero ReLU regions mean it is not an ordinary exact zero boundary. No four-region guarantee, training or generalization claim is made.'),
'learning':('這裡只證明這種架構存在一組能表示XOR的權重。要有效學出所有層的權重，還需要訓練方法、適當容量與其他條件；留給第三章反向傳播。','This establishes that the architecture has weights representing XOR. Learning all layer weights requires a training method, sufficient capacity and other conditions, deferred to Chapter 3 backpropagation.'),
'next':('階梯在非零處導數零，在零處不可微，不能直接提供常規梯度下降所需的有效局部指引。下一節選擇其他非線性；這裡沒有說其他活化處處可微。','A step has zero derivative off zero and no derivative at zero, offering no useful ordinary gradient signal. The next unit considers other nonlinearities without claiming they are differentiable everywhere.'),
'quiz2':('原題的四個選項沒有合理正解，並把輸入層與隱藏層混稱。本頁明示釐清，改檢查閱讀者是否知道要先確認層數的計數約定，不提供答案。','The original options lack a defensible answer and confuse input and hidden roles. The clarified stem checks layer-count conventions; no answer is given.')}

def build(lang):
 global LANG,P,SECOND,TEMP
 LANG=lang;K.LANG=lang;title=tr('KP05 多層感知器與非線性','KP05 Multilayer networks and nonlinearity')
 d=K.make(title,title,'kp05');sid='scene-kp05';d['nodes']['frame-kp05']['name']=title
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['metadata'].update(createdAt=now,updatedAt=now)
 manifest=[];mascot.STORY={p[0]:p[7:] for p in PAGES}
 for i,(kind,zh,en,tz,te,sec,*_) in enumerate(PAGES):
  P=K.Page(prefix=f'{lang}-p{i+1:02d}');K.P=P;SECOND=[];TEMP=[]
  P.rect(0,0,1280,720,fill='#ffffff',z=0);P.rect(0,0,1280,86,fill='#edf4fa',gradientTo='#f6f8fc',gradientAngle=0,gradientStops=[{'offset':.45,'color':'#e9effb'}],z=1);P.rect(0,0,13,86,fill=BLUE,z=2)
  text(48,20,1180,60,tr(zh,en),34 if lang=='zh' else 28,INK,True,name='page-title');text(1130,684,100,28,f'{i+1:02d} / {len(PAGES)}',17,MUT)
  start=len(P.nodes);draw(kind)
  st=len(P.nodes);mascot.draw(K,kind);phase2(st)
  main_ids=[n['id'] for n in P.nodes[start:]];first=[nid for nid in main_ids if nid not in SECOND]
  if kind=='depth':
   t=text(380,355,620,45,tr('停一下：還需要保留三個矩陣嗎？','Pause: must all three matrices remain?'),21,ORANGE,True);TEMP=[t['id']];first+=TEMP;main_ids+=TEMP
  ft=len(P.nodes);P.rect(48,640,1184,56,fill='#eaf5ef',radius=12,z=8)
  take=tr(tz,te)
  if kind=='learning':take=tr('非線性提供可能性，{color=#b24b52}不保證{/color}一定學得會。','Nonlinearity {color=#b24b52}does not ensure{/color} learning.')
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
   if i>=17:note=tr('延伸練習或自我檢核，只給提示不揭答案。'+tz,'Practice or self-check: hints only, no answer. '+te)
   point_=dict(x=i*1440+640,y=360);zoom=1
   if kind=='affine' and beat==1:point_['y']=385;zoom=1.07
   d['steps'].append(dict(id=f'{lang}-p{i+1}-beat{beat+1}',order=len(d['steps']),label=tr(zh,en)+f' · {beat+1}',durationMs=650,holdMs=600,easing='smooth',target=dict(kind='scene-point',sceneId=sid,point=point_,zoom=zoom,referenceViewport=dict(width=1280,height=720)),reveal=dict(timing='after-camera',enter=[dict(nodeId=nid,effect='scale' if beat==1 else 'fade',delayMs=0,durationMs=300) for nid in ids],exitNodeIds=TEMP if beat==1 else []),speakerNotes=note))
  manifest.append(dict(page=i+1,kind=kind,title=tr(zh,en),source_section=sec,first_step=fs,final_step=len(d['steps'])-1,beats=3,extension=i>=17))
 out=R/'output';out.mkdir(exist_ok=True);stem='lec02-kp05-mlp-and-nonlinearity'+('-en' if lang=='en' else '')
 (out/(stem+'.jbd.json')).write_text(json.dumps(d,ensure_ascii=False,indent=1));(out/(lang+'-pages.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=1));print(lang,len(manifest),'pages;',len(d['steps']),'steps;',len(d['nodes']),'nodes')
if __name__=='__main__':save();build('zh');build('en')
