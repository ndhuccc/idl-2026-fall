from pathlib import Path
import json,datetime,math
import pagekit as K
import mascot
from specs import PAGES
from kp04_nums import *
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

def plot(x=120,y=210,size=300,points=None,w=None,b=0,labels=True,new=False,shade=True,axes=True):
 pts=points if points is not None else XOR
 lo,hi=(-.23,1.23) if len(pts)==4 else (.2,4.8)
 fx=lambda v:x+(v-lo)/(hi-lo)*size;fy=lambda v:y+size-(v-lo)/(hi-lo)*size
 def score(a):return w[0]*a[0]+w[1]*a[1]+b
 if w is not None and shade:
  poly=[(lo,lo),(hi,lo),(hi,hi),(lo,hi)];out=[]
  for a,c in zip(poly,poly[1:]+poly[:1]):
   sa,sc=score(a),score(c)
   if sa>=0:out.append(a)
   if (sa>=0)!=(sc>=0):
    t=-sa/(sc-sa);out.append((a[0]+t*(c[0]-a[0]),a[1]+t*(c[1]-a[1])))
  if len(out)>2:P.poly([(fx(a),fy(c)) for a,c in out],fill='#b8ddc9',fillOpacity=.25,z=1)
 if axes:
  P.line([(x,y),(x,y+size),(x+size,y+size)],GREY,2,z=2)
  eq(x+size-10,y+size+8,70,'x_1',18,h=32);eq(x-60,y-30,65,'x_2',18,h=32)
  for v in ((0,1) if len(pts)==4 else (1,3)):
   eq(fx(v)-30,y+size+8,60,str(v),16,h=25);eq(x-45,fy(v)-12,40,str(v),16,h=25)
 if w is not None:
  cuts=[]
  for v in (lo,hi):
   if w[1]:
    q=-(w[0]*v+b)/w[1]
    if lo<=q<=hi:cuts.append((v,q))
   if w[0]:
    q=-(w[1]*v+b)/w[0]
    if lo<=q<=hi:cuts.append((q,v))
  if len(cuts)>=2:P.line([(fx(a),fy(c)) for a,c in [cuts[0],cuts[-1]]],ORANGE,4,z=4)
 for j,(a,c,lab) in enumerate(pts):
  point(fx(a),fy(c),lab,10 if len(pts)==4 else 5)
  if len(pts)==4 and labels:eq(fx(a)-60,fy(c)+12,120,rf'({a},{c})\!:\!{lab}',18,h=35)
  if w is not None and len(pts)==4 and int(score((a,c))>=0)!=lab:P.ellipse(fx(a),fy(c),19,fill='transparent',stroke=RED,sw=3,z=7)
 if new:star(fx(2.5),fy(2.4),15)
 return fx,fy

def legend(y=590):text(60,y,930,36,tr('藍圓：類零　綠三角：類一　紅圈：錯分　紫星：未見點','Blue circle: class 0. Green triangle: class 1. Red ring: wrong. Purple star: unseen.'),18,MUT)
def stamp(x,y,s):
 n=P.rect(x,y,165,50,fill='#fff2d6',stroke=ORANGE,sw=2,radius=8,z=9);n['transform']['rotation']=-8
 t=label(x,y+4,165,42,s,24,ORANGE);t['transform']['rotation']=-8

def draw(kind):
 if kind=='cover':
  badge(tr('承接 KP03：會更新，就代表什麼都能學會嗎？','From KP03: does an update rule mean every task can be learned?'))
  plot(100,220,245,labels=False)
  a=box(410,230,210,80,tr('表示力','Expressive power'),BLUE,fs=25)
  b=box(710,230,250,80,tr('泛化','Generalization'),PURPLE,fs=25)
  # Two independent tests; no causal arrow between them.
  for i in range(3):tick(450+i*65,420)
  q=box(750,385,170,90,'?',PURPLE,'diamond',40)
  text(390,500,580,50,tr('訓練全對，未見資料呢？','What about unseen data?'),25,INK,True)
  stamp(790,560,tr('兩道關卡','TWO TESTS'))
 elif kind=='xor':
  badge(tr('XOR：兩個輸入不同，標籤才是一','XOR: the label is one when the two inputs differ'))
  plot(220,205,355)
  a=box(690,230,220,80,tr('相同輸入','Same inputs'),BLUE,fs=25)
  eq(700,330,200,'y=0',34,c=BLUE)
  b=box(690,420,220,80,tr('不同輸入','Different inputs'),GREEN,fs=25)
  eq(700,520,200,'y=1',34,c=GREEN)
 elif kind=='lines':
  badge(tr('沿用二元輸出：線上與上方預測類一','Binary output: on or above each line predicts class one'))
  plot(100,230,300,w=(1,1),b=-.5);plot(650,230,300,w=(1,1),b=-1.5)
  eq(65,165,390,'x_1+x_2=0.5',27);eq(615,165,390,'x_1+x_2=1.5',27)
  st=len(P.nodes);label(115,566,280,50,tr('三筆對，一筆錯','3 right / 1 wrong'),25,RED);label(665,566,280,50,tr('一筆對，三筆錯','1 right / 3 wrong'),25,RED);phase2(st)
 elif kind=='updates':
  badge(tr('程式實跑｜零起點；固定順序；一輪四筆','Computed run | Zero start; fixed order; four samples per pass'))
  eq(55,158,960,r'\eta=1,\quad \Delta\mathbf w=(y-o)\mathbf x_{\rm aug},\quad s\ge0\Rightarrow o=1',24)
  eq(80,215,890,r'(0,0),(0,1),(1,0),(1,1)',25)
  P.line([(120,295),(120,545),(950,545)],MUT,2,z=2)
  for v in (0,2,4):
   yy=545-v*55;P.line([(120,yy),(950,yy)],GREY,1,dash=True,z=1);eq(65,yy-16,45,str(v),18,h=35)
  for i,p in enumerate(PASSES):
   h=p['updates']*55;xx=155+i*77
   P.rect(xx,545-h,40,h,fill=RED,gradientTo='#de9295',gradientAngle=90,z=4)
   eq(xx-5,548,50,str(i+1),17,h=30);eq(xx-5,545-h-35,50,str(p['updates']),21,h=30)
  text(135,272,400,35,tr('縱軸：每輪更新次數','Vertical: updates per pass'),19,MUT)
  text(650,590,300,30,tr('橫軸：輪次','Horizontal: pass'),20,MUT)
 elif kind=='cycle':
  badge(tr('程式實跑｜完整權重依序為兩路權重、偏置','Computed run | Augmented weights contain two weights and the bias'))
  eq(65,165,940,r'\mathbf w_{\rm aug}=(w_1,w_2,w_0),\quad \mathbf w_{\rm start}=(-1,0,0)',25)
  ns=[]
  for i,v in enumerate(CYCLE):
   xx=125+(i%2)*580;yy=255+(i//2)*235
   n=shape('rectangle',xx,yy,245,110,BLUE)
   eq(xx+10,yy+5,225,str(tuple(int(a) for a in v['new'])),26)
   eq(xx+10,yy+66,225,rf'\mathbf x=({v["x"][0]},{v["x"][1]})',21,h=36);ns.append(n)
  # Arrange traversal TL -> TR -> BR -> BL -> TL, preserving actual sequence.
  # Nodes 2 and 3 are swapped geometrically to make a simple cycle.
  for idx in (2,3):
   delta=580 if idx==2 else -580
   lo=next(i for i,n in enumerate(P.nodes) if n['id']==ns[idx]['id'])
   for n in P.nodes[lo:lo+3]:n['transform']['x']+=delta
  st=len(P.nodes);conn(ns[0],ns[1],PURPLE);conn(ns[1],ns[2],PURPLE,8,0);conn(ns[2],ns[3],PURPLE,12,4);conn(ns[3],ns[0],PURPLE,0,8);phase2(st)
 elif kind=='proof':
  badge(tr('選讀｜第 1 章不等式證明的幾何直觀','Optional | Geometric intuition for the Chapter 1 inequality proof'))
  fx,fy=plot(145,210,330)
  P.line([(fx(0),fy(0)),(fx(1),fy(1))],BLUE,8,z=4,fillOpacity=.5)
  P.line([(fx(0),fy(1)),(fx(1),fy(0))],GREEN,8,z=4)
  star(fx(.5),fy(.5),17,ORANGE)
  eq(545,190,460,r'\frac{(0,0)+(1,1)}{2}',27,h=110,c=BLUE)
  eq(545,340,460,r'\frac{(0,1)+(1,0)}{2}',27,h=110,c=GREEN)
  st=len(P.nodes);eq(540,485,460,'=(0.5,0.5)',31,c=ORANGE);phase2(st)
 elif kind=='grid':
  badge(tr('程式驗算｜有限格點搜尋，不是完整實數域證明','Computed check | A finite grid search is not a proof over real weights'))
  for a in range(7):
   for b in range(7):P.ellipse(150+a*39,250+b*39,7,fill=BLUE,z=3)
  eq(65,160,900,r'w_1,w_2,w_0\in\{-3,-2,-1,0,1,2,3\}',25)
  eq(85,555,360,'7^3=343',31)
  for j,(_,_,y) in enumerate(XOR):point(640+(j%2)*155,285+(j//2)*150,y,22);tick(680+(j%2)*155,285+(j//2)*150,j<3)
  st=len(P.nodes);eq(555,550,430,r'\max_{\rm grid}\operatorname{correct}=3/4',27);phase2(st)
 elif kind in ('boundaries','newpoint'):
  badge(tr('程式重現｜資料種子 3；訓練種子 100、102；每類 12 點','Reproduced | Data seed 3; training seeds 100, 102; 12 points per class'))
  pts=[(float(x[0]),float(x[1]),int(y)) for x,y in DATA]
  for j,bnd in enumerate(BOUNDS):
   xx=100+j*545
   plot(xx,230,310,pts,bnd['w'],bnd['bias'],labels=False,new=kind=='newpoint')
   eq(xx-30,165,380,rf'{bnd["name"]}:\quad 24/24',29,c=BLUE if j==0 else PURPLE)
   if kind=='newpoint':eq(xx-30,560,380,rf'o_{bnd["name"]}={bnd["pred"]}',33,c=PURPLE)
  if kind=='boundaries':legend(585)
  else:eq(330,590,420,r'\mathbf x_{\rm new}=(2.5,2.4)',23,h=35)
 elif kind=='scores':
  badge(tr('程式重現｜顯示係數取八位；計算保留完整精度','Reproduced | Coefficients shown to eight decimals; calculation uses full precision'))
  for j,bnd in enumerate(BOUNDS):
   yy=180+j*220;w=bnd['w'];b=bnd['bias'];name=bnd['name'];col=BLUE if j==0 else PURPLE
   eq(65,yy,940,rf's_{{{name}}}\approx {w[0]:.8f}(2.5)+{w[1]:.8f}(2.4){b:+.8f}',22)
   eq(590,yy+70,400,rf'\approx {bnd["score"]:.3f},\quad o_{{{name}}}={bnd["pred"]}',29,c=col)
   base=180;terms=[w[0]*2.5,w[1]*2.4,b];acc=0
   for k,v in enumerate(terms):
    eq(65,yy+73+k*25,100,['w_1x_1','w_2x_2','w_0'][k],17,h=30)
    nxt=acc+v;P.rect(base+min(acc,nxt)*18,yy+80+k*25,max(abs(v)*18,1),17,fill=GREEN if v>0 else RED,z=4)
    acc=nxt
   P.line([(base,yy+70),(base,yy+155)],MUT,2,dash=True,z=2)
   eq(base-20,yy+155,40,'0',16,h=28)
 elif kind=='unseen':
  badge(tr('補充釐清｜新點真實標籤未知，不能直接評誰較好','Clarification | The new point has no known label, so no winner is established'))
  a=box(90,245,240,110,tr('訓練：全對','Training: perfect'),BLUE)
  b=box(690,245,270,110,tr('未見資料','Unseen data'),PURPLE)
  conn(a,b,PURPLE,lab=tr('仍要評估','Still evaluate'))
  for xx in (140,235,735,840):
   P.rect(xx,425,70,95,fill='#fff8e8',stroke=ORANGE,sw=2,radius=8,z=3)
   if xx<300:tick(xx+35,470)
   else:label(xx,447,70,60,'?',38,ORANGE)
  st=len(P.nodes);text(370,550,590,55,tr('分歧證明不唯一，還沒證明誰較準。','Different answers do not identify a better model.'),25,INK,True);phase2(st)
 elif kind=='remedies':
  badge(tr('連續損失不保證泛化更好，也不能讓直線解 XOR','Continuous loss neither guarantees generalization nor makes a line solve XOR'))
  a=box(75,215,235,90,tr('無法用直線分開','No separating line'),BLUE,fs=23)
  b=box(610,215,345,90,tr('多層＋非線性：KP05','Layers + nonlinearity: KP05'),BLUE,fs=23)
  c=box(75,445,235,90,tr('邊界不唯一','Nonunique boundary'),PURPLE,fs=23)
  d=box(610,445,345,90,tr('LMS 連續損失：KP08','Continuous LMS loss: KP08'),PURPLE,fs=23)
  conn(a,b,BLUE,lab=tr('擴充表示力','More capacity'));conn(c,d,PURPLE,lab=tr('改學習目標','New objective'))
 elif kind=='history':
  badge(tr('年代順序，並非單一事件造成研究寒冬','Chronological order does not imply a single cause of a research winter'))
  ns=[]
  for i,(year,name,desc) in enumerate([(1958,'Rosenblatt','Perceptron'),(1960,'Widrow & Hoff','ADALINE / LMS'),(1969,'Minsky & Papert',tr('單層限制','Single-layer limits'))]):
   xx=70+i*330
   n=shape('ellipse',xx+55,235,120,120,BLUE if i<2 else PURPLE);eq(xx+55,265,120,str(year),27);ns.append(n)
   label(xx-10,405,280,60,name,23,INK);label(xx-10,485,280,65,desc,25,BLUE)
  for a,b in zip(ns,ns[1:]):conn(a,b,MUT)
 elif kind=='next':
  badge(tr('概念示意｜先不指定權重，保留下一個 KP 的問題','Concept sketch | Weights are left open for the next KP'))
  inp=box(70,315,140,100,'XOR',BLUE,fs=30)
  h1=box(450,220,125,100,'?',PURPLE,'ellipse',36);h2=box(450,450,125,100,'?',PURPLE,'ellipse',36)
  out=box(850,335,125,100,'?',GREEN,'diamond',36)
  conn(inp,h1,BLUE,sh=3,th=12);conn(inp,h2,BLUE,sh=5,th=12);conn(h1,out,PURPLE,sh=4,th=13);conn(h2,out,PURPLE,sh=4,th=11)
  label(345,155,340,55,tr('非線性？','Nonlinearity?'),26,PURPLE)
 elif kind=='practice1':
  badge(tr('XNOR：相同輸入時標籤為一；不附解答','XNOR labels equal inputs as one; no worked answer is shown'))
  plot(130,225,330,points=[(a,b,1-y) for a,b,y in XOR])
  for j,s in enumerate([r'(0,0):\quad ?',r'(1,1):\quad ?',r'(0,1):\quad ?',r'(1,0):\quad ?']):eq(550,220+j*92,440,s,28)
 elif kind=='practice2':
  badge(tr('只畫正解；請自己試著放直線','Only labels are shown; try placing a line yourself'))
  for j,name in enumerate(('AND','XOR')):
   pts=[(a,b,int(a and b)) for a,b,_ in XOR] if name=='AND' else XOR
   plot(110+j*540,240,300,points=pts)
   label(110+j*540,170,300,55,name,29,BLUE)
 elif kind=='practice3':
  badge(tr('兩個模型都對訓練資料百分之百正確','Both models classify every training sample correctly'))
  a=box(100,240,230,100,'$A:24/24$',BLUE);b=box(100,465,230,100,'$B:24/24$',PURPLE)
  q=box(760,350,175,110,'?',ORANGE,'diamond',42)
  conn(a,q,BLUE,sh=4,th=13);conn(b,q,PURPLE,sh=4,th=11)
  label(430,175,540,60,tr('需要什麼新證據？','What new evidence is needed?'),28,ORANGE)
 elif kind.startswith('quiz'):
  qq=int(kind[-1]);opts={1:[('任意多類別資料','Arbitrary multiclass data'),('二元且線性可分','Binary, linearly separable'),('任何迴歸問題','Any regression problem'),('需要非線性邊界','Requires nonlinear boundaries')],2:[('AND','AND'),('OR','OR'),('XOR','XOR'),('NOT','NOT')],3:[('收斂太慢','Slow convergence'),('起點影響邊界，可能貼近某類','The start affects the boundary; it may hug a class'),('損失函數連續','The loss is continuous'),('需要非線性活化','Needs nonlinear activation')],4:[('非線性活化','Nonlinear activation'),('LMS 最小化連續損失','LMS minimizes continuous loss'),('堆疊更多層','Stack more layers'),('把邊界改成曲線','Use a curved boundary')]}
  badge(tr('各選項同樣呈現；只保留原題提示','All options have equal styling; only the source hint is provided'))
  for j,(zh,en) in enumerate(opts[qq]):
   xx=90+(j%2)*520;yy=220+(j//2)*230
   shape('ellipse',xx,yy+10,60,60,PURPLE);label(xx,yy+18,60,45,chr(65+j),26,PURPLE)
   text(xx+85,yy,310,115,tr(zh,en),25 if LANG=='zh' else 23,INK,True)
   option_icon(qq,j,xx+110,yy+112)

def option_icon(q,j,x,y):
 # Each choice has a matched visual representation, without selecting an answer.
 if q==1:
  if j==0:
   for k in range(6):point(x+(k%3)*38,y+(k//3)*35,k%2,7,[BLUE,GREEN,PURPLE][k%3])
  elif j==1:
   for k in range(3):point(x+k*28,y+40,0,7);point(x+110+k*28,y,1,7)
   P.line([(x+80,y+55),(x+90,y-12)],ORANGE,3,z=4)
  elif j==2:
   P.line([(x,y+55),(x+175,y-5)],PURPLE,3,z=4)
   for k in range(5):P.ellipse(x+12+k*36,y+47-k*11,5,fill=BLUE,z=5)
  else:
   P.line([(x,y+50),(x+55,y+45),(x+90,y+5),(x+140,y),(x+180,y+10)],PURPLE,3,smooth=True,z=4)
   for k in range(3):point(x+20+k*55,y+48-k*8,0,6);point(x+30+k*55,y-8,1,6)
 elif q==2:
  if j<3:
   for a,b,_ in XOR:
    v=int(a and b) if j==0 else int(a or b) if j==1 else int(a!=b)
    point(x+a*100,y+50-b*50,v,8)
  else:
   text(x-60,y-10,65,30,tr('輸入','Input'),16,MUT)
   text(x-60,y+30,65,30,tr('輸出','Output'),16,MUT)
   for k in (0,1):
    eq(x+k*100-20,y-10,60,str(k),20,h=30)
    eq(x+k*100-20,y+30,60,str(1-k),20,h=30)
 elif q==3:
  if j==0:
   P.ellipse(x+75,y+25,35,fill='#f6f1fc',stroke=PURPLE,sw=3,z=4)
   P.line([(x+75,y+25),(x+75,y+4)],PURPLE,3,z=5);P.line([(x+75,y+25),(x+92,y+35)],PURPLE,3,z=5)
  elif j==1:
   P.line([(x,y+55),(x+170,y-5)],ORANGE,3,z=4)
   for k in range(4):point(x+45+k*35,y+25-k*10,1,7)
  elif j==2:
   P.line([(x,y+5),(x+45,y+42),(x+90,y+55),(x+135,y+42),(x+180,y+5)],PURPLE,3,smooth=True,z=4)
  else:
   P.line([(x,y+50),(x+60,y+45),(x+110,y+5),(x+175,y)],PURPLE,3,smooth=True,z=4)
 elif q==4:
  if j==0:
   P.line([(x,y+50),(x+60,y+45),(x+110,y+5),(x+175,y)],PURPLE,3,smooth=True,z=4)
  elif j==1:
   P.line([(x,y),(x+45,y+40),(x+90,y+55),(x+135,y+40),(x+180,y)],PURPLE,3,smooth=True,z=4)
  elif j==2:
   ns=[]
   for k in range(3):ns.append(shape('ellipse',x+k*76,y+12,28,28,PURPLE))
   conn(ns[0],ns[1],PURPLE);conn(ns[1],ns[2],PURPLE)
  else:
   P.line([(x,y+50),(x+55,y+45),(x+90,y+5),(x+140,y),(x+180,y+10)],PURPLE,3,smooth=True,z=4)

NOTES={
'lines':('兩條線都沿用分數非負輸出一。第一條錯一筆，第二條錯三筆，來源圖說第二條的錯兩筆並不符合這個約定。兩個例子失敗不能證明所有線都失敗；普遍結論引用第1章證明。','Both lines use the same rule: a nonnegative score predicts one. The first makes one mistake and the second three; the source caption saying two for the second is inconsistent with this convention. Two failed examples do not prove every line fails; the general result is established in Chapter 1.'),
'proof':('同一類所處的半平面是凸的，類零兩端點的中點仍需判零，類一兩端點的中點仍需判一。但兩個中點是同一點，形成矛盾。這裡是第1章四條不等式證明的幾何直觀；不是只靠看線交叉就當成數值搜尋證明。','A class half-plane is convex, so it contains the midpoint of any two of its points. The class-zero endpoints require their midpoint to be zero, and the class-one endpoints require it to be one. Both midpoints coincide. This is geometric intuition for the Chapter 1 inequality proof, rather than a finite search argument.'),
'updates':('本次固定順序、零初始化、學習率一、線上判一。每一輪各點被訪問時的錯誤數是三、三、四、四，之後仍四；這不是同一組固定權重的測試錯誤率。十輪實跑只是示範；學不停的普遍結論來自資料不可分。','This run uses fixed order, zero initialization, rate one and output one at zero. Updates counted during online visits are three, three, then four per pass. They are not a static error rate for a fixed weight vector. Ten simulated passes illustrate the behavior; nonseparability establishes the general claim.'),
'cycle':('從第四輪起，完整權重負一、零、零開始，四次訪問全部更新，繞回負一、零、零。不能把輪末權重一樣誤認為沒有更新。停止準則是一整輪沒有誤分類更新，非輪末權重相同。','From pass four, the augmented weights start at minus one, zero, zero. All four visits cause updates and return to the same weights. Equal end states do not mean no updates; stopping requires an entire pass without a mistaken prediction.'),
'grid':('三個權重各搜尋負三到三的七個整數，共三百四十三組，最多對三筆。這只是有限搜尋的驗算，不能代替第1章對所有實數權重的不等式證明。','Seven integers for each of three weights give 343 triples, with at most three correct labels. This finite search corroborates the result but does not replace the Chapter 1 proof over all real weights.'),
'boundaries':('資料完全依原圖程式的亂數種子三重現，每類十二點。兩個訓練使用種子一百與一百零二，改變初始權重和固定樣本順序。兩次都達訓練全對，並非聲稱隨意畫的兩條線就是這次訓練結果。','The source generator with seed three supplies twelve points per class. Training seeds 100 and 102 change both initial weights and fixed sample order. Both runs reach zero training error; the lines are their actual computed boundaries.'),
'newpoint':('紫色星號是同一個未見輸入二點五、二點四。兩個模型的輸出相反，但講義沒有提供這一點的真實標籤，所以不能由此判定哪條線更準，也沒有可比較的泛化誤差率。下一頁核對分數。','The purple star is the same unseen input, 2.5 and 2.4. The models disagree, but no true label is supplied, so this alone cannot identify a better model or a generalization error rate. The next page verifies the scores.'),
'scores':('係數僅在畫面取八位小數，全部運算仍保留原始浮點精度。長條是兩個加權項與偏置的有號累加；A 的分數約二點四六四，B 約負零點四一五。兩者離零有餘裕，顯示用的取位不會翻轉符號。','Displayed coefficients use eight decimals while the computation retains full precision. The signed bars add the two weighted inputs and the bias. Scores are approximately 2.464 and minus 0.415. Both are far enough from zero that display rounding does not reverse their signs.'),
'unseen':('角色拿考卷只是教學比喻。兩條邊界都訓練全對只說明模型不唯一；未見點的真實類別未知。若要選模型可用獨立驗證資料，最終測試資料留作評估，不用同一份測試反覆挑模型。這是補充釐清。','The exam sheets are a teaching metaphor. Perfect training fits show nonuniqueness, while the new label is unknown. Independent validation data can help select a model; reserve final test data for evaluation rather than repeatedly selecting on the test set. This is an added clarification.'),
'remedies':('上路是表示力不足，需更多層配合非線性，留到KP05。下路是學到哪條邊界，改連續損失提供另一種挑選準則，留到KP08。ADALINE仍是線性模型，不能解XOR，LMS也不保證未見資料較好。','The upper route addresses expressive power through layers and nonlinearity in KP05. The lower route changes how a boundary is selected using a continuous objective in KP08. ADALINE remains linear and cannot solve XOR; LMS does not guarantee better unseen performance.'),
'history':('時間軸按講義列出的三個年份排列；它不是因果實驗。避免稱Perceptron是歷史上第一個神經元，也避免把1969年的一本書當作整段研究寒冬的唯一原因。','These are the three milestones in the handout, arranged chronologically, not a causal experiment. Avoid calling the Perceptron the first neuron or attributing an entire research winter to one 1969 book.'),
'quiz1':('原題「演算法只適用」容易被誤解為不可分資料完全不能使用，因此題幹釐清為收斂保證的條件。保留四個選項與提示，不揭答案。','The original wording could imply the algorithm cannot even be run on nonseparable data. The stem instead asks about the convergence guarantee. All four choices and the hint are retained without giving the answer.'),
'quiz4':('來源題幹把連續損失當成泛化改善的保證，與本講義的限制說明不一致。改問ADALINE使用的學習目標，保留四個選項與提示，不揭答案。','The original stem risks presenting continuous loss as a generalization guarantee. It now asks which objective ADALINE uses, retaining all options and the hint without giving the answer.')}

def build(lang):
 global LANG,P,SECOND,TEMP
 LANG=lang;K.LANG=lang;title=tr('KP04 感知器的限制與泛化','KP04 Perceptron limits and generalization')
 d=K.make(title,title,'kp04');sid='scene-kp04';d['nodes']['frame-kp04']['name']=title
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['metadata'].update(createdAt=now,updatedAt=now)
 manifest=[]
 mascot.STORY={p[0]:p[7:] for p in PAGES}
 for i,(kind,zh,en,tz,te,sec,*_) in enumerate(PAGES):
  P=K.Page(prefix=f'{lang}-p{i+1:02d}');K.P=P;SECOND=[];TEMP=[]
  P.rect(0,0,1280,720,fill='#ffffff',z=0)
  P.rect(0,0,1280,86,fill='#edf4fa',gradientTo='#f6f8fc',gradientAngle=0,gradientStops=[{'offset':.45,'color':'#e9effb'}],z=1)
  P.rect(0,0,13,86,fill=BLUE,z=2)
  text(48,20,1180,60,tr(zh,en),34 if lang=='zh' else 28,INK,True,name='page-title')
  text(1130,684,100,28,f'{i+1:02d} / {len(PAGES)}',17,MUT)
  start=len(P.nodes);draw(kind)
  st=len(P.nodes);mascot.draw(K,kind);phase2(st)
  main_ids=[n['id'] for n in P.nodes[start:]]
  first=[nid for nid in main_ids if nid not in SECOND]
  # A short prompt is shown in beat one and exits as the mascot starts to explain.
  if kind=='cycle':
   st=len(P.nodes);t=text(415,395,250,65,tr('權重一樣，\n代表沒更新？','Same weights:\nno updates?'),21,ORANGE,True)
   TEMP=[t['id']];first+=TEMP;main_ids+=TEMP
  ft=len(P.nodes);P.rect(48,640,1184,56,fill='#eaf5ef',radius=12,z=8)
  take=tr(tz,te)
  if kind=='unseen':take=tr('訓練全對，{color=#b24b52}不是{/color}對未見資料的保證。','A perfect training fit is {color=#b24b52}no guarantee{/color} on unseen data.')
  text(70,650,1130,42,take,26 if lang=='zh' else 23,GREEN,True,name='takeaway')
  groups=[first,SECOND,[n['id'] for n in P.nodes[ft:]]]
  P.check();assert not P.warnings,P.warnings
  for n in P.nodes:
   n['sceneId']=sid;n['transform']['x']+=i*1440
   if n['type']=='connector':
    for wp in n['connector'].get('waypoints',[]):wp['x']+=i*1440
   n['initiallyHidden']=n['id'] in main_ids+groups[-1]
   if n['type']=='text':n['style']['fontFamily']='Noto Sans CJK TC' if lang=='zh' else 'Arial'
   d['nodes'][n['id']]=n;d['scenes'][sid]['nodeIds'].append(n['id'])
  fs=len(d['steps'])
  for beat,ids in enumerate(groups):
   note=tr(*NOTES[kind]) if kind in NOTES and beat==0 else tr(tz,te)
   if beat==0 and kind=='cover':note=tr('本節回到KP02的零一標籤，線上判一；這與KP03只是標籤編碼不同。','This unit uses the binary labels of KP02, predicting one at zero. KP03 used a different label coding. ')+note
   if beat==0 and kind not in NOTES:note=tr('先觀察圖中的條件、資料與閱讀順序。'+tz,'First identify conditions, data and the reading order. '+te)
   if beat==0 and kind=='cover':note+=tr('本節回到KP02的零一標籤，線上判一；KP03使用正負標籤。','This unit returns to KP02 binary labels, predicting one at zero; KP03 used signed labels.')
   if beat==1:note=tr('配合新出現的圖解與角色提示，說明：'+tz,'Explain the newly revealed evidence and the mascot prompt: '+te)
   if i>=14:note=tr('這是練習或自我檢核頁，不提供答案。'+tz,'This is a practice or self-check page; no answer is supplied. '+te)
   point_=dict(x=i*1440+640,y=360);zoom=1
   if kind in ('updates','scores') and beat==1:point_['y']=370;zoom=1.07
   d['steps'].append(dict(id=f'{lang}-p{i+1}-beat{beat+1}',order=len(d['steps']),label=tr(zh,en)+f' · {beat+1}',durationMs=650,holdMs=600,easing='smooth',target=dict(kind='scene-point',sceneId=sid,point=point_,zoom=zoom,referenceViewport=dict(width=1280,height=720)),reveal=dict(timing='after-camera',enter=[dict(nodeId=nid,effect='scale' if beat==1 else 'fade',delayMs=0,durationMs=300) for nid in ids],exitNodeIds=TEMP if beat==1 else []),speakerNotes=note))
  manifest.append(dict(page=i+1,kind=kind,title=tr(zh,en),source_section=sec,first_step=fs,final_step=len(d['steps'])-1,beats=3,extension=i>=14))
 out=ROOT/'output';out.mkdir(exist_ok=True);stem='lec02-kp04-perceptron-limits-and-generalization'+('-en' if lang=='en' else '')
 (out/(stem+'.jbd.json')).write_text(json.dumps(d,ensure_ascii=False,indent=1));(out/(lang+'-pages.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=1))
 print(lang,len(manifest),'pages;',len(d['steps']),'steps;',len(d['nodes']),'native nodes')
if __name__=='__main__':save();build('zh');build('en')
