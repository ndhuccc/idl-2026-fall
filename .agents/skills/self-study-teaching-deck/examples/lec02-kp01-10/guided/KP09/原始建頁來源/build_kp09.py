from pathlib import Path
import json,datetime,math
import pagekit as K
import mascot
from specs import PAGES,MATH
from kp09_nums import *
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

def axes(x,y,w,h,lo=-.2,hi=1.2,xlab='x_1',ylab='x_2'):
 fx=lambda a:x+(a-lo)/(hi-lo)*w;fy=lambda b:y+h-(b-lo)/(hi-lo)*h
 P.rect(x,y,w,h,fill='#eff5fa',stroke=GREY,sw=1,fillOpacity=.6,z=1)
 for v in [0,1]:
  P.line([(fx(v),y),(fx(v),y+h)],GREY,1,dash=True,z=2);P.line([(x,fy(v)),(x+w,fy(v))],GREY,1,dash=True,z=2)
  eq(fx(v)-20,y+h+5,40,str(v),18,h=30);eq(x-42,fy(v)-14,35,str(v),18,h=30)
 eq(x+w-25,y+h+32,65,xlab,21,h=40);eq(x-35,y-42,75,ylab,21,h=40)
 return fx,fy

def xorplot(x=125,y=235,s=330,features=False,bound=False,centres=False):
 fx,fy=axes(x,y,s,s,lo=-.2,hi=1.2,xlab=r'\phi_1' if features else 'x_1',ylab=r'\phi_2' if features else 'x_2')
 if bound:
  lo=-.2;hi=1.2;a=max(lo,BOUND-hi);b=min(hi,BOUND-lo)
  P.line([(fx(a),fy(BOUND-a)),(fx(b),fy(BOUND-b))],GREEN,4,z=4)
 if centres:
  for a,b in MU:
   P.line([(fx(a)-10,fy(b)-10),(fx(a)+10,fy(b)+10)],ORANGE,4,z=8);P.line([(fx(a)-10,fy(b)+10),(fx(a)+10,fy(b)-10)],ORANGE,4,z=8)
   # Source radius .7 is an isoresponse contour, not a cutoff; clipped to chart.
   pts=[]
   for t in np.linspace(0,2*math.pi,361):
    xx=a+.7*math.cos(t);yy=b+.7*math.sin(t)
    if -.2<=xx<=1.2 and -.2<=yy<=1.2:pts.append((fx(xx),fy(yy)))
    elif len(pts)>1:P.line(pts,ORANGE,2,dash=True,z=3);pts=[]
   if len(pts)>1:P.line(pts,ORANGE,2,dash=True,z=3)
 for j,(a,b) in enumerate(PHI if features else X):
  if features and j==2:continue
  point(fx(a),fy(b),bool(Y[j]),11,c=PURPLE if Y[j] else BLUE)
  s0=(r'(e^{-1},e^{-1})' if j==1 else ('(1,e^{-2})' if j==0 else '(e^{-2},1)')) if features else f'({a:g},{b:g})'
  eq(fx(a)-70,fy(b)+(15 if b<.5 else -52),140,s0,18,h=45)
 return fx,fy

def topology(x=95,y=355,gap=285,chosen=False):
 nodes=[]
 for j in range(3):
  c=GREEN if chosen and j>0 else BLUE;n=shape('ellipse',x+j*gap,y,75,75,c,'#e9f5ef' if chosen and j>0 else '#edf5fc');eq(x+j*gap,y+6,75,str(j+1),32,h=60,c=c);nodes.append(n)
 for a,b in zip(nodes,nodes[1:]):
  n=conn(a,b,GREY);n['connector']['direction']='none'
 return nodes

def mesh(arr,x,y,s):
 fx,fy=axes(x,y,s,s,lo=0,hi=1,xlab='w_{j1}',ylab='w_{j2}')
 for i in range(8):
  for row in [arr[i,:,:],arr[:,i,:]]:P.line([(fx(a),fy(b)) for a,b in row],BLUE,1.6,z=3)
 for a,b in arr.reshape(-1,2):P.ellipse(fx(a),fy(b),3.8,fill=PURPLE,z=5)

def formulas(kind,x=70,y=170,w=940,dy=125,size=27):
 for j,s in enumerate(MATH.get(kind,[])):eq(x,y+j*dy,w,s,size,h=110,c=GREEN if j==len(MATH[kind])-1 else INK)

def draw(kind):
 if kind=='cover':
  badge(tr('承接 KP08 的最小二乘｜兩條不同的學習路線','From KP08 least squares | Two different learning routes'))
  a=box(65,315,155,100,tr('輸入','Input'),BLUE,'ellipse',30)
  for j,(s,c,t) in enumerate([('RBF',PURPLE,tr('有標籤','With labels')),('SOM',GREEN,tr('無標籤','No labels'))]):
   yy=205+j*235;b=box(395,yy,220,105,s,c,'ellipse',34);d=box(800,yy,200,105,t,c,'diamond',26);conn(a,b,c,sh=2 if j==0 else 6);conn(b,d,c)
  stamp(720,580,tr('兩種目標','TWO GOALS'))
 elif kind=='local':
  badge(tr('精算剖面｜一維、中心零、標準差一','Exact 1D profile | Centre zero; standard deviation one'))
  fx=lambda a:95+(a+3)/6*390;fy=lambda b:550-b*260
  P.rect(95,270,390,280,fill='#eff5fa',stroke=GREY,sw=1,z=1)
  P.line([(fx(a),fy(math.exp(-a*a/2))) for a in np.linspace(-3,3,181)],BLUE,4,z=4)
  for a in [-3,0,1,3]:eq(fx(a)-30,560,60,str(a),20,h=35)
  P.line([(fx(0),fy(1)),(fx(0),550)],ORANGE,2,dash=True,z=3)
  eq(100,195,380,r'\phi(x)=e^{-x^2/2}',29,h=80);eq(220,290,150,'1',26,h=45,c=BLUE)
  st=len(P.nodes);eq(515,225,485,MATH[kind][0],26,h=125);eq(525,420,470,MATH[kind][1],28,h=100);phase2(st)
 elif kind=='fixed':
  badge(tr('限定本課流程｜一般 RBF 也可以再調中心與寬度','Course setup | Other RBF training schemes may also adjust centres and widths'))
  ns=[]
  for j,s in enumerate(['K-means',tr('固定高斯','Fix Gaussians'),tr('最小二乘','Least squares')]):ns.append(box(65+j*345,315,235,115,s,K.PAL[j],'ellipse',27))
  conn(ns[0],ns[1]);conn(ns[1],ns[2],GREEN)
  label(60,205,260,90,tr('由資料群心與\n群內變異決定','Centres / widths\nfrom the data'),23,BLUE)
  st=len(P.nodes);eq(410,475,220,r'(\boldsymbol\mu_i,\sigma_i)',30,h=90,c=PURPLE);eq(780,475,220,r'\mathbf W',36,h=90,c=GREEN);phase2(st)
 elif kind in ['linear','bias']:
  badge(tr('固定局部特徵，再學線性組合','Fixed local features; trainable linear combination'))
  a=box(65,340,150,90,r'$\mathbf x$',BLUE,'ellipse',32);end=box(820,340,170,90,r'$\hat y$',GREEN,'ellipse',34)
  names=[r'$\phi_1$',r'$\phi_2$']+(['$1$'] if kind=='bias' else [])
  for j,s in enumerate(names):
   yy=190+j*(125 if kind=='bias' else 245);n=box(415,yy,175,85,s,PURPLE,'ellipse',33);conn(a,n,PURPLE,sh=2 if j==0 else 6);conn(n,end,GREEN,th=14 if j==0 else 10)
  st=len(P.nodes)
  if kind=='linear':eq(80,540,920,MATH[kind][0],31,h=110)
  else:eq(60,535,450,MATH[kind][0],25,h=100);eq(540,545,460,MATH[kind][1],28,h=90)
  phase2(st)
 elif kind=='matrix':
  badge(tr('樣本放在欄｜不要和每列一筆的程式慣例混用','Samples are columns | Distinguish this from row-sample software conventions'))
  for k,(xx,ww,hh,lab) in enumerate([(90,200,165,r'\mathbf W^\top:k\times m'),(405,280,165,r'\Phi:m\times n'),(800,190,165,r'\hat{\mathbf Y}:k\times n')]):
   n=shape('rectangle',xx,240,ww,hh,K.PAL[k]);eq(xx-35,430,ww+70,lab,25,h=80)
   for j in range(1,4):P.line([(xx+ww*j/4,240),(xx+ww*j/4,405)],K.PAL[k],1,z=4)
  eq(310,290,80,r'\times',36,h=70);eq(705,290,80,'=',36,h=70);st=len(P.nodes);formulas(kind,y=165,dy=355,size=26);phase2(st)
 elif kind=='solve':
  badge(tr('最小範數解｜固定特徵；數值實作解方程或用 SVD','Minimum-norm fit | Fixed features; solve the system or use SVD in code'))
  eq(70,165,940,MATH[kind][0],28,h=100);eq(70,275,940,MATH[kind][1],31,h=105)
  st=len(P.nodes)
  for j,(s,t) in enumerate([(MATH[kind][2],tr('滿列秩','Full row rank')),(MATH[kind][3],tr('奇異時','If singular'))]):
   yy=395+j*115;label(70,yy+20,170,50,t,23,PURPLE);eq(255,yy,740,s,28,h=110,c=GREEN)
  phase2(st)
 elif kind=='xorinput':
  badge(tr('講義補充算例｜中心手選；虛線圈不是硬截斷','Added handout example | Hand-picked centres; dashed circles are not hard cutoffs'))
  xorplot(140,245,320,centres=True)
  eq(515,200,490,MATH[kind][0],25,h=100);eq(515,350,490,MATH[kind][1],26,h=110)
  st=len(P.nodes);eq(525,520,470,r'r=0.7\Rightarrow\phi=e^{-0.49}\approx0.613',24,h=95,c=ORANGE);phase2(st)
 elif kind=='xordist':
  badge(tr('完整計算｜本例的分母二倍變異數等於一','Full calculation | Twice the variance equals one in this example'))
  headers=[r'\mathbf x',r'\|\mathbf x-\boldsymbol\mu_1\|^2',r'\|\mathbf x-\boldsymbol\mu_2\|^2']
  for j,s in enumerate(headers):eq(80+j*310,175,280,s,27,h=100,c=PURPLE)
  for i,(p,dd) in enumerate(zip(X,D)):
   yy=290+i*75;P.line([(70,yy+65),(1000,yy+65)],GREY,1,z=1)
   for j,s in enumerate([f'({p[0]:g},{p[1]:g})',f'{int(dd[0])}',f'{int(dd[1])}']):eq(80+j*310,yy,280,s,28,h=60)
  st=len(P.nodes);eq(80,585,920,r'\phi_i=e^{-d_i^2}',27,h=70,c=GREEN);phase2(st)
 elif kind=='xorfeatures':
  badge(tr('不需要一一對應｜重疊兩點都屬於正類','A one-to-one mapping is unnecessary | Both coincident points have label one'))
  xorplot(650,270,300,features=True)
  for j,s in enumerate([r'\mathbf x',r'(\phi_1,\phi_2)']):eq(65+j*230,180,220,s,26,h=65,c=PURPLE)
  last=[]
  for i,(p,ff) in enumerate(zip(X,PHI)):
   yy=285+i*75;eq(65,yy,180,f'({p[0]:g},{p[1]:g})',24,h=60);n=eq(255,yy,320,rf'\approx({ff[0]:.6f},{ff[1]:.6f})',22,h=60);last.append(n)
  st=len(P.nodes);eq(80,585,520,r'e^{-1}\approx0.367879,\quad e^{-2}\approx0.135335',22,h=65,c=GREEN);phase2(st)
 elif kind=='xorweights':
  badge(tr('兩個對稱權重同為 a｜偏置 b 使兩種輸出恰好相差一','Symmetry gives equal weights a | Bias b makes the two outputs differ by one'))
  for j,s in enumerate(MATH[kind]):eq(80,170+j*112,920,s,29 if j<2 else 27,h=110,c=INK if j<2 else GREEN)
  # Move solved coefficients into beat two.
  SECOND.extend(n['id'] for n in P.nodes[-2:])
 elif kind=='xorfit':
  badge(tr('逐筆核對｜計算用精確指數與完整係數；顯示小數只做近似','Verify every sample | Exact exponentials and full coefficients; decimals are approximate'))
  eq(70,170,940,r'a=-1/(1-e^{-1})^2,\quad b=-a(1+e^{-2})',27,h=100)
  for j,(inputs,ff,out) in enumerate([(r'(0,0),(1,1)',r'1+e^{-2}',0),(r'(0,1),(1,0)',r'2e^{-1}',1)]):
   yy=325+j*150;n=box(80,yy,250,90,'$'+inputs+'$',BLUE,'ellipse',26);end=eq(410,yy,580,rf'\hat y=a({ff})+b={out}',29,h=100,c=GREEN);conn(n,end,GREEN)
 elif kind=='xorboundary':
  badge(tr('分類門檻｜正類三角、負類圓點；a 為負，因此不等號反向','Threshold | Positive triangles, negative circles; a is negative, so reverse the inequality'))
  xorplot(130,235,285,features=True,bound=True)
  eq(515,180,490,MATH[kind][0],29,h=95)
  eq(515,310,490,r'1+e^{-2}\approx1.135335',27,h=90,c=BLUE);eq(515,420,490,r'2e^{-1}\approx0.735759',27,h=90,c=PURPLE)
  st=len(P.nodes);eq(65,570,940,MATH[kind][1],27,h=110,c=GREEN);phase2(st)
 elif kind=='width':
  badge(tr('補充概念實算｜中心固定為零與一；每幅使用同一組兩中心','Added exact profiles | Fixed centres zero and one; two bases in every plot'))
  for j,sig in enumerate([.15,.5,2.]):
   xx=90+j*310;label(xx,185,270,45,r'$\sigma='+str(sig)+'$',25,INK)
   fx=lambda a:xx+(a+.5)/2*265;fy=lambda b:530-b*230
   P.rect(xx,285,265,245,fill='#eff5fa',stroke=GREY,sw=1,z=1)
   for mu,c in [(0,BLUE),(1,PURPLE)]:P.line([(fx(a),fy(math.exp(-(a-mu)**2/(2*sig**2)))) for a in np.linspace(-.5,1.5,181)],c,3,z=4)
   for a in [0,1]:eq(fx(a)-25,540,50,str(a),20,h=35)
  eq(80,590,910,r'\phi_i(x)=\exp[-(x-\mu_i)^2/(2\sigma^2)]',25,h=65,c=GREEN)
 elif kind=='som':
  badge(tr('Kohonen，1982｜一個神經元是一個資料空間原型','Kohonen, 1982 | Each neuron represents a prototype in data space'))
  ns=topology(x=300,y=350,gap=250);a=box(70,210,170,100,r'$\mathbf x$',PURPLE,'ellipse',32)
  for n in ns:conn(a,n,PURPLE,sh=6,th=0)
  for j in range(3):eq(260+j*250,465,155,rf'\mathbf w_{j+1}\in\mathbb R^d',25,h=80)
  st=len(P.nodes);stamp(735,570,tr('沒有標籤','NO LABELS'));phase2(st)
 elif kind=='twospaces':
  badge(tr('概念對照｜左圖位置是權重；右圖位置是固定格子索引','Concept comparison | Left positions are weights; right positions are fixed grid indices'))
  fx,fy=axes(120,280,285,285)
  left=[];right=[]
  for j,(a,b) in enumerate(W0):left.append(box(fx(a)-26,fy(b)-26,52,52,'$'+str(j+1)+'$',BLUE,'ellipse',20))
  for j in range(3):right.append(box(650+j*120,350,55,55,'$'+str(j+1)+'$',PURPLE,'ellipse',24))
  for a,b in zip(right,right[1:]):n=conn(a,b,GREY);n['connector']['direction']='none'
  st=len(P.nodes)
  for a,b in zip(left,right):conn(a,b,ORANGE,sh=4,th=0,curve=True,dash=True)
  phase2(st);label(105,195,370,60,tr('輸入／資料空間','Input / data space'),27,BLUE);label(645,230,350,60,tr('格子／拓撲空間','Grid / topology space'),27,PURPLE)
 elif kind=='winner':
  badge(tr('原始權重｜輸入是橙色菱形，三顆原型是藍色圓點','Original weights | Orange diamond: input; blue circles: prototypes'))
  fx,fy=axes(120,270,285,285)
  for j,(a,b) in enumerate(W0):
   P.ellipse(fx(a),fy(b),9,fill=BLUE,z=4);eq(fx(a)-65,fy(b)+12,130,rf'\mathbf w_{j+1}',22,h=40);P.line([(fx(a),fy(b)),(fx(.9),fy(.7))],GREY,1,dash=True,z=2)
  shape('diamond',fx(.9)-10,fy(.7)-10,20,20,ORANGE,ORANGE);eq(150,185,310,r'\mathbf x=(0.9,0.7)',27,h=75,c=ORANGE)
  for j,(a,b) in enumerate(SX-W0):eq(500,215+j*110,510,rf'd_{j+1}=\sqrt{{({a:.1f})^2+({b:.1f})^2}}\approx{DIST[j]:.6f}',25,h=110)
  st=len(P.nodes);eq(500,555,510,r'z=\arg\min_j d_j=3',29,h=90,c=GREEN);phase2(st)
 elif kind=='update':
  badge(tr('硬鄰域手算｜鄰域為二、三；學習率零點五','Hard-neighbourhood example | Neurons two and three; learning rate 0.5'))
  fx,fy=axes(120,285,280,280)
  for j,((a,b),(c,d)) in enumerate(zip(W0,W1)):
   n=shape('ellipse',fx(a)-8,fy(b)-8,16,16,BLUE,'#ffffff');eq(fx(a)-65,fy(b)+15,130,rf'\mathbf w_{j+1}',20,h=40)
   if j>0:
    end=shape('ellipse',fx(c)-8,fy(d)-8,16,16,GREEN,GREEN);conn(n,end,GREEN,sh=4,th=12)
  shape('diamond',fx(.9)-10,fy(.7)-10,20,20,ORANGE,ORANGE)
  for j,s in enumerate(MATH[kind]):eq(495,205+j*125,515,s,23,h=110,c=GREEN if j<2 else MUT)
  SECOND.extend(n['id'] for n in P.nodes if n.get('type')=='connector')
 elif kind=='rule':
  badge(tr('這頁是硬鄰域規則｜圖中綠色節點更新，藍色不動','Hard neighbourhood | Green nodes update; the blue node stays fixed'))
  topology(x=140,y=215,gap=290,chosen=True)
  st=len(P.nodes);eq(60,355,940,MATH[kind][0],29,h=100);eq(60,455,940,MATH[kind][1],29,h=100);eq(80,555,920,MATH[kind][2],27,h=75);phase2(st)
 elif kind=='schedule':
  badge(tr('講義的索引照錄｜右側是概念圖，不是實驗時間曲線','Handout indexing retained | Right: concept sketch, not an experimental trace'))
  eq(60,190,940,MATH[kind][0],31,h=110);eq(70,300,930,MATH[kind][1],26,h=80)
  for j in range(3):
   xx=210+j*310;r=82-j*25;P.ellipse(xx,490,r,fill='#dfebf8',stroke=PURPLE,sw=2,fillOpacity=.6,z=3);P.ellipse(xx,490,8,fill=PURPLE,z=5)
   if j<2:
    a=shape('ellipse',xx+r,485,1,1,BLUE,BLUE);b=shape('ellipse',xx+310-(r-25),485,1,1,BLUE,BLUE);conn(a,b,BLUE)
  label(75,580,920,45,tr('初始率另給；上式從下一個率開始，本文不自行改索引。','Set the initial rate separately; the formula begins with the next rate.'),22,MUT)
 elif kind=='gaussian':
  badge(tr('來源程式精算｜第一筆輸入與勝者；格子座標以零起算','Source computation | First input and winner; zero-based grid coordinates'))
  h=np.array(FIRST['h']);zi,zj=FIRST['winner'];xx,yy=105,255
  for i in range(8):
   for j in range(8):P.rect(xx+j*37,yy+i*37,32,32,fill=PURPLE,fillOpacity=float(h[i,j]),stroke='#ffffff',sw=1,z=3)
  star(xx+zj*37+16,yy+zi*37+16,13,ORANGE)
  eq(80,175,360,rf'\mathbf r_z=({zi},{zj})',26,h=65,c=ORANGE)
  for j,s in enumerate(MATH[kind]):eq(440,170+j*108,565,s,25,h=105)
  st=len(P.nodes);eq(80,565,340,rf'\mathbf x_0\approx({FIRST["x"][0]:.4f},{FIRST["x"][1]:.4f})',22,h=70,c=GREEN);phase2(st)
 elif kind=='grid':
  badge(tr('原程式重現｜NumPy default_rng(0)，先抽初始權重，再依序抽輸入','Source reproduction | default_rng(0); draw initial weights before all inputs'))
  label(100,175,350,45,tr('初始：零次更新','Initial: zero updates'),25,BLUE);label(625,175,350,45,tr('訓練後：四千次','After 4,000 updates'),25,GREEN)
  mesh(GRID0,130,270,270);st=len(P.nodes);mesh(GRID,650,270,270);phase2(st)
  eq(70,218,930,MATH[kind][0],24,h=65);eq(70,598,930,MATH[kind][1],24,h=60)
 elif kind in ['compare','next','practice3']:
  badge(tr('流程圖｜箭頭表示資料或運算流程','Flow diagram | Arrows show data or computation flow'))
  if kind=='compare':rows=[('RBF',tr('固定基函數','Fixed features'),tr('用標籤求權重','Fit to labels')),('SOM',tr('找勝者與鄰域','Winner + neighbours'),tr('移動原型','Move prototypes'))]
  elif kind=='next':rows=[(tr('局部函數','Local bases'),tr('加權組合','Weighted sum'),tr('更多函數？','More functions?'))]
  else:rows=[('RBF',tr('第一層固定','First layer fixed'),tr('第二層待求','Output unknown')),('MLP',tr('第一層待求','Hidden unknown'),tr('第二層待求','Output unknown'))]
  for i,row in enumerate(rows):
   yy=220+i*235 if len(rows)>1 else 330;ns=[box(65+j*345,yy,235,110,s,K.PAL[i],'ellipse',25) for j,s in enumerate(row)];conn(ns[0],ns[1],K.PAL[i]);conn(ns[1],ns[2],K.PAL[i])
  if kind=='next':stamp(720,555,tr('下一個問題','NEXT QUESTION'))
 elif kind=='practice1':
  badge(tr('題目與提示｜不顯示特徵值或分類結果','Question and hint | No feature values or classification answer'))
  xorplot(135,285,280);shape('diamond',275-10,425-10,20,20,ORANGE,ORANGE)
  formulas(kind,x=470,y=200,w=540,dy=205,size=24)
 elif kind=='practice2':
  badge(tr('題目與提示｜不高亮勝者，不畫更新終點','Question and hint | Neither the winner nor the updated position is marked'))
  fx,fy=axes(130,300,270,270)
  for j,(a,b) in enumerate([(0,0),(1,1)]):P.ellipse(fx(a),fy(b),10,fill=BLUE,z=4);eq(fx(a)-45,fy(b)+15,90,rf'\mathbf w_{j+1}',22,h=45)
  shape('diamond',fx(.8)-10,fy(.6)-10,20,20,ORANGE,ORANGE);formulas(kind,x=470,y=210,w=540,dy=205,size=24)
 elif kind.startswith('quiz'):draw_quiz(int(kind[-1]))
 else:raise ValueError(kind)

def draw_quiz(q):
 badge(tr('保留講義四選項｜同色同尺寸，不附答案','All four source choices | Equal styling; no answer key'))
 choices={
 1:[('由反向傳播訓練','Train by backpropagation'),('K-means 取群心與變異數','K-means centres and variances'),('隨機初始化後凍結','Randomly initialize and freeze'),('用牛頓法迭代','Iterate with Newton’s method')],
 2:[(r'$(\Phi\Phi^\top)^{-1}$',)*2,(r'$(\Phi\Phi^\top)^\dagger\Phi\mathbf Y^\top$',)*2,(r'$\Phi^{-1}$',)*2,(r'$\mathbf W=\mathbf Y$',)*2],
 3:[('距離總和最大者','Largest distance sum'),('距離最小者','Smallest distance'),(r'$\|\mathbf w_j\|$ 最小者',r'Smallest $\|\mathbf w_j\|$'),('隨機選取','Random selection')],
 4:[('所有神經元','All neurons'),('只有勝者本身','Only the winner'),('勝者鄰域內的神經元','Neurons in the winner’s neighbourhood'),('輸出層以外的神經元','Neurons outside the output layer')]
 }
 for j,(zh,en) in enumerate(choices[q]):
  xx=75+j%2*495;yy=220+j//2*195;shape('ellipse',xx,yy,52,52,PURPLE);eq(xx,yy+3,52,chr(65+j),26,h=45,c=PURPLE)
  text(xx+68,yy+3,360,100,tr(zh,en),26 if LANG=='zh' else 24,INK,True)
  for k in range(4):P.ellipse(xx+90+k*65,yy+125,8,fill='#dcd2eb',stroke=PURPLE,sw=1,z=3)

NOTES={
 'fixed':('本課RBF流程：先依訓練資料K-means設定中心及群內變異，再凍結，僅輸出層用最小二乘。第一層不是完全沒有資料驅動的準備；一般RBF也有調中心與寬度的訓練法。Broomhead與Lowe，1988，沿用講義所列。','In this course, K-means determines centres and within-cluster variances before they are frozen for output least squares. Preparation is data-dependent. Other RBF methods can adjust centres and widths. The handout credits Broomhead and Lowe, 1988.'),
 'solve':('正規方程來自梯度2Phi Phi轉置W減2Phi Y轉置為零。Phi滿列秩時Gram可逆。奇異時所列偽逆給最小範數最小二乘解，仍可能有其他解。數值上用SVD或解方程，避免直接形成逆。','Setting 2 Phi Phi-transpose W minus 2 Phi Y-transpose to zero gives the normal equations. Full row rank makes the Gram invertible. In the singular case, the pseudoinverse gives the minimum-norm least-squares fit; other minimizers may exist. Use SVD or solve the system in code.'),
 'xorinput':('XOR中心是手選而非K-means；二倍變異數等於一。半徑0.7虛線圈依來源圖，實際為約0.613的等反應線，並非圈外為零，圖界裁切未延伸。','The XOR centres are hand-picked, not obtained by K-means. Twice the variance is one. Source dashed circles of radius 0.7 indicate response about 0.613, not zero outside; they are clipped to the chart.'),
 'xorweights':('對稱性給a1=a2=a。兩種特徵和相減，a等於負一除以(1-exp(-1))平方；偏置b等於負a乘(1+exp(-2))。完整精度輸出0110；若直接使用四捨五入權重，輸出只是近似。','Symmetry gives equal coefficients a. Subtracting the two equations yields a=-1/(1-exp(-1))^2, then b=-a(1+exp(-2)). Full precision gives 0110; rounded displayed coefficients only approximate these values.'),
 'width':('這是新增的一維概念剖面，不是額外來源訓練實驗。窄基函數的空隙處，無偏置模型輸出趨近零；有偏置則趨近偏置，不能一概說趨近零。近奇異需要穩定求解與可能的正則化，偽逆不自動修復統計泛化問題。','These are added exact one-dimensional profiles, not a source training experiment. Away from narrow bases, a model without bias approaches zero; with bias it approaches that bias. Near singularity calls for stable methods and possibly regularization; a pseudoinverse alone does not solve generalization issues.'),
 'winner':('歐氏距離依完整平方和再開根號，取近似0.860233、0.447214、0.141421。原教材逐維絕對值相加等同L1，本例為1.2、0.6、0.2，勝者也同為三。平手時應固定規則；來源模擬用argmin第一個索引。','Compute full squared sums before square roots: approximately 0.860233,0.447214,0.141421. The original per-coordinate absolute sum is L1; values 1.2,0.6,0.2 select the same winner. Fix a tie rule; source argmin selects the first index.'),
 'schedule':('照錄講義的eta下一步等於eta初始乘一減tau/t；初始率另給，tau零到t減一時，下一率從初始值下降到eta初始/t。此索引並不是本次4000步圖的程式排程，不能把兩者混稱。鄰域逐漸縮小而非每步對半。','The handout uses eta^(tau+1)=eta^(0)(1-tau/t), with eta^(0) set separately and tau from zero to t-1. The next rates run from eta^(0) down to eta^(0)/t. This indexing differs from the 4000-update figure’s code schedule. Shrink neighbourhoods gradually rather than halving every step.'),
 'gaussian':('這是來源產圖實驗的軟鄰域，和手算硬鄰域分開。r是固定格子索引不是資料權重；高斯h對所有有限距離皆為正。初始eta0.51、半徑3.4；最後執行tau3999，eta0.010125、半徑0.40075，並未執行tau4000。','This source simulation uses a soft Gaussian neighbourhood, distinct from the hard hand example. r denotes fixed grid indices. Gaussian h is positive for every finite distance. Initial eta is 0.51 and radius 3.4; at the final executed tau=3999 they are 0.010125 and 0.40075. Tau=4000 is not executed.'),
 'grid':('完全重現default_rng(0)：先抽8乘8乘2初始常態亂數，然後每輪抽兩個均勻輸入；4000輪更新。每個網格有64個點，16條八點科學折線表現112條固定鄰接邊。這是單一seed、均勻正方形資料的展示，不保證一般SOM完全無拓撲摺疊，也不保證所有鄰近資料在格子也鄰近。','Reproduce default_rng(0): draw the 8x8x2 normal initialization before all uniform inputs, then perform 4000 updates. Each grid has 64 nodes and 16 eight-point scientific polylines representing 112 adjacency edges. This one-seed uniform-square demonstration is not a general guarantee of a fold-free SOM or preservation of every input neighbourhood.'),
 'next':('萬有逼近是下一KP的主題，不在本頁宣稱已證明。表達能力不等於最佳化一定成功或有限資料一定能泛化。SOM是競爭式原型映射，不能把所有模型的機制都說成相同的疊層前饋網路。','Universal approximation is the next KP, not proved here. Representation does not guarantee optimization success or generalization. A SOM is a competitive prototype map; these models do not all share an identical stacked feedforward mechanism.'),
}
def build(lang):
 global LANG,P,SECOND,TEMP
 LANG=lang;K.LANG=lang;title=tr('KP09 RBF 與 SOM','KP09 RBF and SOM')
 d=K.make(title,title,'kp09');sid='scene-kp09';d['nodes']['frame-kp09']['name']=title
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['metadata'].update(createdAt=now,updatedAt=now)
 manifest=[];mascot.STORY={p[0]:p[7:] for p in PAGES}
 for i,(kind,zh,en,tz,te,sec,*_) in enumerate(PAGES):
  P=K.Page(prefix=f'{lang}-p{i+1:02d}');K.P=P;SECOND=[];TEMP=[]
  P.rect(0,0,1280,720,fill='#ffffff',z=0);P.rect(0,0,1280,86,fill='#edf4fa',gradientTo='#f6f8fc',gradientAngle=0,gradientStops=[{'offset':.45,'color':'#e9effb'}],z=1);P.rect(0,0,13,86,fill=BLUE,z=2)
  text(48,20,1180,60,tr(zh,en),34 if lang=='zh' else 28,INK,True,name='page-title');text(1130,684,100,28,f'{i+1:02d} / {len(PAGES)}',17,MUT)
  start=len(P.nodes);draw(kind)
  st=len(P.nodes);mascot.draw(K,kind);phase2(st)
  main_ids=[n['id'] for n in P.nodes[start:]];first=[nid for nid in main_ids if nid not in SECOND]
  if kind=='update':
   t=text(495,565,510,60,tr('停一下：勝者的鄰居也會移動嗎？','Pause: do the winner’s neighbours move too?'),21,ORANGE,True);TEMP=[t['id']];first+=TEMP;main_ids+=TEMP
  ft=len(P.nodes);P.rect(48,640,1184,56,fill='#eaf5ef',radius=12,z=8)
  take=tr(tz,te)
  if kind=='never':take=tr('最大似然解{color=#b24b52}不等於{/color}分類錯誤數最少。','Best likelihood {color=#b24b52}need not{/color} mean fewest errors.')
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
   if i>=24:note=tr('延伸練習或自我檢核，只給提示不揭答案。'+tz,'Practice or self-check: hints only, no answer. '+te)
   point_=dict(x=i*1440+640,y=360);zoom=1
   if kind=='update' and beat==1:point_['y']=365;zoom=1.03
   d['steps'].append(dict(id=f'{lang}-p{i+1}-beat{beat+1}',order=len(d['steps']),label=tr(zh,en)+f' · {beat+1}',durationMs=650,holdMs=600,easing='smooth',target=dict(kind='scene-point',sceneId=sid,point=point_,zoom=zoom,referenceViewport=dict(width=1280,height=720)),reveal=dict(timing='after-camera',enter=[dict(nodeId=nid,effect='scale' if beat==1 else 'fade',delayMs=0,durationMs=300) for nid in ids],exitNodeIds=TEMP if beat==1 else []),speakerNotes=note))
  manifest.append(dict(page=i+1,kind=kind,title=tr(zh,en),source_section=sec,first_step=fs,final_step=len(d['steps'])-1,beats=3,extension=i>=24))
 out=ROOT/'output';out.mkdir(exist_ok=True);stem='lec02-kp09-rbf-and-som'+('-en' if lang=='en' else '')
 (out/(stem+'.jbd.json')).write_text(json.dumps(d,ensure_ascii=False,indent=1));(out/(lang+'-pages.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=1));print(lang,len(manifest),'pages;',len(d['steps']),'steps;',len(d['nodes']),'nodes')
if __name__=='__main__':save();build('zh');build('en')
