from pathlib import Path
import json,datetime,math
import numpy as np
import pagekit as K
from specs import PAGES
from kp10_nums import XS,NS,approx,sigmoid,save
ROOT=Path(__file__).parent;P=None;LANG='zh';SECOND=[]
INK,MUT,BLUE,PURPLE,GREEN,ORANGE,GREY,RED=K.INK,K.MUT,K.BLUE,K.PURPLE,K.GREEN,K.ORANGE,K.GREY,K.RED
BAN='#173e70';text,label=K.text,K.label
tr=lambda a,b:a if LANG=='zh' else b
def eq(x,y,w,s,size=28,h=80,c=INK):
 if any(q in s for q in [r'\frac',r'\sum',r'\sup',r'\int']):h=max(h,110)
 return K.eq(x,y,w,s,size,h,c)
def phase2(st):SECOND.extend(n['id'] for n in P.nodes[st:])
def shape(kind,x,y,w,h,c=BLUE,fill=None,z=3,**style):
 n=P.rect(x,y,w,h,fill=fill or '#edf5fc',stroke=c,sw=2,radius=12 if kind=='rectangle' else 0,z=z,**style);n['shapeKind']=kind;return n
def box(x,y,w,h,s,c=BLUE,kind='ellipse',fs=27):
 n=shape(kind,x,y,w,h,c);label(x+8,y+15,w-16,h-20,s,fs,c);return n
def conn(a,b,c=BLUE,sh=4,th=12,curve=False,dash=False,direction='forward'):
 n=P._add('connector',0,0,1,1,{'fill':'transparent','stroke':c,'textColor':INK,'fontSize':20,'borderWidth':3,'borderStyle':'dashed' if dash else 'solid','radius':0},z=25,name='native-flow')
 n['connector']={'sourceNodeId':a['id'],'sourceHook':sh,'targetNodeId':b['id'],'targetHook':th,'curve':curve,'direction':direction,'arrowSize':'medium'};return n
def dot(x,y,c=BLUE,r=7,kind='ellipse'):return shape(kind,x-r,y-r,2*r,2*r,c,c,z=7)
def star(x,y,r=13,c=ORANGE):return dot(x,y,c,r,'star')
def flow(items,y=315,x=60,w=1160,h=115,fs=26):
 gap=65;bw=(w-gap*(len(items)-1))/len(items);ns=[]
 for j,item in enumerate(items):
  s,c=item if isinstance(item,tuple) else (item,K.PAL[j%4]);ns.append(box(x+j*(bw+gap),y,bw,h,s,c,fs=fs))
 for a,b in zip(ns,ns[1:]):conn(a,b,GREY)
 return ns

def axes(x=110,y=260,w=1060,h=280,xr=(0,1),yr=(-1.15,1.15),xt=(0,.5,1),yt=(-1,0,1),xlab='x',ylab=None):
 fx=lambda a:x+(a-xr[0])/(xr[1]-xr[0])*w;fy=lambda b:y+h-(b-yr[0])/(yr[1]-yr[0])*h
 P.rect(x,y,w,h,fill='#f1f6fb',gradientTo='#fbfcfe',gradientType='radial',stroke=GREY,sw=1,fillOpacity=.7,z=1,shadowColor='#17324d',shadowBlur=8,shadowOpacity=.08)
 for a in xt:
  P.line([(fx(a),y),(fx(a),y+h)],GREY,1,dash=True,z=2);eq(fx(a)-35,y+h+3,70,f'{a:g}',24,h=42,c=MUT)
 for b in yt:
  P.line([(x,fy(b)),(x+w,fy(b))],GREY,1,dash=True,z=2);eq(x-65,fy(b)-22,55,f'{b:g}',24,h=44,c=MUT)
 if xlab:eq(x+w-20,y+h+37,70,xlab,24,h=45,c=MUT)
 if ylab:eq(x-45,y-45,180,ylab,24,h=45,c=MUT)
 return fx,fy

def line_data(xs,ys,fx,fy,c=BLUE,sw=3,dash=False):
 pts=[(fx(float(a)),fy(float(b))) for a,b in zip(xs,ys)]
 for j in range(0,len(pts)-1,180):P.line(pts[j:j+181],c,sw,dash=dash,z=5)

def sineplot(n=None,x=110,y=290,w=1060,h=250,ideal=False,midpoints=False,only_target=False):
 fx,fy=axes(x,y,w,h)
 line_data(XS,np.sin(2*np.pi*XS),fx,fy,PURPLE)
 if n:
  t=np.linspace(0,1,n+1);v=np.sin(2*np.pi*(t[:-1]+t[1:])/2)
  if ideal:
   for i in range(n):P.line([(fx(t[i]),fy(v[i])),(fx(t[i+1]),fy(v[i]))],BLUE,4,z=5)
  elif not only_target:line_data(XS,approx(n),fx,fy,BLUE,3.5)
  if midpoints:
   for j in range(n):
    P.line([(fx(t[j]),y),(fx(t[j]),y+h)],ORANGE,1,dash=True,z=3);star(fx((t[j]+t[j+1])/2),fy(v[j]),9)
 return fx,fy

def textrows(rows,x=80,y=200,w=1120,dy=110,size=27):
 for j,(s,c) in enumerate(rows):label(x,y+j*dy,w,90,s,size,c)

def stamp(x,y,s):
 n=P.rect(x,y,215,48,fill='#fff1d4',stroke=ORANGE,sw=2,radius=7,z=8);n['transform']['rotation']=-5
 t=label(x,y+5,215,40,s,22,ORANGE);t['transform']['rotation']=-5

def draw(p):
 k=p['kind'];M=p['math']
 if k=='intro':
  # A conceptual sketch, not a training experiment.
  fx,fy=axes(100,200,350,280);line_data(np.linspace(0,1,121),np.sin(2*np.pi*np.linspace(0,1,121)),fx,fy,PURPLE)
  a=box(560,265,250,125,tr('一層隱藏層','One hidden layer'),BLUE,fs=29);b=box(960,265,250,125,tr('存在逼近','A fit exists'),GREEN,fs=28);conn(a,b,GREEN)
  eq(100,540,350,r'f\in C([0,1])',28,h=80,c=PURPLE)
  st=len(P.nodes);label(500,465,700,80,tr('怎麼找到參數？\n留給學習演算法。','How do we find parameters?\nThat requires a learning method.'),28,ORANGE);stamp(860,565,tr('表示，不是訓練','REPRESENTATION'));phase2(st)
 elif k=='network':
  ins=[box(60,210+j*110,130,80,'$'+s+'$',BLUE,fs=28) for j,s in enumerate(['x_1','x_2','x_n'])]
  hs=[box(520,200+j*110,180,85,'$'+r'\sigma(z_{'+['1','2','N'][j]+r'})'+'$',PURPLE,fs=27) for j in range(3)];end=box(1040,345,170,90,r'$F(\mathbf x)$',GREEN,fs=28)
  for a in ins:
   for b in hs:conn(a,b,BLUE)
  for a in hs:conn(a,end,GREEN)
  st=len(P.nodes);eq(215,535,850,M[0],29,h=110);phase2(st)
  label(200,165,350,65,tr('輸入權重與偏置','Input weights and biases'),24,BLUE);label(745,240,280,50,tr('輸出權重','Output weights'),24,GREEN);eq(745,160,475,r'z_i=\mathbf w_i^\top\mathbf x+\theta_i',26,h=90,c=PURPLE)
 elif k=='conditions':
  fx,fy=axes(100,245,480,290,xr=(-6,6),yr=(-.05,1.05),xt=(-6,0,6),yt=(0,1),xlab='s');xx=np.linspace(-6,6,181);line_data(xx,sigmoid(xx),fx,fy,PURPLE,4)
  eq(95,155,500,M[1],29,h=100)
  for j,s in enumerate([tr('連續：沒有跳躍','Continuous: no jumps'),tr('有界：輸出有限','Bounded: finite range'),tr('非常數：能有不同反應','Nonconstant: responses can differ')]):
   dot(700,235+j*105,GREEN,12,'diamond');text(745,210+j*105,470,75,s,27,INK,True)
  st=len(P.nodes);eq(620,520,590,M[0],27,h=85);phase2(st)
 elif k=='quantifiers':
  flow([(r'$f$',PURPLE),(r'$\varepsilon>0$',ORANGE),(r'$N,\alpha,\mathbf w,\theta$',BLUE),(r'$\|f-F\|_\infty<\varepsilon$',GREEN)],y=265,h=100,fs=27)
  st=len(P.nodes);eq(65,435,1150,M[0],31,h=90);eq(65,535,1150,M[1],31,h=100);phase2(st)
 elif k=='sup':
  fx,fy=axes(110,280,700,265);ys=np.sin(2*np.pi*XS);delta=.13*(1+np.cos(2*np.pi*XS))/2
  line_data(XS,ys,fx,fy,PURPLE);line_data(XS,ys+delta,fx,fy,BLUE)
  for a in [.1,.25,.4,.6,.8]:P.line([(fx(a),fy(math.sin(2*math.pi*a))),(fx(a),fy(math.sin(2*math.pi*a)+.13*(1+math.cos(2*math.pi*a))/2))],ORANGE,3,z=6)
  eq(70,155,1140,M[0],28,h=110)
  st=len(P.nodes);eq(810,325,400,M[1],25,h=120);label(825,500,380,75,tr('示意：紫是目標，\n藍是近似。','Schematic: purple target,\nblue approximation.'),24,MUT);phase2(st)
 elif k=='basis':
  for j in [0,1]:
   yy=210+j*205
   for i in range(3):
    fx,fy=axes(80+i*195,yy,155,120,xr=(-2,2),yr=(-1.1,1.1),xt=(),yt=(),xlab=None)
    xx=np.linspace(-2,2,121);line_data(xx,np.sin((i+1)*xx) if j==0 else sigmoid(4*(xx-(i-1)*.8)),fx,fy,PURPLE if j==0 else BLUE,3)
   a=box(735,yy+15,160,80,tr('加權和','Weighted sum'),BLUE,fs=24);b=box(1030,yy+15,180,80,tr('逼近函數','Approximate'),GREEN,fs=24);conn(a,b,GREEN)
  st=len(P.nodes);eq(60,540,1160,M[0],30,h=100);phase2(st)
 elif k in ['cells','heights']:
  sineplot(5,y=295,h=245,ideal=k=='heights',midpoints=True,only_target=k=='cells')
  if k=='cells':
   # Hide the approximation: this page introduces the partition, not the completed fit.
   pass
  eq(70,150,1140,M[0],28,h=90)
  st=len(P.nodes);eq(70,585,1140,M[1],28,h=45);phase2(st)
 elif k=='steep':
  fx,fy=axes(110,300,1060,230,xr=(.45,.55),yr=(-.05,1.05),xt=(.45,.5,.55),yt=(0,.5,1))
  for steep,c in [(20,GREEN),(2000,BLUE)]:
   xx=np.linspace(.45,.55,1001);line_data(xx,sigmoid(steep*(xx-.5)),fx,fy,c,4)
  P.line([(fx(.45),fy(0)),(fx(.5),fy(0))],GREY,3,dash=True,z=6);P.line([(fx(.5),fy(1)),(fx(.55),fy(1))],GREY,3,dash=True,z=6)
  eq(70,155,750,M[0],31,h=110);eq(825,170,370,M[1],29,h=95)
  st=len(P.nodes);label(120,565,1030,55,tr('綠：陡度二十　藍：二千　虛線：理想跳躍','Green: steepness 20; blue: 2000; dashed: ideal jump'),25,MUT);phase2(st)
 elif k=='jumps':
  n=5;t=np.arange(6)/5;v=np.sin(2*np.pi*(np.arange(5)+.5)/5);fx=lambda j:150+j*215;fy=lambda a:440-a*110
  for j in range(5):
   prev=0 if j==0 else v[j-1];val=v[j];top=min(fy(prev),fy(val));hh=abs(fy(prev)-fy(val));P.rect(fx(j),top,90,max(hh,1),fill=BLUE if val>=prev else ORANGE,z=3);eq(fx(j)-25,275,145,rf'v_{j+1}',27,h=55)
   if j<4:P.line([(fx(j)+90,fy(val)),(fx(j+1),fy(val))],GREY,2,dash=True,z=4)
  eq(65,155,1140,M[0]+r',\quad '+M[1],29,h=105)
  st=len(P.nodes);eq(65,545,1140,M[2],29,h=110);phase2(st)
 elif k=='parameters':
  for j,s in enumerate([r'$\sigma(k(x-t_1))$',r'$\sigma(k(x-t_2))$',r'$\sigma(0)$']):
   a=box(100,200+j*125,330,85,s,PURPLE,fs=28);b=box(800,200+j*125,350,85,[r'$\alpha_1$',r'$\alpha_2$',r'$2v_1$'][j],GREEN,fs=30);conn(a,b,GREEN)
  eq(65,540,1140,M[0],30,h=100)
  st=len(P.nodes);eq(440,310,350,M[1],26,h=110);phase2(st)
 elif k in ['curve5','curve20']:
  n=5 if k=='curve5' else 20;sineplot(n,y=300,h=225);eq(100,250,200,r'f(x)',25,h=50,c=PURPLE);eq(325,250,230,r'F_{N,2000}(x)',25,h=50,c=BLUE)
  eq(55,150,1170,M[0],28,h=115);eq(80,585,1110,M[1],25,h=45)
  err=float(np.max(np.abs(np.sin(2*np.pi*XS)-approx(n))))
  st=len(P.nodes);eq(600,250,590,rf'\max_{{x\in G}}|f(x)-F_{{{n},2000}}(x)|\approx{err:.6f}',25,h=90,c=ORANGE);phase2(st)
 elif k=='errors':
  errs=[float(np.max(np.abs(np.sin(2*np.pi*XS)-approx(n)))) for n in NS]
  fx,fy=axes(130,305,650,235,xr=(math.log10(3),math.log10(80)),yr=(-1.5,0),xt=(),yt=(),xlab=None)
  pts=[(fx(math.log10(n)),fy(math.log10(e))) for n,e in zip(NS,errs)];P.line(pts,BLUE,4,z=5)
  for (a,b),n in zip(pts,NS):dot(a,b,BLUE,7);eq(a-35,550,70,str(n),24,h=45)
  for v in [.04,.1,.3,.8]:
   yy=fy(math.log10(v));P.line([(130,yy),(780,yy)],GREY,1,dash=True,z=2);eq(50,yy-20,70,str(v),24,h=45)
  label(230,590,450,40,tr('雙對數座標','Log-log axes'),24,MUT)
  eq(60,155,1160,M[0],27,h=100);eq(60,225,1160,M[1],24,h=65,c=MUT)
  for j,(n,e) in enumerate(zip(NS,errs)):eq(820,280+j*51,385,rf'N={n}:\quad e_G\approx{e:.6f}',24,h=50,c=BLUE)
 elif k=='bound':
  # Added exact single-cell illustration: interval [0,.1], midpoint .05.
  label(100,220,435,50,tr('示例：十格中的第一格','Example: first of ten cells'),24,MUT)
  fx,fy=axes(110,280,425,255,xr=(0,.1),yr=(0,.7),xt=(0,.05,.1),yt=(0,.6),xlab=None)
  xx=np.linspace(0,.1,181);line_data(xx,np.sin(2*np.pi*xx),fx,fy,PURPLE,4)
  vm=math.sin(.1*math.pi);P.line([(110,fy(vm)),(535,fy(vm))],BLUE,4,z=5);star(fx(.05),fy(vm),12)
  P.poly([(fx(.05),fy(vm)),(fx(.1),fy(vm)),(fx(.1),fy(vm+math.pi/10))],fill='#f9e2bf',fillOpacity=.45,stroke=ORANGE,sw=2,z=4)
  eq(570,180,640,M[0],27,h=110);eq(570,335,640,M[1],30,h=115)
  st=len(P.nodes);eq(55,585,1170,M[2],26,h=45,c=GREEN);phase2(st)
 elif k=='budget':
  flow([r'$0.01$',r'$N>100\pi$',r'$N\approx320$'],y=210,h=100,fs=31)
  for j,s in enumerate(M):eq(80,340+j*90,1120,s,25 if j==2 else 29,h=95,c=GREEN if j==1 else ORANGE if j==2 else INK)
  eq(80,590,1120,r'G=\{j/4000:j=0,\ldots,4000\}',24,h=45,c=MUT)
 elif k=='dense':
  P.ellipse(400,370,280,175,fill='#edf3fb',stroke=BLUE,sw=3,z=2,shadowColor='#17324d',shadowBlur=8,shadowOpacity=.1)
  P.ellipse(560,350,125,100,fill='#ebf6ed',stroke=GREEN,sw=2,dash=True,z=3);star(560,350,16,PURPLE)
  for a,b in [(270,310),(410,420),(510,385),(580,310),(650,350),(340,355)]:dot(a,b,BLUE,8)
  label(140,170,550,70,tr('示意：函數空間','Schematic: function space'),27,MUT)
  eq(700,220,505,M[0],27,h=120);eq(700,430,505,M[1],29,h=110,c=GREEN)
 elif k=='hb':
  a=box(70,300,285,125,r'$\overline{\mathcal H_\sigma}$',BLUE,fs=30);b=box(510,300,220,125,r'$L\ne0$',ORANGE,'diamond',34);c=box(890,300,320,125,r'$L(h)=0$',GREEN,fs=32);conn(a,b);conn(b,c,GREEN)
  eq(65,175,1150,M[0],32,h=105)
  st=len(P.nodes);eq(65,510,1150,M[1],31,h=110);phase2(st)
 elif k=='riesz':
  ns=flow([r'$f$',r'$L$',r'$\int f\,d\mu$',tr('一個數','One number')],y=240,h=105,fs=29)
  eq(65,380,1150,M[0],33,h=110)
  st=len(P.nodes);eq(65,525,1150,M[1],29,h=110);phase2(st)
 elif k=='contradiction':
  ns=flow([tr('每個單元都積成零','All unit integrals zero'),tr('判別性結果','Discriminatory result'),r'$\mu=0$',r'$L=0$'],y=290,h=110,fs=25)
  eq(65,155,1150,M[0],29,h=110)
  st=len(P.nodes);eq(65,510,1150,M[1],31,h=110,c=ORANGE);phase2(st)
 elif k=='limits':
  dot(640,225,ORANGE,17,'triangle')
  center=box(475,325,330,110,tr('存在一組好參數','A good fit exists'),GREEN,fs=29)
  for (x,y,s) in [(65,175,tr('需要多少單元？','How many units?')),(890,175,tr('怎麼找到參數？','How to find weights?')),(65,490,tr('未見資料會好嗎？','Will it generalize?')),(890,490,tr('能到全域最佳嗎？','A global optimum?'))]:
   n=box(x,y,325,100,s,ORANGE,fs=25);conn(center,n,ORANGE,sh=14 if x<400 else 2,th=4 if x<400 else 12,dash=True)
 elif k.startswith('myth') and k.endswith('q'):
  n=box(435,220,410,150,'?',ORANGE,'diamond',60)
  ns=[box(85,455,405,110,tr('先寫出你的理由','Write your reason first'),BLUE,fs=28),box(790,455,405,110,tr('翻頁後再核對','Check on the next page'),BLUE,fs=28)]
  for a in ns:conn(n,a,BLUE,sh=10 if a==ns[0] else 6,th=0)
 elif k=='myth1a':
  flow([(tr('{color=#168269}存在{/color}','{color=#168269}Exists{/color}'),GREEN),(tr('找到','Find'),ORANGE),(tr('泛化','Generalize'),ORANGE)],y=285,h=130,fs=34)
  label(90,500,1100,85,tr('後兩個箭頭都需要額外證據，不是定理自動給的。','Both later arrows need additional evidence; the theorem does not supply them.'),28,MUT)
 elif k=='myth2a':
  fx,fy=axes(110,260,580,275,xr=(-2,2),yr=(-.1,2.1),xt=(-2,0,2),yt=(0,1,2));xx=np.linspace(-2,2,181);line_data(xx,np.maximum(0,xx),fx,fy,BLUE,4)
  eq(720,220,480,M[0],32,h=110);label(720,375,480,150,tr('連續，但無界。\n不是全域多項式。','Continuous but unbounded.\nNot a global polynomial.'),29,GREEN)
 elif k=='myth3a':
  for j,(s,n) in enumerate([(tr('淺層：可能很寬','Shallow: may be wide'),8),(tr('深層：重用組合','Deep: reuse composition'),3)]):
   x=145+j*650;label(x-70,180,550,60,s,29,BLUE)
   for a in range(n):
    yy=275+a*35 if j==0 else 290+a*90;dot(x+50 if j==0 else x+a*65,yy,BLUE,12)
   if j==1:
    nodes=[box(x-30+a*140,305,85,85,'$h_'+str(a+1)+'$',PURPLE,fs=27) for a in range(3)]
    for a,b in zip(nodes,nodes[1:]):conn(a,b,PURPLE)
  label(80,560,1120,55,tr('示意：某些組合函數有深度效率優勢，並非每個目標。','Schematic: some compositional functions benefit from depth; not every target.'),25,MUT)
 elif k=='timeline':
  years=[1943,1958,1960,1982,1986,1988];names=['M-P','Perceptron','ADALINE','SOM','MLP','RBF'];mechs=[('固定邏輯','Fixed logic'),('錯分更新','Mistake updates'),('平方損失','Squared loss'),('無標籤原型','Unlabeled prototypes'),('多層與反傳','Layers and backprop'),('局部基函數','Local bases')]
  # Ordinal historical sequence: positions deliberately do not encode elapsed years.
  for j,(year,name,mech) in enumerate(zip(years,names,mechs)):
   xx=80+j*200;n=box(xx,300,170,90,name,BLUE if j%2==0 else PURPLE,fs=25 if name!='Perceptron' else 22)
   eq(xx-5,215,180,str(year),28,h=65);label(xx-15,420,200,85,tr(*mech),24,MUT)
   if j:conn(prev,n,GREY)
   prev=n
  label(60,555,1160,70,tr('按事件順序排列，水平間距不代表年數；MLP 的一九八六年指反傳普及。','Event order, not a proportional year axis; MLP’s 1986 marks backpropagation’s popularization.'),25,MUT)
 elif k=='practice1q':
  fx,fy=axes(110,275,660,270,yr=(0,1),xt=(0,.25,.5,.75,1),yt=(0,.5,1));line_data(np.linspace(0,1,101),np.linspace(0,1,101),fx,fy,PURPLE)
  for j in range(4):
   xx=(j+.5)/4;box(fx(xx)-28,fy(xx)-34,56,68,'?',BLUE,fs=26)
  eq(795,230,420,M[0],28,h=110);eq(785,410,430,M[1],25,h=125)
 elif k=='practice1a':
  fx,fy=axes(110,275,575,250,yr=(0,1),xt=(0,.25,.5,.75,1),yt=(0,.5,1));line_data(np.linspace(0,1,101),np.linspace(0,1,101),fx,fy,PURPLE)
  for j in range(4):
   v=(j+.5)/4;P.line([(fx(j/4),fy(v)),(fx((j+1)/4),fy(v))],BLUE,4,z=5);star(fx(v),fy(v),9)
  eq(710,230,500,M[0],25,h=110);eq(710,425,500,M[2],26,h=95)
  st=len(P.nodes);eq(55,585,1170,M[1],25,h=45);phase2(st)
 elif k=='practice1check':
  fx,fy=axes(110,305,620,220,yr=(0,1),xt=(0,.25,.5,.6,.75,1),yt=(0,.5,1));xx=np.linspace(0,1,1001);ff=.125+.25*sum(sigmoid(20*(xx-t)) for t in [.25,.5,.75]);line_data(xx,ff,fx,fy,BLUE)
  for j in range(4):P.line([(fx(j/4),fy((j+.5)/4)),(fx((j+1)/4),fy((j+.5)/4))],GREY,3,dash=True,z=4)
  dot(fx(.6),fy(.625),ORANGE,9,'diamond');eq(110,250,620,r'k=20',25,h=50,c=BLUE)
  eq(55,150,1170,M[0],25,h=115);eq(745,330,465,M[1],26,h=115)
  st=len(P.nodes);eq(65,585,1150,M[2],28,h=45,c=GREEN);phase2(st)
 elif k=='practice2q':
  flow([r'$\varepsilon=0.05$',r'$\pi/N<0.05$',r'$N=?$'],y=280,h=125,fs=31)
  eq(65,165,1150,M[0],29,h=110);eq(65,500,1150,M[1],32,h=110)
 elif k=='practice2a':
  flow([r'$N>20\pi$',r'$62.831853\ldots$',r'$N=63$'],y=255,h=115,fs=31)
  eq(65,415,1150,M[0],31,h=110);eq(65,530,1150,M[1],31,h=110,c=GREEN)
 elif k in ['practice3q','practice3a']:
  flow([(tr('表示','Represent'),GREEN),(tr('搜尋參數','Search weights'),ORANGE),(tr('未見資料','Unseen data'),BLUE)],y=280,h=140,fs=32)
  label(70,505,1140,95,tr('存在？　找到？　泛化？' if k.endswith('q') else '每一關都要有自己的條件與驗證。','Exists? Found? Generalizes?' if k.endswith('q') else 'Each stage needs its own assumptions and evidence.'),31,MUT)
 elif k.startswith('quiz') and k.endswith('q'):
  for j,(zh,en) in enumerate(p['options']):
   x=70+j%2*600;y=205+j//2*210;dot(x+28,y+25,PURPLE,25);label(x,y+3,56,50,chr(65+j),27,'#ffffff');text(x+75,y,495,130,tr(zh,en),26 if LANG=='zh' else 24,INK,True)
   P.line([(x+75,y+150),(x+520,y+150)],GREY,1,dash=True,z=2)
 elif k.startswith('quiz') and k.endswith('a'):
  n=int(k[4]);answer=chr(65+p['answer']);a=box(75,270,190,155,answer,GREEN,'diamond',54)
  if n==1:items=[tr('連續','Continuous'),tr('有界','Bounded'),tr('非常數','Nonconstant')]
  elif n==2:items=[r'$L\ne0$',r'$L(h)=0$',r'$L\ \text{continuous}$']
  elif n==3:items=[tr('存在','Exists'),tr('非搜尋法','No search method'),tr('非訓練保證','No training guarantee')]
  else:items=[tr('生成函數','Generating functions'),tr('加權組合','Weighted sum'),tr('逼近目標','Approximate target')]
  ns=flow(items,y=290,x=410,w=800,h=105,fs=24);conn(a,ns[0],GREEN)
  label(90,510,1100,85,tr('參考解析依講義推得；逐項理由請開啟自學導讀。','Suggested explanation from the handout; open the study notes for option-by-option reasoning.'),27,MUT)
 elif k=='summary':
  flow([tr('條件','Assumptions'),tr('構造','Construction'),tr('稠密','Density'),tr('限制','Limits')],y=285,h=130,fs=31)
  eq(65,475,1150,M[0],30,h=125)
 elif k=='next':
  ns=flow([tr('目標與損失','Target and loss'),tr('反傳梯度','Backprop gradients'),tr('更新權重','Update weights')],y=275,h=140,fs=29);conn(ns[-1],ns[0],ORANGE,sh=8,th=8,curve=True)
  st=len(P.nodes);eq(65,510,1150,M[0],33,h=110,c=GREEN);phase2(st)
 else:raise ValueError(k)

def build(lang):
 global P,LANG,SECOND
 LANG=lang;K.LANG=lang;title=tr('KP10 萬有逼近定理｜圖解自學版','KP10 Universal approximation — Diagram self-study')
 d=K.make(title,title,'kp10');sid='scene-kp10';now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['metadata'].update(createdAt=now,updatedAt=now)
 manifests=[]
 for i,p in enumerate(PAGES):
  P=K.Page(prefix=f'{lang}-p{i+1:02d}');K.P=P;SECOND=[]
  P.rect(0,0,1280,720,fill='#ffffff',z=0);P.rect(0,0,1280,86,fill='#f5f8fc',gradientTo='#ffffff',gradientStops=[{'offset':.5,'color':'#f0f5fa'}],z=1)
  text(45,16,1000,82,tr(p['title'],p['title_en']),34 if lang=='zh' else 32,INK,True,name='page-title')
  text(1050,34,185,40,f'KP10 · {i+1}/{len(PAGES)}',18,MUT,False,'right')
  text(50,108,1180,48,tr(p['bridge'],p['bridge_en']),20,MUT,name='bridge')
  start=len(P.nodes);draw(p)
  main=[n['id'] for n in P.nodes[start:]]
  # In conceptual pages, reveal at least the main conclusion-associated labels in beat two.
  if not SECOND:
   candidates=[n for n in P.nodes[start:] if n['type']=='text']
   if candidates:SECOND=[candidates[-1]['id']]
  first=[nid for nid in main if nid not in SECOND]
  ft=len(P.nodes);P.rect(50,640,1180,70,fill=BAN,radius=12,z=7,shadowColor='#17324d',shadowBlur=9,shadowOpacity=.12)
  text(70,643,1140,64,tr(p['takeaway'],p['takeaway_en']),24,'#ffffff',True,'center',name='takeaway')
  groups=[first,SECOND,[n['id'] for n in P.nodes[ft:]]]
  P.check();assert not P.warnings,P.warnings
  for n in P.nodes:
   n['sceneId']=sid;n['transform']['x']+=i*1440;n['initiallyHidden']=n['id'] in main+groups[-1]
   if n['type']=='text':n['style']['fontFamily']='Noto Sans CJK TC' if lang=='zh' else 'Arial'
   d['nodes'][n['id']]=n;d['scenes'][sid]['nodeIds'].append(n['id'])
  fs=len(d['steps']);full=tr(*p['notes']);take=tr(p['takeaway'],p['takeaway_en']);bridge=tr(p['bridge'],p['bridge_en']);maths='\n\n'.join('$'+s+'$' for s in p['math'])
  # All notes are complete explanatory prose, not a generic animation instruction.
  notes=[bridge+'\n\n'+full,tr('對照圖中的標註與算式，逐項核對：','Use the visual labels and equations to check each step:')+'\n\n'+full+('\n\n'+maths if maths else ''),take+'\n\n'+full+tr('\n\n讀完請用自己的話重述這一頁，再前進。','\n\nRestate this page in your own words before moving on.')]
  for beat,ids in enumerate(groups):
   d['steps'].append(dict(id=f'{lang}-p{i+1}-beat{beat+1}',order=len(d['steps']),label=tr(p['title'],p['title_en'])+f' · {beat+1}',durationMs=550,holdMs=500,easing='smooth',target=dict(kind='scene-point',sceneId=sid,point=dict(x=i*1440+640,y=360),zoom=1,referenceViewport=dict(width=1280,height=720)),reveal=dict(timing='after-camera',enter=[dict(nodeId=nid,effect='fade',delayMs=0,durationMs=280) for nid in ids],exitNodeIds=[]),speakerNotes=notes[beat]))
  manifests.append(dict(page=i+1,kind=p['kind'],title=tr(p['title'],p['title_en']),first_step=fs,final_step=len(d['steps'])-1,beats=3,source_anchors=p['source_anchors']))
 out=ROOT/'output';out.mkdir(exist_ok=True);stem='lec02-kp10-universal-approximation'+('-en' if lang=='en' else '')
 (out/(stem+'.jbd.json')).write_text(json.dumps(d,ensure_ascii=False,indent=1));(out/(lang+'-pages.json')).write_text(json.dumps(manifests,ensure_ascii=False,indent=1));print(lang,len(PAGES),'pages;',len(d['steps']),'steps;',len(d['nodes']),'nodes')
if __name__=='__main__':save();build('zh');build('en')
