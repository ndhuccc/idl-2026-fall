from pathlib import Path
import json,datetime,math
import pagekit as K
import mascot
from specs import PAGES
from kp06_nums import *
ROOT=Path(__file__).parent;P=None;LANG='zh';SECOND=[];TEMP=[]
INK,MUT,BLUE,PURPLE,GREEN,ORANGE,GREY,RED=K.INK,K.MUT,K.BLUE,K.PURPLE,K.GREEN,K.ORANGE,K.GREY,K.RED
text,label,eq=K.text,K.label,K.eq
tr=lambda a,b:a if LANG=='zh' else b

def phase2(start):SECOND.extend(n['id'] for n in P.nodes[start:])
def badge(s):text(50,110,950,46,s,21,MUT)
def shape(kind,x,y,w,h,c=BLUE,fill=None,z=3,**style):
 n=P.rect(x,y,w,h,fill=fill or '#edf5fc',stroke=c,sw=2,radius=12 if kind=='rectangle' else 0,z=z,**style);n['shapeKind']=kind;return n

def box(x,y,w,h,s,c=BLUE,kind='rectangle',fs=25):
 n=shape(kind,x,y,w,h,c);lh=h-16;label(x+8,y+8,w-16,lh,s,fs,c);return n

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

FORMULAS={
'linear':(r'f(a)=a',r'f(a)\in\mathbb R',r'f^{\prime}(a)=1'),
'sigmoid':(r's(a)=\frac{1}{1+e^{-a}}',r's(a)\in(0,1)',r's^{\prime}(a)=s(a)(1-s(a))'),
 'tanh':(r'f(a)=\frac{e^a-e^{-a}}{e^a+e^{-a}}',r'f(a)\in(-1,1)',r'f^{\prime}(a)=1-f(a)^2'),
 'gaussian':(r'f(a)=e^{-a^2}',r'f(a)\in(0,1]',r'f^{\prime}(a)=-2ae^{-a^2}'),
 'relu':(r'f(a)=\max(0,a)',r'f(a)\in[0,\infty)',r'a>0:\ f^{\prime}=1;\ a<0:\ f^{\prime}=0'),
 'leaky':(r'a\ge0:\ f=a;\ a<0:\ f=0.01a',r'f(a)\in\mathbb R',r'a>0:\ f^{\prime}=1;\ a<0:\ f^{\prime}=0.01'),
 'prelu':(r'a\ge0:\ f=a;\ a<0:\ f=\alpha a',r'\alpha>0\Rightarrow f(a)\in\mathbb R',r'a>0:\ f^{\prime}=1;\ a<0:\ f^{\prime}=\alpha'),
 'elu':(r'a\ge0:\ f=a;\ a<0:\ f=\alpha(e^a-1)',r'\alpha>0:\ f(a)\in(-\alpha,\infty)',r'a>0:\ f^{\prime}=1;\ a<0:\ f^{\prime}=\alpha e^a'),
 'softplus':(r'f(a)=\ln(1+e^a)',r'f(a)\in(0,\infty)',r'f^{\prime}(a)=\frac{1}{1+e^{-a}}'),
}
def plot(keys,x=105,y=215,w=450,h=320,xmin=-3,xmax=3,ymin=-1.3,ymax=3.2,der=False):
 fx=lambda v:x+(v-xmin)/(xmax-xmin)*w
 fy=lambda v:y+h-(v-ymin)/(ymax-ymin)*h
 P.rect(x,y,w,h,fill='#edf4fa',stroke=GREY,sw=1,fillOpacity=.48,z=1)
 for av in [xmin,0,xmax]:
  if xmin<=av<=xmax:eq(fx(av)-30,y+h+5,60,f'{av:g}',18,h=32)
 for v in [ymin,0,ymax]:
  if ymin<=v<=ymax:eq(x-65,fy(v)-16,55,f'{v:g}',17,h=30)
 if ymin<=0<=ymax:P.line([(x,fy(0)),(x+w,fy(0))],GREY,2,z=2)
 if xmin<=0<=xmax:P.line([(fx(0),y),(fx(0),y+h)],GREY,2,z=2)
 eq(x+w-30,y+h+35,60,'a',20,h=32)
 eq(x-30,y-45,100,'f(a)',21,h=34)
 for j,key in enumerate(keys):
  fn=FUN[key] if isinstance(key,str) else key
  pts=[(fx(a),fy(fn(a))) for a in [xmin+(xmax-xmin)*i/599 for i in range(600)]]
  assert all(x-.01<=a<=x+w+.01 and y-.01<=b<=y+h+.01 for a,b in pts),(keys,ymin,ymax)
  for start in range(0,599,180):P.line(pts[start:start+181],K.PAL[j%4],4,z=4)
 if der:
  st=len(P.nodes);pts=[(fx(a),fy(DER[keys[0]](a))) for a in [xmin+(xmax-xmin)*i/599 for i in range(600)]]
  for start in range(0,599,180):P.line(pts[start:start+181],PURPLE,3,dash=True,z=4)
  label(x+20,y-40,w-20,32,tr('實線：輸出　虛線：導數','Solid: output   Dashed: derivative'),19,MUT);phase2(st)
 return fx,fy

def curvepage(kind):
 badge(tr('精確函數曲線｜橫軸為活化前的分數','Exact curves | Horizontal axis: pre-activation score'))
 ranges={'linear':(-3.3,3.3),'sigmoid':(-.08,1.08),'tanh':(-1.2,1.2),'gaussian':(-1,1.2),'relu':(-.3,3.2),'leaky':(-.3,3.2),'prelu':(-1.6,3.2),'elu':(-1.3,3.2),'softplus':(-.2,3.3)}
 lo,hi=ranges[kind]
 keys=[kind]
 if kind=='prelu':keys=[lambda a:max(.01*a,a),lambda a:max(.2*a,a),lambda a:max(.5*a,a)]
 fx,fy=plot(keys,ymin=lo,ymax=hi,der=kind in ['sigmoid','tanh','gaussian'])
 forms=FORMULAS[kind]
 eq(595,190,410,forms[0],23 if kind in ['prelu','elu','leaky'] else 28,h=110)
 eq(590,325,420,forms[1],25,h=70,c=BLUE)
 st=len(P.nodes);eq(590,420,420,forms[2],22 if kind in ['prelu','elu','leaky','relu'] else 26,h=110,c=PURPLE)
 if kind=='sigmoid':
  star(fx(0),fy(.5),10,ORANGE);eq(595,550,410,r's(0)=\frac12,\ s^{\prime}(0)=\frac14',25,h=70,c=ORANGE)
 if kind=='tanh':eq(595,550,410,r'f(0)=0,\ f^{\prime}(0)=1',25,h=60,c=ORANGE)
 if kind=='gaussian':star(fx(0),fy(1),10,ORANGE);label(100,590,450,35,tr('補充釐清：鐘形，不是 S 形','Added: bell-shaped'),22,ORANGE)
 if kind in ['relu','leaky','prelu']:
  shape('triangle',fx(0)-12,fy(0)-12,24,24,ORANGE,'#fff3d8',z=7)
  label(100,590,460,36,tr('零點採實作約定，不是一般導數','Zero: a coding convention'),20,ORANGE)
 if kind=='prelu':eq(95,160,470,r'\alpha=0.01,\ 0.2,\ 0.5',24,h=42);label(595,552,410,65,tr('補充：非正參數時值域會改變','Added: nonpositive alpha changes the range'),21,ORANGE)
 if kind=='elu':
  P.line([(105,fy(-1)),(555,fy(-1))],ORANGE,2,dash=True,z=3)
  eq(95,160,470,r'\alpha=1,\quad\lim_{a\to-\infty}f(a)=-1',23,h=45)
  eq(590,550,425,r'f^{\prime}(0^-)=\alpha,\ f^{\prime}(0^+)=1',23,h=60,c=ORANGE)
 if kind=='softplus':
  P.line([(fx(-3),fy(0)),(fx(0),fy(0)),(fx(3),fy(3))],GREY,2,dash=True,z=3)
  label(95,590,460,32,tr('灰虛線：ReLU 對照','Gray dashed: ReLU reference'),21,MUT)
 phase2(st)

def draw(kind):
 if kind in FORMULAS:curvepage(kind);return
 if kind=='cover':
  badge(tr('承接 KP05：非線性要能改變輸出，也要能傳回梯度','From KP05: nonlinearity shapes outputs and gradient flow'))
  ns=[box(90+j*345,210,220,95,s,K.PAL[j],'diamond' if j==1 else 'ellipse',32) for j,s in enumerate([r'$a$',r'$f$',r'$z$'])];conn(ns[0],ns[1]);conn(ns[1],ns[2],PURPLE)
  st=len(P.nodes)
  for j,k in enumerate(['sigmoid','tanh','relu']):
   plot([k],x=95+j*320,y=390,w=235,h=170,ymin=-1.3,ymax=3.2);label(95+j*320,325,235,40,k,24,BLUE)
  phase2(st)
 elif kind=='map':
  badge(tr('依講義分族；「跨層」指多候選／向量運算','Handout taxonomy | “Cross-layer” means multi-candidate/vector operations'))
  groups=[('Linear\nStep · Sign\nSoftmax',4,BLUE,'ellipse'),('Sigmoid · tanh\nGaussian',3,PURPLE,'ellipse'),('Softplus · ReLU\nLeaky · PReLU\nELU',5,GREEN,'ellipse'),('Maxout',1,ORANGE,'diamond')]
  center=box(440,315,155,110,'13',BLUE,'ellipse',46)
  for j,(s,n,c,kind_) in enumerate(groups):
   x=70+(j%2)*625;y=195+(j//2)*245
   label(x,y-34,285,32,tr(['基礎','S 形組（含例外）','ReLU 家族','多候選運算'][j],['Basic','S-group + exception','ReLU family','Multi-candidate'][j]),22,c);b=shape(kind_,x,y,285,145,c);label(x+8,y+25,269,110,s,24,c);conn(center,b,c,sh=[14,2,10,6][j],th=[4,12,4,12][j]);label(x,y+147,285,42,str(n)+tr(' 種',' function' if n==1 else ' functions'),23,c)
 elif kind=='step':
  badge(tr('零點約定：本頁取右側值；導數判定不受這個選擇改變','At zero we choose the right-hand value; the derivative remains undefined'))
  for j,(low,name) in enumerate([(0,'Binary step'),(-1,'Sign')]):
   x=100+j*485;y=230;w=370;h=250;fx=lambda a:x+(a+2)/4*w;fy=lambda v:y+h-(v+1.3)/2.6*h
   P.rect(x,y,w,h,fill='#f0f5fa',fillOpacity=.5,z=1)
   P.line([(x,fy(0)),(x+w,fy(0))],GREY,2,z=2);P.line([(fx(0),y),(fx(0),y+h)],GREY,2,z=2)
   for a,b,v in [(-2,0,low),(0,2,1)]:P.line([(fx(a),fy(v)),(fx(b),fy(v))],BLUE,5,z=4)
   for v,fill in [(low,'white'),(1,BLUE)]:P.ellipse(fx(0),fy(v),7,fill=fill,stroke=BLUE,sw=2,z=5)
   label(x,170,w,45,name,27,BLUE);eq(x-30,510,w+60,r'f(a)=\mathbf1[a\ge0]' if j==0 else r'f(a)=2\mathbf1[a\ge0]-1',24,h=45);eq(x,555,w,r'f(a)\in\{'+str(low)+r',1\}',23,h=40)
   for v in set([low,0,1]):eq(x-42,fy(v)-15,40,str(v),19,h=30)
   eq(fx(0)-20,y+h+2,40,'0',17,h=28)
  st=len(P.nodes);eq(70,600,930,r'a\ne0:\ f^{\prime}(a)=0;\qquad a=0:\ \nexists f^{\prime}(0)',25,h=38,c=PURPLE);phase2(st)
 elif kind in ['softmax','softmaxcalc']:
  badge(tr('有限分數、至少兩類；輸出可視為類別模型機率','Finite scores, at least two classes; categorical model probabilities'))
  if kind=='softmax':
   a=box(65,245,210,105,r'$\mathbf a$',BLUE,'ellipse',32);b=box(405,245,210,105,r'$e^{a_i}$',PURPLE,'ellipse',32);c=box(765,245,210,105,r'$p_i$',GREEN,'ellipse',32);conn(a,b);conn(b,c,PURPLE)
   eq(80,380,910,r'p_i=\frac{e^{a_i}}{\sum_{j=1}^{m}e^{a_j}},\quad 0<p_i<1,\quad\sum_i p_i=1',29,h=120)
   st=len(P.nodes);xx=95
   for j,p in enumerate(PROBS):P.rect(xx,535,870*p,56,fill=K.PAL[j],z=4);label(xx,542,870*p,40,str(j+1),23,'#ffffff');xx+=870*p
   phase2(st)
  else:
   eq(80,160,920,r'\mathbf a=(2,1,0),\quad S=e^2+e^1+e^0\approx11.107338',26,h=110)
   for j,e in enumerate(EXPS):
    y=295+j*95;eq(45,y,220,rf'e^{2-j}\approx{e:.4f}',23,h=70)
    P.rect(280,y+9,350*e/EXPS[0],40,fill=K.PAL[j],z=3)
   st=len(P.nodes)
   eq(650,300,345,r'\mathbf p=\frac{(e^2,e^1,e^0)}{S}',29,h=110)
   eq(650,430,345,r'\approx(0.665,0.245,0.090)',25,h=110)
   label(250,590,650,32,tr('長條：未正規化的指數；不是機率','Bars: exponentials, not probabilities'),21,MUT);phase2(st)
 elif kind=='softmaxjac':
  badge(tr('補充曲線例子：只改第一個分數；其餘兩個固定為零','Added curve example: vary only the first score; keep the others at zero'))
  plot([lambda a:math.exp(a)/(math.exp(a)+2),lambda a:1/(math.exp(a)+2)],ymin=0,ymax=1)
  eq(95,165,470,r'\mathbf a=(a,0,0)',26,h=45)
  eq(600,225,400,r'\frac{\partial p_i}{\partial a_j}=p_i(\delta_{ij}-p_j)',28,h=110)
  st=len(P.nodes);eq(600,365,400,r'i=j:\quad p_i(1-p_i)',27,h=80,c=BLUE);eq(600,465,400,r'i\ne j:\quad -p_ip_j',27,h=80,c=PURPLE)
  label(110,590,450,32,tr('藍：第一類　紫：其餘各類','Blue: first; purple: each other'),21,MUT);phase2(st)
 elif kind=='gradient':
  badge(tr('只示範五個 Sigmoid 活化導數因子，並非完整梯度上界','Five Sigmoid activation factors only; not a bound on the full gradient'))
  ns=[box(70+j*185,200,120,85,r'$\frac14$',PURPLE,'ellipse',30) for j in range(5)]
  for a,b in zip(ns,ns[1:]):conn(a,b,PURPLE)
  for j in range(5):
   x=90+j*185;h=240*.25**(j+1);bn=P.rect(x,560-h,80,h,fill=BLUE,z=4);bn['height']=h;bn['style']['borderWidth']=0;eq(x-35,565,150,r'(\frac14)^'+str(j+1),23,h=65)
  st=len(P.nodes);eq(95,335,900,r'\prod_{k=1}^{5}s^{\prime}(a_k)\le\left(\frac14\right)^5=0.0009765625',32,h=110,c=PURPLE);phase2(st)
 elif kind=='dying':
  badge(tr('概念示意｜針對這批相關訓練輸入與普通資料梯度','Concept sketch | Relevant training inputs and ordinary data gradients'))
  mid=box(405,295,205,105,'ReLU',PURPLE,'diamond',32)
  for j,a in enumerate([-3,-2,-1]):
   y=195+j*140;n=box(70,y,130,80,'$'+str(a)+'$',BLUE,'ellipse',30);conn(n,mid)
   out=box(805,y,130,80,'$0$',GREY,'ellipse',30);conn(mid,out,GREY)
  st=len(P.nodes);eq(170,555,710,r'\forall n:\ a_n<0\ \Rightarrow\ f^{\prime}(a_n)=0',28,h=70,c=RED);phase2(st)
 elif kind in ['minus','plus']:
  minus=kind=='minus';av=-2 if minus else 2
  badge(tr('程式精算｜ELU 參數為一；Leaky 負側斜率固定 0.01','Computed | ELU $\\alpha=1$; Leaky negative slope $0.01$'))
  fx,fy=plot(['relu','leaky','elu','softplus'],xmin=-3 if minus else 1,xmax=3,ymin=-1.3 if minus else .8,ymax=3.2)
  P.line([(fx(av),215),(fx(av),535)],ORANGE,2,dash=True,z=5)
  eq(100,165,450,f'a={av}',26,h=45)
  label(595,180,410,45,tr('函數值與局部斜率','Value and local slope'),23,MUT)
  st=len(P.nodes)
  for j,k in enumerate(['relu','leaky','elu','softplus']):
   n=NUMS[k][str(av)];y=260+j*85
   label(590,y,108,45,{'relu':'ReLU','leaky':'Leaky','elu':'ELU','softplus':'Softplus'}[k],21,K.PAL[j])
   valforms={'relu':rf'\max(0,{av})','leaky':rf'0.01({av})' if minus else str(av),'elu':rf'e^{{{av}}}-1' if minus else str(av),'softplus':rf'\ln(1+e^{{{av}}})'}
   dforms={'relu':'0' if minus else '1','leaky':'0.01' if minus else '1','elu':rf'e^{{{av}}}' if minus else '1','softplus':rf'\frac{{1}}{{1+e^{{{-av}}}}}'}
   eq(705,y-12,300,valforms[k]+rf'\approx{n["value"]:.3f}',22,h=55,c=K.PAL[j])
   eq(705,y+30,300,rf'f^{{\prime}}={dforms[k]}\approx{n["derivative"]:.3f}',21,h=60,c=K.PAL[j])
   P.ellipse(fx(av),fy(n['value']),7+j,fill='transparent',stroke=K.PAL[j],sw=2,z=6)
  label(595,595,410,28,tr('小數為近似；數值未先取位數','Rounded only at the end'),18,MUT);phase2(st)
 elif kind=='maxout':
  badge(tr('補充反例｜固定分支的值域未必是全實數','Added counterexample | Fixed branches need not cover every real value'))
  plot([lambda a:a,lambda a:-a,'maxout'],ymin=-3.2,ymax=3.2)
  eq(580,180,430,r'f(\mathbf x)=\max_j(\mathbf w_j^\top\mathbf x+b_j)',26,h=110)
  eq(580,315,430,r'\max(a,-a)=|a|\in[0,\infty)',26,h=75,c=GREEN)
  st=len(P.nodes);eq(580,420,430,r'\frac{\partial f}{\partial a_j}=\mathbf1[j=j^\ast]',27,h=110,c=PURPLE)
  eq(580,535,430,r'\nabla_{\mathbf x}f=\mathbf w_{j^\ast}',27,h=70,c=PURPLE);label(95,590,450,34,tr('唯一贏家時；平手可能不可微','Unique winner; beware ties'),21,ORANGE);phase2(st)
 elif kind=='hidden':
  badge(tr('常用起點；梯度仍取決於權重、資料與整體訓練設定','A common starting point; gradients also depend on weights, data and training'))
  a=box(80,315,185,110,'ReLU',BLUE,'ellipse',31);b=box(425,300,210,140,tr('負側卡住？','Stuck negative?'),ORANGE,'diamond',26);conn(a,b)
  st=len(P.nodes)
  for j,s in enumerate(['Leaky ReLU','PReLU']):c=box(785,200+j*260,200,100,s,GREEN,'ellipse',27);conn(b,c,GREEN,sh=2 if j==0 else 6,th=12)
  stamp(90,510,tr('先觀察','INSPECT FIRST'));phase2(st)
 elif kind=='output':
  badge(tr('回歸例子允許任意實數；多類別例子是互斥分類','Regression target: any real number; multiclass: mutually exclusive labels'))
  for j,(zh,en,s) in enumerate([('實數回歸','Real-valued regression','Linear'),('二元分類','Binary classification','Sigmoid'),('互斥多類別','Exclusive multiclass','Softmax')]):
   y=200+j*140;a=box(80,y,290,85,tr(zh,en),K.PAL[j],'ellipse',25);b=box(685,y,285,85,s,K.PAL[j],'ellipse',27);conn(a,b,K.PAL[j])
  st=len(P.nodes);text(110,596,880,34,tr('補充：模型機率不保證與真實頻率吻合。','Added: model probabilities need not be calibrated.'),22,ORANGE,True);phase2(st)
 elif kind=='derive':
  badge(tr('選讀推導｜以鏈式法則把函數值與導數連起來','Optional derivation | Connect function values and derivatives with the chain rule'))
  eq(65,180,535,r's^{\prime}(a)=\frac{e^{-a}}{(1+e^{-a})^2}',30,h=110)
  eq(65,335,535,r'=\frac{1}{1+e^{-a}}\frac{e^{-a}}{1+e^{-a}}',30,h=110)
  st=len(P.nodes);eq(65,490,535,r'=s(a)(1-s(a))',30,h=95,c=PURPLE)
  plot(['softplus','sigmoid'],x=660,y=265,w=320,h=230,ymin=0,ymax=3.3)
  eq(590,560,420,r'\frac{d}{da}\ln(1+e^a)=\frac{e^a}{1+e^a}=s(a)',23,h=72,c=PURPLE);phase2(st)
 elif kind=='next':
  badge(tr('概念示意｜本頁只建立資料流，損失的形式留到 KP07','Concept sketch | Establish the data flow; loss formulas come in KP07'))
  a=box(80,210,210,105,tr('預測','Prediction'),BLUE,'ellipse',30);b=box(80,465,210,105,tr('目標','Target'),PURPLE,'ellipse',30);c=box(455,320,210,125,tr('損失','Loss'),GREEN,'diamond',32);conn(a,c);conn(b,c,PURPLE)
  st=len(P.nodes);out=box(805,325,170,110,r'$L$',ORANGE,'ellipse',36);conn(c,out,GREEN);phase2(st)
 elif kind=='practice1':
  badge(tr('練習｜圖與講者備註均不揭數值答案','Practice | Neither the figure nor the notes reveals the numerical answer'))
  fx,fy=plot(['sigmoid'],ymin=0,ymax=1)
  P.line([(fx(1),215),(fx(1),535)],ORANGE,2,dash=True,z=5);eq(fx(1)-35,550,70,'1',23,h=45)
  eq(590,225,420,r's(1)=\frac{1}{1+e^{-1}}=?',31,h=110)
  st=len(P.nodes);eq(590,420,420,r's^{\prime}(1)=s(1)(1-s(1))=?',27,h=110,c=PURPLE);phase2(st)
 elif kind=='practice2':
  badge(tr('練習｜保持所有輸出候選同樣呈現','Practice | Every output candidate has equal visual styling'))
  for j in range(3):
   y=205+j*135;a=box(80,y,160,85,'$1$',BLUE,'ellipse',30);b=box(430,y,180,85,r'$e^1$',PURPLE,'ellipse',30);c=box(825,y,140,85,'?',GREEN,'ellipse',32);conn(a,b);conn(b,c,PURPLE)
 elif kind=='practice3':
  badge(tr('練習｜所有相關輸入皆為負；比較兩個單元的局部斜率','Practice | All relevant inputs are negative; compare local slopes'))
  for j,s in enumerate(['ReLU','Leaky ReLU']):
   y=240+j*235;a=box(80,y,210,90,tr('全為負數','All negative'),BLUE,'ellipse',26);b=box(450,y,220,90,s,PURPLE,'diamond',27);c=box(850,y,110,90,'?',GREEN,'ellipse',32);conn(a,b);conn(b,c,PURPLE)
 elif kind.startswith('quiz'):
  opts={
  'quiz1':[('輸出以零為中心','Zero-centered outputs'),('正側斜率一、負側稀疏','Positive slope one; sparse negative side'),('直接保證機率校準','Guaranteed probability calibration'),('負側斜率固定非零','Fixed nonzero negative slope')],
  'quiz2':[('ReLU','ReLU'),('Maxout','Maxout'),('Softmax','Softmax'),('tanh','tanh')],
  'quiz3':[('一次輸入負數就永久停止','One negative input stops it forever'),('所有相關輸入負、局部導數零','All relevant inputs negative; zero local slope'),('正側斜率變成負一','Positive-side slope becomes minus one'),('輸出機率沒有校準','Uncalibrated output probabilities')],
  'quiz4':[('正側斜率是否固定為一','Whether the positive slope is fixed at one'),('負側斜率是否由訓練學習','Whether training learns the negative slope'),('是否整組正規化','Whether it normalizes the full vector'),('輸出是否只能零或一','Whether it outputs only zero or one')]}
  badge(tr('四個選項等樣式；提示不代替你做判斷','Equal styling for all choices; hints do not make the decision for you'))
  for j,(zh,en) in enumerate(opts[kind]):
   x=70+(j%2)*500;y=210+(j//2)*210
   shape('ellipse',x,y,50,50,PURPLE);label(x,y+4,50,42,chr(65+j),25,PURPLE);text(x+68,y,350,95,tr(zh,en),24 if LANG=='zh' else 23,INK,True)
   xx=x+105;yy=y+118;ww=240;hh=65
   P.line([(xx,yy+hh),(xx+ww,yy+hh)],GREY,1,z=3)
   if kind=='quiz2':
    if j==2:
     for z,pv in enumerate([.5,.3,.2]):P.rect(xx+z*80,yy+hh-pv*hh,50,pv*hh,fill='#ded1f1',stroke=PURPLE,sw=2,z=4)
    else:
     fn=[FUN['relu'],FUN['maxout'],None,FUN['tanh']][j]
     pts=[(xx+ww*z/80,yy+hh-(fn(-2+4*z/80)+2)/4*hh) for z in range(81)];P.line(pts,PURPLE,3,z=4)
   elif kind=='quiz4':
    if j in [0,1]:
     P.line([(xx,yy+hh),(xx+ww/2,yy+hh*.7),(xx+ww,yy)],PURPLE,3,z=4)
     if j==1:
      P.ellipse(xx+30,yy+20,15,fill='#f0eafb',stroke=PURPLE,sw=2,z=4);P.line([(xx+30,yy+20),(xx+40,yy+12)],PURPLE,3,z=5)
    elif j==2:
     for z in range(3):P.rect(xx+z*80,yy+20,65,40,fill='#f0eafb',stroke=PURPLE,sw=2,z=4)
    else:
     for z in [0,1]:shape('ellipse',xx+z*140,yy+20,45,45,PURPLE);eq(xx+z*140,yy+20,45,str(z),22,h=45,c=PURPLE)
   elif kind=='quiz1':
    if j in [0,1,3]:
     pts=[(xx,yy+hh if j!=0 else yy+hh*.8),(xx+ww/2,yy+hh if j==1 else yy+hh*.5),(xx+ww,yy)]
     P.line(pts,PURPLE,3,smooth=j==0,z=4)
    else:
     for z in range(3):shape('ellipse',xx+z*80,yy+20,36,36,PURPLE)
   else:
    if j in [0,1]:
     for z in range(3):
      P.ellipse(xx+z*80,yy+40,13,fill='#f0eafb',stroke=PURPLE,sw=2,z=4)
      if j==0 and z>0:P.line([(xx+z*80-7,yy+33),(xx+z*80+7,yy+47)],PURPLE,2,z=5)
    elif j==2:P.line([(xx,yy),(xx+ww,yy+hh)],PURPLE,3,z=4)
    else:
     for z in range(3):P.rect(xx+z*80,yy+hh-(z+1)*15,45,(z+1)*15,fill='#f0eafb',stroke=PURPLE,sw=2,z=4)
 else:raise ValueError(kind)

NOTES={
'linear':('單一線性隱藏層可以存在於其他地方仍有非線性的模型；只有全線性活化的網路才整體坍縮為一次仿射映射。回歸的線性輸出適用於無限制實數目標，不是所有回歸任務都必須使用。','A linear hidden layer may coexist with nonlinear layers elsewhere. Only entirely linear activations collapse the network into one affine map. Linear regression outputs suit unrestricted real targets, not every possible regression task.'),
'map':('講義表格十二列中，Binary step與Sign共用一列但算兩種，因此共十三種。Gaussian沿用來源分組位置，但它是鐘形而非S形。「跨層」並非跨不同網路深度計算；Softmax用整個候選向量，Maxout用多個候選分支。','The twelve table rows count thirteen named functions because Binary step and Sign share one row. Gaussian retains its source grouping but is bell-shaped. “Cross-layer” does not mean crossing network depths: Softmax uses a vector and Maxout multiple candidates.'),
'softmax':('Softmax每個輸出介於零與一且整組和一，條件是有限分數與至少兩個類別。輸出是模型的互斥類別機率，不自動保證與真實頻率校準。Sigmoid逐分量運算、沒有共同分母，不等於統計上證明各標籤獨立。','For finite logits and at least two classes, each Softmax output lies between zero and one and their sum is one. These are categorical model probabilities, not a calibration guarantee. Pointwise Sigmoid has no shared denominator; this does not establish statistical label independence.'),
'softmaxjac':('補充曲線使用分數a、零、零。紫曲線是第二類與第三類各自的機率，它們完全重合，不能將其看成兩類總和。delta是Kronecker符號，i等於j時一，否則零。正對角表示本類分數上升會提高本類機率，負非對角表示其餘類別分配下降。','The added example uses logits a, zero, zero. Purple represents each of the two identical other probabilities, not their combined probability. Kronecker delta is one for i=j and zero otherwise. Diagonal entries are positive; off-diagonal entries are negative.'),
'gradient':('Sigmoid導數最大四分之一，因此五個活化導數因子連乘至多0.0009765625。這不是完整梯度的上界：多層反向傳播還包含權重矩陣與上游梯度。ReLU正側因子一也不保證整體梯度不消失或不爆炸。','The maximum Sigmoid derivative is one quarter, so the product of five activation factors is at most 0.0009765625. This is not a bound on the complete gradient, which also contains weight matrices and upstream gradients. Positive ReLU factors of one do not guarantee against vanishing or exploding gradients.'),
'prelu':('PReLU正側斜率一，負側斜率alpha由訓練學習。本圖比較三個正參數。alpha大於零才具有全實數值域；alpha等於零退化為ReLU，alpha小於零時兩側輸出都非負。零點一般有折角，只有alpha一時接成直線。','PReLU has positive slope one and a learned negative slope alpha. The curves show three positive parameters. A real-valued full range requires alpha>0. Alpha zero is ReLU; negative alpha makes both sides nonnegative. The kink disappears only when alpha is one.'),
'elu':('ELU值域負alpha到正無限是alpha正時的結論。負側導數alpha乘exp(a)，零點左導數alpha、右導數一，只有alpha一才在零點可微。本圖用alpha一。下限負一不在有限輸入時到達。','The range (-alpha,infinity) assumes alpha>0. The negative derivative is alpha exp(a). The one-sided derivatives at zero are alpha and one, so differentiability at zero requires alpha=1, used in this plot. The lower limit minus one is not attained at finite input.'),
'dying':('當這個單元對所有相關訓練輸入都在負側，其局部導數為零，普通資料損失經過此單元的梯度受阻。不是一次負輸入就永遠死亡，也不排除動量、正則化或上游分布變動讓狀態改變。Leaky或PReLU非零負斜率保留路徑，但上游梯度或輸入仍可能為零。','If this unit is negative for all relevant training inputs, its zero local derivative blocks the ordinary data-loss gradient through it. One negative input is insufficient; momentum, regularization or changed upstream distributions may alter the state. Nonzero Leaky/PReLU slopes retain a path but do not guarantee an update.'),
'maxout':('候選a_j等於w_j轉置x加b_j。唯一贏家j星時，對候選的偏導是贏家一其餘零，但對原始輸入的梯度是贏家的權重向量。本圖補充例max(a,-a)為絕對值，值域非負，說明固定Maxout的值域不能一概寫全實數；平手點通常可能不可微。','Candidate a_j is w_j transpose x plus b_j. For a unique winner, candidate derivatives are one for the winner and zero otherwise; the input gradient is the winner’s weight vector. The added absolute-value example has a nonnegative range, refuting a universal full-real range for fixed Maxout models. Ties may be nondifferentiable.'),
'plus':('在正二處ReLU、Leaky與alpha一的ELU同為二、導數一；Softplus為ln(1+exp(2))，導數為Sigmoid(2)，有限正值下不等於一。數字只在最後取到三位小數。曲線中前三條在正側重合，並未漏畫。','At plus two ReLU, Leaky and alpha-one ELU all output two with derivative one. Softplus is ln(1+exp(2)), with derivative Sigmoid(2), below one at finite input. Round only the final display to three decimals. The first three positive-side curves overlap exactly.'),
'softplus':('Softplus在有限輸入嚴格為正，導數是Sigmoid，嚴格介於零與一。它平滑近似ReLU，既不是負側恰零，也不是有限正側斜率恰一。','Softplus is strictly positive at finite input. Its derivative is Sigmoid, strictly between zero and one. It smoothly approximates ReLU without equaling zero on the negative side or having a slope of exactly one at finite positive inputs.'),
'output':('無限制實數回歸可用Linear、二元分類可用Sigmoid，互斥多類別用Softmax。多標籤問題若用多個Sigmoid是額外任務設定，與互斥分類不同；本頁不展開。輸出值域位於零一間不是已校準的證明。','Linear suits unrestricted real targets, Sigmoid binary classification and Softmax exclusive multiclass classification. Multilabel tasks use a different setup and are not developed here. An output in (0,1) alone does not prove calibrated probabilities.'),
}

def build(lang):
 global LANG,P,SECOND,TEMP
 LANG=lang;K.LANG=lang;title=tr('KP06 活化函數','KP06 Activation functions')
 d=K.make(title,title,'kp06');sid='scene-kp06';d['nodes']['frame-kp06']['name']=title
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['metadata'].update(createdAt=now,updatedAt=now)
 manifest=[];mascot.STORY={p[0]:p[7:] for p in PAGES}
 for i,(kind,zh,en,tz,te,sec,*_) in enumerate(PAGES):
  P=K.Page(prefix=f'{lang}-p{i+1:02d}');K.P=P;SECOND=[];TEMP=[]
  P.rect(0,0,1280,720,fill='#ffffff',z=0);P.rect(0,0,1280,86,fill='#edf4fa',gradientTo='#f6f8fc',gradientAngle=0,gradientStops=[{'offset':.45,'color':'#e9effb'}],z=1);P.rect(0,0,13,86,fill=BLUE,z=2)
  text(48,20,1180,60,tr(zh,en),34 if lang=='zh' else 28,INK,True,name='page-title');text(1130,684,100,28,f'{i+1:02d} / {len(PAGES)}',17,MUT)
  start=len(P.nodes);draw(kind)
  st=len(P.nodes);mascot.draw(K,kind);phase2(st)
  main_ids=[n['id'] for n in P.nodes[start:]];first=[nid for nid in main_ids if nid not in SECOND]
  if kind=='softmaxcalc':
   t=text(665,540,335,55,tr('停一下：分母相同嗎？','Pause: same denominator?'),21,ORANGE,True);TEMP=[t['id']];first+=TEMP;main_ids+=TEMP
  ft=len(P.nodes);P.rect(48,640,1184,56,fill='#eaf5ef',radius=12,z=8)
  take=tr(tz,te)
  if kind=='hidden':take=tr('ReLU 是常用起點，{color=#b24b52}不是{/color}所有問題的保證。','ReLU is a common start, {color=#b24b52}not{/color} a universal guarantee.')
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
   if kind=='softmaxcalc' and beat==1:point_['y']=385;zoom=1.07
   d['steps'].append(dict(id=f'{lang}-p{i+1}-beat{beat+1}',order=len(d['steps']),label=tr(zh,en)+f' · {beat+1}',durationMs=650,holdMs=600,easing='smooth',target=dict(kind='scene-point',sceneId=sid,point=point_,zoom=zoom,referenceViewport=dict(width=1280,height=720)),reveal=dict(timing='after-camera',enter=[dict(nodeId=nid,effect='scale' if beat==1 else 'fade',delayMs=0,durationMs=300) for nid in ids],exitNodeIds=TEMP if beat==1 else []),speakerNotes=note))
  manifest.append(dict(page=i+1,kind=kind,title=tr(zh,en),source_section=sec,first_step=fs,final_step=len(d['steps'])-1,beats=3,extension=i>=24))
 out=ROOT/'output';out.mkdir(exist_ok=True);stem='lec02-kp06-activation-functions'+('-en' if lang=='en' else '')
 (out/(stem+'.jbd.json')).write_text(json.dumps(d,ensure_ascii=False,indent=1));(out/(lang+'-pages.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=1));print(lang,len(manifest),'pages;',len(d['steps']),'steps;',len(d['nodes']),'nodes')
if __name__=='__main__':save();build('zh');build('en')
