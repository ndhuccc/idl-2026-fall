from pathlib import Path
import sys,json,math,copy,datetime
ROOT=Path(__file__).parent
TOOLS=ROOT.parent/'idl-kp01-toolchain'/'.agents/skills/image-to-jbd-page/tools'
sys.path.insert(0,str(TOOLS))
from jbd_page import Page
from new_deck import make

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

