from pathlib import Path
import json,copy,math,datetime,re,sys
import pagekit as K
import mascot
R=Path(__file__).parent
P=None;LANG='zh';SECOND=[]
tr=lambda a,b:a if LANG=='zh' else b
INK,MUT,BLUE,PURPLE,GREEN,ORANGE,GREY=K.INK,K.MUT,K.BLUE,K.PURPLE,K.GREEN,K.ORANGE,K.GREY
text,label=K.text,K.label

def eq(x,y,w,s,size=26,h=110,c=INK):return K.eq(x,y,w,s,size,h,c)
def shape(kind,x,y,w,h,c=BLUE,fill=None,z=3):
 n=P.rect(x,y,w,h,fill=fill or '#edf5fc',stroke=c,sw=2,radius=12 if kind=='rectangle' else 0,z=z);n['shapeKind']=kind;return n
def box(x,y,w,h,s,c=BLUE,fs=25,kind='ellipse'):
 n=shape(kind,x,y,w,h,c);label(x+8,y+16,w-16,h-18,s,fs,c);return n
def conn(a,b,c=BLUE,sh=4,th=12,direction='forward',curve=False):
 n=P._add('connector',0,0,1,1,{'fill':'transparent','stroke':c,'textColor':INK,'fontSize':18,'borderWidth':3,'borderStyle':'solid','radius':0},z=25,name='native-flow');n['connector']={'sourceNodeId':a['id'],'sourceHook':sh,'targetNodeId':b['id'],'targetHook':th,'curve':curve,'direction':direction,'arrowSize':'medium'};return n

def flow(y=485):
 ns=[]
 for j,s in enumerate([tr('先查條件','Check conditions'),tr('代入方法','Apply the method'),tr('核對結果','Verify the result')]):ns.append(box(65+j*330,y,245,100,s,K.PAL[j],23))
 for a,b in zip(ns,ns[1:]):conn(a,b,GREY)

def points(n):
 return [(n['transform']['x']+p['x']/100*n['width'],n['transform']['y']+p['y']/100*n['height']) for p in n.get('polygonPoints',[])]
def load_body(k,oldpage,lang):
 u=R/f'KP{k:02d}';d=json.loads((u/f'base-{lang}.jbd.json').read_text());idx=oldpage-1;ns=[copy.deepcopy(n) for n in d['nodes'].values() if n['sceneId']!=d['rootSceneId'] and idx*1440<=n['transform']['x']<(idx+1)*1440]
 for n in ns:n['transform']['x']-=idx*1440
 actor=next((j-3 for j,n in enumerate(ns) if n.get('name')=='concept-dialogue'),len(ns));ns=ns[:actor]
 selected=[]
 for n in ns:
  x,y=n['transform']['x'],n['transform']['y']
  if n.get('name') in ['page-title','takeaway'] or y>=640 or (n['type']=='text' and y<160 and ((n['width']>=900 and n['height']<=55) or re.search(r'生物神經元的簡化機制示意|A simplified view of a biological neuron',n.get('text','')))) or (n['type']=='shape' and y<86 and n['width']>=1200) or (n['type']=='shape' and x==0 and y==0):continue
  if n['type']=='text' and re.match(r'^\d{2}\s*/\s*\d+',n.get('text','')):continue
  selected.append(n)
 keep={n['id'] for n in selected};selected=[n for n in selected if n['type']!='connector' or all(n['connector'][end+'NodeId'] in keep for end in ['source','target'])]
 # Convert legacy hand-drawn flow arrows to native connectors. Scientific data curves have no arrowhead.
 removed=set();replacements=[]
 for head in selected:
  if head.get('shapeKind')!='polygon' or len(head.get('polygonPoints',[]))!=3 or max(head['width'],head['height'])>35:continue
  hp=points(head)
  candidates=[]
  for line in selected:
   if line.get('shapeKind')!='polyline' or line['id'] in removed:continue
   lp=points(line)
   if len(lp)<2:continue
   dist=min(math.dist(lp[-1],p) for p in hp)
   if dist<2.5 and math.dist(lp[0],lp[-1])>25:candidates.append((dist,line,lp))
  if not candidates:continue
  _,line,lp=min(candidates,key=lambda t:t[0]);removed.update([head['id'],line['id']]);replacements.append((lp[0],lp[-1],line.get('style',{}).get('stroke',BLUE),len(lp)>2))
 selected=[n for n in selected if n['id'] not in removed]
 # Reserve the mascot column in the first two legacy decks, whose old diagrams used the full width.
 geom=[n for n in selected if n['type']!='connector'];left=min([n['transform']['x'] for n in geom] or [50]);right=max([n['transform']['x']+n['width'] for n in geom] or [1000]);sx=min(1,950/(right-left)) if k<=2 else 1;dx=50-left*sx if k<=2 else 0
 mapping={};start=len(P.nodes)
 for n in selected:
  oldid=n['id']
  original_y=n['transform']['y']
  if n['type']=='text' and original_y<160:n['transform']['y']=168
  if k==1 and n['type']=='text' and original_y<130 and n.get('text','').startswith('$'):
   n['transform']['y']=142;n['height']=43;n['style']['fontSize']=22
  if n['type']=='text':
   val=n.get('text','')
   if val=='OR: one line separates':n['text']='OR: separable'
   if val=='XOR: no separating line':n['text']='XOR: not separable'
   if val=='$x_0=1$':n['width']=max(n['width'],150);n['height']=max(n['height'],95)
   if val in ['$-1-(1)=-2$','$-1-(-1)=0$']:n['height']=max(n['height'],78)
   if r'\mathbf x_{\rm new}' in val:n['height']=max(n['height'],48);n['transform']['y']=min(n['transform']['y'],580)
   if val==r'$\frac12\sum(y-\hat y)^2$':n['height']=110;n['transform']['y']=220;n['style']['fontSize']=26
   if val==r'$x_0=0.5:\ (0,1,1,1),\quad 1/4$':n['height']=48;n['transform']['y']=588
  n['id']=f'{P.prefix}-{len(P.nodes)+1}';mapping[oldid]=n['id'];n['transform']['x']=dx+n['transform']['x']*sx;n['width']*=sx;n['initiallyHidden']=True
  if n['type']=='text':
   n['style']['fontSize']=max(18,n['style'].get('fontSize',18));n['zIndex']=40;n['height']=max(n['height'],n['style']['fontSize']*1.35)
  else:n['zIndex']=25 if n['type']=='connector' else min(20,n.get('zIndex',3))
  P.nodes.append(n)
 for n in P.nodes[start:]:
  if n['type']=='connector':
   for end in ['source','target']:n['connector'][end+'NodeId']=mapping[n['connector'][end+'NodeId']]
 for a,b,c,curved in replacements:
  ends=[]
  for x,y in [a,b]:ends.append(P.ellipse(dx+x*sx,y,.5,fill='transparent',stroke='transparent',z=1))
  conn(ends[0],ends[1],c,curve=curved)
 # Retain the original evidence reveal where available, excluding the removed old mascot/footer.
 oldsteps=[s for s in d['steps'] if s['target'].get('point',{}).get('x')==idx*1440+640]
 late={e['nodeId'] for s in oldsteps[1:] for e in s.get('reveal',{}).get('enter',[])}
 SECOND.extend(mapping[x] for x in late if x in mapping)
 if not SECOND:
  tx=[n for n in P.nodes[start:] if n['type']=='text']
  if tx:SECOND.append(tx[-1]['id'])
 return dict(reused_body_nodes=len(selected),converted_legacy_arrows=len(replacements),horizontal_scale=sx)

def draw(p,k,lang):
 global SECOND
 typ=p['kind'];stats={}
 if typ=='base':return load_body(k,p['base_page'],lang)
 if typ=='question':
  q=tr(*p['question']);text(65,190,930,220,q,26 if lang=='zh' else 24,INK,True,name='question')
  a=box(410,430,230,100,'?',ORANGE,40,'diamond')
  b=box(65,545,245,70,tr('寫下你的判斷','Write your judgment'),BLUE,23);c=box(745,545,245,70,tr('下一頁核對','Check next page'),BLUE,23);conn(a,b,BLUE,sh=10,th=0);conn(a,c,BLUE,sh=6,th=0)
 elif typ=='quiz':
  q=tr(*p['question']);text(65,165,930,110,q,24 if lang=='zh' else 22,INK,True,name='question')
  for j,(zh,en) in enumerate(p['options']):
   yy=285+j*83;circle=shape('ellipse',65,yy,48,48,PURPLE);label(65,yy+3,48,45,chr(65+j),25,PURPLE)
   text(140,yy-2,850,75,tr(zh,en),24 if lang=='zh' else 22,INK,True,name='choice')
 elif typ=='calculation' and p['math']:
  for j,s in enumerate(p['math']):
   yy=195+j*135;n=shape('star' if j==0 else 'ellipse',65,yy+15,50,50,K.PAL[j]);label(65,yy+18,50,45,str(j+1),25,K.PAL[j]);e=eq(145,yy,850,s,26,120,GREEN if j==len(p['math'])-1 else INK)
   if j>0:SECOND.extend([e['id']])
 elif typ=='concept' and p['math']:
  for j,s in enumerate(p['math']):
   yy=190+j*135;shape('star',65,yy+25,28,28,ORANGE,ORANGE);eq(115,yy,870,s,26,120)
 elif typ=='quizanswer':
  answer=p['answer'];labeltext=tr('原題更正','Correction') if answer is None else chr(65+answer)
  a=box(65,190,220,115,labeltext,GREEN,28,'diamond');b=box(425,190,550,115,tr('條件、方向與限定範圍','Assumptions, direction and scope'),BLUE,25);conn(a,b,GREEN)
  text(70,360,920,230,tr(*p['answer_text']),25 if lang=='zh' else 23,INK,True,name='explanation')
 else:
  if typ=='explain':body=tr(*p['answer_text'])
  elif typ=='scope':body=tr(*p['notes'])
  elif typ=='review':body=tr('回想：輸入是什麼？中間做哪種運算？輸出怎麼判斷？\n再說出一個適用條件，與一個不能保證的事情。','Recall the input, intermediate operation and output decision.\nState one assumption and one thing the method does not guarantee.')
  elif typ=='concept':body=tr(*p.get('answer_text',p['notes']))
  else:body=tr(*p['notes'])
  text(65,185,930,255,body,25 if lang=='zh' else 23,INK,True,name='explanation');flow(490)
  if typ=='scope':
   shape('triangle',965,600,28,28,ORANGE,ORANGE);badge=P.rect(70,450,210,35,fill='#fff0d6',stroke=ORANGE,sw=1,radius=5,z=8);badge['transform']['rotation']=-4;bt=label(70,452,210,35,tr('先查適用條件','CHECK SCOPE'),20,ORANGE);bt['transform']['rotation']=-4
 if typ in ['calculation','concept'] and p['math']:
  # Step markers plus progression rail express the derivation order.
  P.line([(90,270),(90,540)],GREY,2,dash=True,z=2)
 return stats

BUBBLES={1:('總和等於門檻，\n會放電嗎？','At the threshold,\ndo we fire?'),2:('這筆猜對，\n還需要改嗎？','If it is correct,\ndo we update?'),3:('先查標籤，\n錯分才更新？','Check the label.\nUpdate on mistakes?'),4:('是學不會，\n還是表示不了？','Failure to learn,\nor to represent?'),5:('層與層之間，\n哪裡轉了彎？','Between layers,\nwhere is the bend?'),6:('這個區間，\n梯度能通過嗎？','In this interval,\ncan gradients pass?'),7:('這把尺，\n會放大哪種錯？','Which errors\ndoes this amplify?'),8:('先比分數，\n還是比機率？','Compare a score\nor a probability?'),9:('先找勝者，\n還要拉誰？','Find the winner.\nWho else moves?')}

def build(k,lang):
 global P,LANG,SECOND
 LANG=lang;K.LANG=lang;unit=R/f'KP{k:02d}';plan=json.loads((unit/'plan.json').read_text());stem=(unit/'stem.txt').read_text();title=f'KP{k:02d} '+tr('導學自學版','Guided self-study');d=K.make(title,title,f'guided-{k}');sid=f'scene-guided-{k}';man=[];stats=[]
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['metadata'].update(createdAt=now,updatedAt=now)
 for i,p in enumerate(plan):
  P=K.Page(prefix=f'{lang}-p{i+1:02d}');K.P=P;SECOND=[]
  P.rect(0,0,1280,720,fill='#ffffff',z=0);P.rect(0,0,1280,90,fill='#eef4fb',gradientTo='#ffffff',gradientStops=[{'offset':.5,'color':'#f5f8fc'}],z=1)
  text(48,16,1010,85,tr(p['title'],p['title_en']),34 if lang=='zh' else 32,INK,True,name='page-title');text(1070,34,165,35,f'KP{k:02d} · {i+1}/{len(plan)}',18,MUT,False,'right')
  text(50,110,950,53,tr(p['bridge'],p['bridge_en']),19,MUT,name='bridge')
  start=len(P.nodes);st=draw(p,k,lang);stats.append(st)
  m=p['mascot'];bz,be=BUBBLES[k] if p['kind']=='base' else (m['bubble'],m['bubble_en'])
  specific={'anatomy':('誰負責接收，\n誰負責整合？','Who receives,\nwho integrates?'),'weighted':('每條輸入，\n都一樣重要嗎？','Is every input\nequally important?'),'bias':('門檻移到零，\n偏置怎麼變？','Move to zero.\nWhat is the bias?'),'batch':('一起更新，\n等於逐筆更新嗎？','Batch updates: same\nas sequential ones?'),'normal':('沿邊界走，\n分數會改變嗎？','Move on the line.\nDoes score change?')}
  if p.get('base_kind') in specific:bz,be=specific[p['base_kind']]
  if p['kind'] in ['question','quiz']:bz,be='先寫理由，\n下一頁再核對。','State a reason.\nThen check next.'
  mascot.STORY={p['kind']:(m['prop'],m['mood'],bz,be)};a0=len(P.nodes);mascot.draw(K,p['kind']);SECOND.extend(n['id'] for n in P.nodes[a0:])
  label(1045,578,190,40,tr('教學比喻','Teaching metaphor'),18,MUT)
  SECOND.append(P.nodes[-1]['id']);main=[n['id'] for n in P.nodes[start:]];first=[x for x in main if x not in SECOND]
  f0=len(P.nodes);P.rect(48,640,1184,70,fill='#eaf5ef',radius=12,z=8)
  take=tr(p['takeaway'],p['takeaway_en'])
  if p['kind']=='review':take=tr('用自己的話重述{color=#b76716}適用條件{/color}，再檢查例外。','Restate the {color=#b76716}assumptions{/color}, then check exceptions.')
  text(70,644,1140,64,take,24 if lang=='zh' else 23,GREEN,True,name='takeaway')
  groups=[first,SECOND,[n['id'] for n in P.nodes[f0:]]]
  for n in P.nodes:
   n['sceneId']=sid;n['transform']['x']+=i*1440;n['initiallyHidden']=n['id'] in main+groups[-1]
   if n['type']=='text':n['style']['fontFamily']='Noto Sans CJK TC' if lang=='zh' else 'Arial';n['zIndex']=40
   d['nodes'][n['id']]=n;d['scenes'][sid]['nodeIds'].append(n['id'])
  fs=len(d['steps']);full=tr(*p['notes']);maths='\n\n'.join('$'+s+'$' for s in p['math'])
  notes=[tr(p['bridge'],p['bridge_en'])+'\n\n'+full,tr('現在對照新出現的證據，逐項核對。','Now verify each piece of newly revealed evidence.')+'\n\n'+full+('\n\n'+maths if maths else ''),tr(p['takeaway'],p['takeaway_en'])+'\n\n'+full+tr('\n\n請用自己的話重述，再往下讀。','\n\nRestate the idea in your own words before continuing.')]
  for beat,ids in enumerate(groups):d['steps'].append(dict(id=f'{lang}-p{i+1}-b{beat+1}',order=len(d['steps']),label=tr(p['title'],p['title_en'])+f' · {beat+1}',durationMs=450,holdMs=450,easing='smooth',target=dict(kind='scene-point',sceneId=sid,point=dict(x=i*1440+640,y=360),zoom=1,referenceViewport=dict(width=1280,height=720)),reveal=dict(timing='after-camera',enter=[dict(nodeId=nid,effect='fade',delayMs=0,durationMs=250) for nid in ids],exitNodeIds=[]),speakerNotes=notes[beat]))
  man.append(dict(page=i+1,kind=p['kind'],title=tr(p['title'],p['title_en']),first_step=fs,final_step=len(d['steps'])-1,beats=3))
 out=unit/'output';out.mkdir(exist_ok=True);name=stem+('-en' if lang=='en' else '')
 (out/(name+'.jbd.json')).write_text(json.dumps(d,ensure_ascii=False,indent=1));(out/(lang+'-pages.json')).write_text(json.dumps(man,ensure_ascii=False,indent=1));(out/'reuse-audit.json').write_text(json.dumps(stats,indent=1));print(k,lang,len(plan),'pages',len(d['nodes']),'nodes')
if __name__=='__main__':
 ks=[int(x) for x in sys.argv[1:]] or list(range(1,10))
 for k in ks:
  for lang in ['zh','en']:build(k,lang)
