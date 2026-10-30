from pathlib import Path
import json,re,ast,copy
from answers import PRACTICE,QUIZ_KEYS,QUIZ_NOTES
from english_options import EN
R=Path(__file__).parent
src=(R/'.agents/skills/self-study-teaching-deck/tools/check_coverage.py').read_text();mod=ast.parse(src);mod.body=[n for n in mod.body if not isinstance(n,ast.Expr)];env={};exec(compile(mod,'coverage definitions','exec'),env);anchors=env['anchors']
CJK=lambda s:bool(re.search(r'[一-鿿]',s))
def block(s,label):
 m=re.search(r'^\*\*'+re.escape(label)+r'\*\*[^\n]*\n',s,re.M)
 if not m:return ''
 head=m.group(0).split('**',2)[-1].strip();head=re.sub(r'^[（(][^）)]*[）)]\s*[：:]?\s*','',head).lstrip('：: ');tail=s[m.end():];end=re.search(r'^\*\*|^### ',tail,re.M);return (head+'\n'+tail[:end.start() if end else None]).strip()
def numbered(t):
 return [m.group(2).strip() for m in re.finditer(r'(?ms)^(\d+)\. (.*?)(?=^\d+\. |\Z)',t)]
def pieces(s):
 return [t.strip() for t in re.split(r'\n\s*\n',s) if t.strip()]
def lang_text(s,lang):
 ps=[p for p in pieces(s) if (CJK(p) if lang=='zh' else not CJK(p)) and not p.startswith('![') and not p.startswith('[')]
 return '\n\n'.join(ps)
def page(kind,zh,en,tz,te,anc,bz,be,visual,math=None,nz='',ne='',**extras):
 return dict(kind=kind,title=zh,title_en=en,takeaway=tz,takeaway_en=te,source_anchors=anc,bridge=bz,bridge_en=be,visual=visual,math=math or [],notes=[nz,ne],new_terms=[],numbers=[],mascot=dict(mood='think',prop='ruler',bubble='先看條件，\n你會怎麼判斷？',bubble_en='Check conditions.\nWhat do you expect?'),**extras)
CORRECT={
1:('本模型只是加總與門檻的計算抽象，不是完整生理模擬；平手採大於等於，固定權重不等於已會學習。','This is a computational abstraction, not a full biological simulation. Equality fires; fixed weights do not imply learning.'),
2:('更新只保證當筆分數往正確方向，不能保證所有點立即變好。零初值縮放學習率的等價需要相同順序與平手規則；批次與逐筆通常不是同一條軌跡。','An update improves this sample’s score direction, not every sample. Rate-scaling equivalence needs zero starts and consistent order/ties; batch and sequential trajectories can differ.'),
3:('幾何真距離是|s|除以空間權重範數。錯分時ys為負且|s|=-ys。未正規化Perceptron準則是分段線性而非處處可微；損失零在邊界仍可能有錯分，更新條件要另判。','True distance is |s| divided by the spatial norm; for strict misclassification |s|=-ys. The unnormalized criterion is piecewise linear, not everywhere differentiable. Zero criterion at a boundary can coexist with a mistake; keep the update condition.'),
4:('XOR無法被單一仿射門檻表示。特定種子、初值或搜尋格點只說明該次實驗，不構成所有情況的證明；ADALINE換連續目標不自動保證較佳泛化。','A single affine threshold cannot represent XOR. Individual seeds, starts or grid searches illustrate particular runs, not universal proofs. ADALINE’s continuous objective does not guarantee better generalization.'),
5:('無中間非線性時，無偏置網路合併為線性，有偏置為仿射；瓶頸維度可限制秩。非線性增加可能的表示，但不保證合適參數一定學得到。','Without intermediate nonlinearities, composition is linear without biases and affine with them; bottlenecks may constrain rank. Nonlinearity enables richer representations without guaranteeing parameter learning.'),
6:('ReLU在零不可微，正側導數一只避免局部縮小，不保證全網路無梯度消失。Softmax是聯合正規化；maxout是多個仿射分數的最大值；固定負斜率0.01只是本課例。','ReLU is nondifferentiable at zero; its positive derivative does not guarantee network-wide gradients. Softmax jointly normalizes, maxout maximizes affine responses, and 0.01 is an example fixed negative slope.'),
7:('平方和與均方差一個樣本數。KL需指定方向與支撐；單一sigmoid不需自己和為一。Hinge的分數餘裕不自動等於幾何距離；Huber值與一階導數連續，不代表二階處處連續。','Sums and means differ by sample count. KL needs direction and support conditions; one sigmoid need not sum to one. Hinge score margin is not automatically geometric distance. Huber is C1, not everywhere C2.'),
8:('一般牛頓式beta-H逆g用於尋找極大或極小駐點都相同；對數似然H為負，令A=-H才得到加A逆g。凹性不保證有限MLE或每步改善；完全或準完全分離都要留意。Logistic輸出是連續機率；最大似然不等於最少分類錯誤。','The generic Newton stationary-point step is beta-H inverse g for minima and maxima. With negative likelihood Hessian, A=-H gives plus A inverse g. Concavity alone ensures neither a finite MLE nor improving steps; consider complete/quasi separation. Logistic outputs continuous probabilities; likelihood is not a minimum-error count.'),
9:('本課RBF先固定中心寬度，只學輸出；一般變體也可調中心。XOR要偏置，截短小數不給精確零一。硬鄰域外不動，高斯軟鄰域遠處仍有微小更新；格子距離與輸入距離不同。','This RBF setup fixes centres and widths, though variants may learn them. Exact XOR fitting needs bias and full precision. Hard and Gaussian neighbourhoods differ; grid distance is distinct from input distance.')}

def make_plan(k):
 unit=R/f'KP{k:02d}';s=(unit/'source-lecture.md').read_text();aa=anchors(s);base=[json.loads((unit/f'base-{lang}.jbd.json').read_text()) for lang in ['zh','en']];man=[json.loads((unit/f'base-{lang}-pages.json').read_text()) for lang in ['zh','en']];pp=[];nextpage=None
 headings=[a for a in aa if re.match(r'^\d+\.',a)];sections={}
 for match in re.finditer(r'(?ms)^### (.*?)\n(.*?)(?=^### |\Z)',s):sections[env['short'](match.group(1))]=match.group(2)
 for old in man[0]:
  kind=old['kind'];idx=old['page']-1
  if kind.startswith(('practice','quiz')):continue
  sec=old['source_section'];a=[h for h in headings if any(re.search(r'[§第]'+str(j)+r'(?!\d)',sec) for j in [int(h.split('.')[0])])]
  if not a:a=['學習目標'] if kind=='cover' else ['本單元重點整理']
  notes=[];takes=[];titles=[];maths=[]
  for li,lang in enumerate(['zh','en']):
   ds=base[li];nodes=[n for n in ds['nodes'].values() if n['sceneId']!=ds['rootSceneId'] and idx*1440<=n['transform']['x']<(idx+1)*1440];take=next((n['text'] for n in nodes if n.get('name')=='takeaway'),man[li][idx]['title']);takes.append(take);titles.append(man[li][idx]['title'])
   pgsteps=[st for st in ds['steps'] if st['target'].get('point',{}).get('x')==idx*1440+640];oldnote=max([st.get('speakerNotes','') for st in pgsteps],key=len,default='')
   # Reuse source-verified explanations and add contextual source prose, with explicit corrections taking priority.
   prose=lang_text(sections.get(a[0],''),lang);pars=pieces(prose);q=' '.join([titles[-1],take]);tokens=set(re.findall(r'[一-鿿]|[A-Za-z]{3,}',q));rank=sorted(enumerate(pars),key=lambda v:len(tokens.intersection(set(re.findall(r'[一-鿿]|[A-Za-z]{3,}',v[1])))),reverse=True);support='\n\n'.join(pars[j] for j,_ in sorted(rank[:2]))
   notes.append(oldnote+('\n\n講義補讀：\n' if lang=='zh' else '\n\nSource context:\n')+support+'\n\n'+CORRECT[k][li])
   if li==0:maths=[m for n in nodes if n['type']=='text' and n.get('name')!='concept-dialogue' for m in re.findall(r'\$([^$]+)\$',n.get('text',''))]
  if kind=='cover':
   titles=takes;a=['學習目標','符號表'];terms=[]
   for line in sections.get('符號表','').splitlines():
    cells=re.split(r'(?<!\\)\|',line)[1:-1]
    if len(cells)>=3 and cells[0].strip().startswith('$'):terms.append([c.strip() for c in cells[:3]])
   notes[0]+='\n\n符號速查：\n'+'\n'.join(a+': '+b for a,b,c in terms);notes[1]+='\n\nSymbols:\n'+'\n'.join(a+': '+c for a,b,c in terms)
  bz='承接前置知識：先看本單元要回答什麼。' if not pp else '上一頁：'+re.sub(r'\{[^}]+\}','',pp[-1]['takeaway'])
  be='From prerequisites: identify this unit’s central question.' if not pp else 'Previously: '+re.sub(r'\{[^}]+\}','',pp[-1]['takeaway_en'])
  p=page('base',*titles,*takes,a,bz,be,'沿用已核對原生科學圖與精算位置，補導學角色和完整導讀',maths,*notes,base_page=old['page'],base_kind=kind)
  if kind=='cover':p['new_terms']=[{'symbol':a,'meaning_zh':b,'meaning_en':c} for a,b,c in terms]
  if '符號' in sec:p['source_anchors'].append('符號表')
  if '更正' in sec:p['source_anchors'].append('來源更正與取捨')
  if kind in ['next','learn']:p['source_anchors']=['重點:留給下一個知識點的問題'];nextpage=p
  else:pp.append(p)
 # Optional source material is explicitly taught, not hidden in an unassigned anchor.
 deeper_z=block(s,'想深入');deeper_e=block(s,'Going deeper');dm=re.findall(r'\$([^$]+)\$',deeper_z)
 if deeper_z:
  for start in range(0,max(1,len(dm)),3):
   p=page('concept','再深一層：把條件與主張接起來','Go deeper: connect assumptions and claims','選讀內容也要分清來源敘述與適用條件。','Separate the optional claim from its assumptions.',['想深入'],'主線走完後，再看講義的選讀內容。','After the main route, revisit the optional material.','原生流程連接條件、方法、結論，完整推理在導讀',dm[start:start+3],deeper_z+'\n\n本稿釐清：'+CORRECT[k][0],deeper_e+'\n\nClarification: '+CORRECT[k][1]);p['answer_text']=[deeper_z,deeper_e];pp.append(p)
 # Every misconception gets an independent question followed by an explanation.
 mz=[t[2:].strip() for t in block(s,'常見迷思').splitlines() if t.startswith('- ')];me=[t[2:].strip() for t in block(s,'Common misconceptions').splitlines() if t.startswith('- ')]
 for j,z in enumerate(mz,1):
  split=z.split('——',1);claim=split[0];expl=split[1] if len(split)>1 else z;e=me[j-1] if j<=len(me) else ''
  em=re.match(r'\s*["“](.*?)["”]\s*(.*)',e,re.S);ec=em.group(1) if em else 'Does this source claim hold under its stated conditions?';ex=em.group(2) if em else e
  p=page('question',f'迷思{j}：這個說法成立嗎？',f'Misconception {j}: does the claim hold?','先寫下你的判斷與條件，再翻頁核對。','Write your judgment and conditions before checking.',[f'常見迷思#{j}'],'主線有了，試著辨認一個容易混淆的說法。','Use the main ideas to examine a tempting claim.','中立問題與兩個待核對分支',[],claim,ec,question=[claim,ec],self_check=True);pp.append(p)
  p=page('explain',f'迷思{j}解析：條件不能省略',f'Misconception {j}: keep the conditions','先把適用範圍說清楚，才判斷結論。','State the scope before accepting the conclusion.',[f'常見迷思#{j}'],'核對上一頁：先問是哪個模型、哪個條件。','Check the previous page: which model and which assumptions?','條件到推理到正確結論，角色提醒例外',[],expl+'\n\n'+CORRECT[k][0],ex+'\n\n'+CORRECT[k][1],answer_text=[expl,ex],self_check=True);pp.append(p)
 # Three exercises, with complete calculations on the following answer pages.
 qz=numbered(block(s,'小練習'));qe=numbered(block(s,'Practice'))
 for j,(z,e,ans) in enumerate(zip(qz,qe,PRACTICE[k]),1):
  ztitle,etitle,eqs,tz,te,nz,ne=ans
  p=page('question',f'練習{j}：先獨立完成，再翻頁',f'Practice {j}: solve it before turning the page','先列已知、公式與條件，不要跳過中間步驟。','List givens, formulas and conditions; keep intermediate steps.',[f'小練習#{j}'],'把剛才的方法換成自己的手算。','Now apply the method independently.','完整原題與提示，不指示答案',[],z,e,question=[z,e],self_check=True);pp.append(p)
  for start in range(0,max(1,len(eqs)),3):
   p=page('calculation',ztitle,etitle,tz,te,[f'小練習#{j}']+(['解題示範'] if j==1 else []),'核對上一頁：先算，再解釋方向或條件。','Check the previous page: calculate, then explain the direction or conditions.','三步算式配對原生標記與連接線，概念題用條件推理流程',eqs[start:start+3],nz,ne,self_check=True);p['numbers']=['added_practice'];pp.append(p)
 # Keep all four/eight source choices; corrections are explicitly recorded below.
 table=block(s,'自我檢核');qs=[re.split(r'(?<!\\)\|',l)[1:-1] for l in table.splitlines() if re.match(r'^\|\s*\d+\.',l)]
 for j,cells in enumerate(qs,1):
  raw=cells[0].strip();parts=raw.split(' / ',1);z=parts[0];e=parts[1] if len(parts)>1 else f'Source question {j}';opts=[]
  for c in cells[1:5]:
   zz=c.strip().split(' / ',1);opts.append([zz[0],zz[1] if len(zz)>1 else zz[0]])
  # Plain-Chinese-only options are translated below rather than left in the English deck.
  opts=[[pair[0],EN[k][j-1][q]] for q,pair in enumerate(opts)]
  if k==4 and j==4:z='ADALINE 改用哪一種訓練目標？';e='Which training objective does ADALINE use?'
  if k==5 and j==2:opts[2][0]='輸入資料層，不算可訓練運算層'
  if k==5 and j==3:opts[1][0]='單一線性映射；含偏置時為仿射'
  if k==7 and j==4:opts[1][0]='有效機率：多類正規化，或二元 p 與 1-p'
  if k==8 and j==6:z='講義引用的 logistic 早期用途是？';e='Which early logistic application is cited in the handout?'
  p=page('quiz',f'檢核{j}：選出最符合條件的一項',f'Check {j}: choose the best supported option','選項樣式相同；先作答，再看後頁解析。','Choices have equal styling; answer before reading the explanation.',[f'自我檢核#{j}'],'檢核概念：答案之外，還要能說出理由。','Check understanding: justify the choice, not just its letter.','四選項同色等尺寸，不以角色或圖示洩漏答案',[],z,e,question=[z,e],options=opts,quiz_index=j,self_check=True);pp.append(p)
  nz,ne=QUIZ_NOTES[k][j-1];key=QUIZ_KEYS[k][j-1]
  p=page('quizanswer',f'檢核{j}解析：對照條件逐項判斷',f'Check {j}: justify the choice from its conditions','答對之外，也要知道其他選項錯在哪裡。','Explain both the supported option and why the alternatives fail.',[f'自我檢核#{j}'],'核對上一頁：把選項放回講義條件。','Check the previous choices against the handout’s assumptions.','正解／更正標記連到三段理由，完整排除說明在導讀',[],nz,ne,answer_text=[nz,ne],answer=key,self_check=True);pp.append(p)
 # A complete review and explicit source-correction page close the unit.
 sumz='\n\n'.join(block(s,label) for label in ['一句話重點','白話說明','正式定義','例子']);sume='\n\n'.join(block(s,label) for label in ['One-sentence summary','Plain explanation','Formal definition','Example'])
 suma=[a for a in aa if a.startswith('重點:') and '下一' not in a]+['本單元重點整理']
 pp.append(page('review','重點回顧：把輸入、運算與限制連起來','Review: connect inputs, operations and limitations','用自己的話重述方法，再檢查例外。','Restate the method, then check exceptions.',suma,'練習與檢核完成後，回到整個單元。','After exercises and checks, reconnect the whole unit.','條件、核心方法、算例、限制的知識流程',[],sumz+'\n\n'+CORRECT[k][0],sume+'\n\n'+CORRECT[k][1]))
 pp.append(page('scope','來源與限制：別把特例當成保證','Source and scope: do not turn an example into a guarantee','原始數字可重現，適用條件也必須保留。','Reproduce the values and preserve their scope.',['來源更正與取捨'],'最後核對：哪些是講義例子，哪些是一般結論？','Finally: which statements are examples, and which are general claims?','來源到重算到條件三個關卡，提醒科學圖與角色比喻不同',[],CORRECT[k][0],CORRECT[k][1]))
 if nextpage:pp.append(nextpage)
 else:raise AssertionError('missing next page')
 for i,p in enumerate(pp,1):
  p['page']=i
  def clean(t):
   return t.replace('這個離散的損失','這個分段線性、在接點可能不可微的準則').replace('This discrete loss','This piecewise-linear criterion').replace('所以牛頓法沒有局部極值的問題','所以存在的最優解是全域最優，但普通牛頓步仍可能惡化目標').replace("so Newton's method has no local-extrema problem",'so any existing optimum is global, but plain Newton steps can still worsen the objective').replace('預測在空隙處接近 0','無偏置輸出在空隙處接近 0；有偏置則接近偏置').replace('the prediction goes toward 0','the bias-free output approaches zero; with a bias it approaches that bias')
  p['notes']=[clean(t) for t in p['notes']]
  if 'answer_text' in p:p['answer_text']=[clean(t) for t in p['answer_text']]
  if k==8 and p['kind']=='explain':
   if p['source_anchors']==['常見迷思#3']:p['answer_text']=list(CORRECT[k]);p['notes']=list(CORRECT[k])
   if p['source_anchors']==['常見迷思#2']:p['answer_text']=['邏輯回歸先輸出連續機率，再用門檻分類；名稱不表示它只輸出離散類別。','Logistic regression outputs continuous probabilities before thresholding into classes.'];p['notes']=p['answer_text']
  p['note']='舊版已核對的原生圖重用，新增導學稿、承接與題解；本次未改上游教材。'
  kind_=p.get('base_kind','');prop='ruler'
  if p['kind'] in ['question','quiz']:prop='cards'
  elif p['kind'] in ['calculation','quizanswer','explain']:prop='steps' if p['math'] else 'cards'
  elif any(w in kind_ for w in ['update','newton','rate','rule','bias','learning','batch']):prop='dial'
  elif any(w in kind_ for w in ['coding','four','algebra','errors','data','supervised']):prop='tags'
  elif any(w in kind_ for w in ['matrix','grid','layer','dense','map','roles']):prop='square'
  elif any(w in kind_ for w in ['gate','step','train','output','hidden']):prop='gate'
  elif any(w in kind_ for w in ['anatomy','compare','next','learn','remed']):prop='bridge'
  elif any(w in kind_ for w in ['grad','lms','loss','sum','absolute','squared','kl','huber']):prop='basket'
  p['mascot'].update(prop=prop)
  if p['kind'] in ['calculation','quizanswer','explain']:p['mascot'].update(bubble='先查條件，\n再核對每一步。',bubble_en='Check the scope.\nVerify each step.')
  if p['kind']=='base' and p.get('base_kind')=='cover':p['mascot'].update(bubble='這次要解決\n哪一個問題？',bubble_en='What problem\nare we solving?')
 (unit/'plan.json').write_text(json.dumps(pp,ensure_ascii=False,indent=2));(unit/'圖解證據表.json').write_text(json.dumps(pp,ensure_ascii=False,indent=2))
 lines=[]
 for p in pp:
  lines.append(f"## 第{p['page']}頁／共{len(pp)}頁：{p['title']}\n來源錨點：{', '.join(p['source_anchors'])}；唯一結論：{p['takeaway']}。\n承接：{p['bridge']}。主圖：{p['visual']}。\n逐區：1280×720白底，標題32–36，主圖x50–1000/y170–630，右側x1025–1255為藍頭紫身橙觸角機器人，表情{p['mascot']['mood']}，道具{p['mascot']['prop']}，泡泡「{p['mascot']['bubble']}」。角色是教學比喻，不是實驗數據。底部淺綠結論橫幅。\n數學：{'；'.join('$'+m+'$' for m in p['math']) or '概念流程，無新增計算結果'}。全部數學物件用KaTeX，不用cases或iff；字級不低於16，文字高於實心形狀。\n三拍：先問題與主圖，後證據和角色，最後結論；每拍完整導讀。題目頁不洩答案，下一頁解析。原生connector表示流程，精算科學折線不超200點。\n避免：把來源特例說成普遍保證、漏偏置／平手約定／尺度條件、只有結果沒有同頁算式。中英幾何一致。三輪實際渲染、目視批評修正、coverage/style與原生播放檢查。\n中文自學稿：{p['notes'][0]}\n英文自學稿：{p['notes'][1]}\n")
 (unit/'圖解提示詞.md').write_text('\n'.join(lines));return len(pp)
if __name__=='__main__':
 for k in range(1,10):print('KP',k,make_plan(k),'pages planned before drawing')
