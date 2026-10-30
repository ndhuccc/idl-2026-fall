"""原生視覺主角；設計先寫入視覺改版提示詞，角色只作概念比喻。"""
import math
# kind: prop, expression, Traditional Chinese speech, English speech
STORY={
'cover':('routes','happy','兩條路，同一招！','Two routes.\nOne rule!'),
'coding':('tags','happy','換標籤，\n也換輸出！','New labels,\nnew outputs!'),
'four':('tags','think','先配對，再相減。','Pair them,\nthen subtract.'),
'algebra':('tags','happy','反號相減，\n變成兩倍！','Opposite signs:\ndouble the label!'),
'rule':('rate','think','換刻度，\n方向不變。','New scale,\nsame direction.'),
'normal':('square','think','沿邊界走，\n和權重垂直。','Boundary steps:\nnormal to weights.'),
'distance':('ruler','think','分數要先換成距離！','Score needs\na distance scale!'),
'labeldistance':('tags','think','還要看正確標籤！','Check the\ntrue label too!'),
'tie':('warning','confused','站在線上，也會猜錯？','On the line,\nbut wrong?'),
'distancesA':('ruler','think','別急著取小數！','Round only\nat the end!'),
'distancesB':('ruler','happy','正負，要連標籤看。','The label\nsets the sign.'),
'loss':('basket','confused','零損失，不一定全對！','Zero loss,\nyet a mistake!'),
'gradient':('slope','think','順著下降方向走。','Follow the\ndescent direction.'),
'descent':('routes','happy','學習率先對齊！','Match the\nrates first!'),
'numeric_update':('dial','work','修正錯點，\n轉動旋鈕！','A mistake?\nAdjust the knobs!'),
'numeric_score':('bridge','happy','這次跨過零了！','This time,\nwe cross zero!'),
'trace':('steps','happy','每一步都對得上！','Every step\nmatches!'),
'gate':('gate','work','猜錯，才開工！','Work only\non a mistake!'),
'next':('warning','confused','一條線，真的夠嗎？','Is one line\nalways enough?'),
'practice1':('dial','think','這次換你轉旋鈕。','Your turn\nto adjust!'),
'practice2':('ruler','think','找最短的路！','Find the\nshortest path!'),
'practice3':('tags','think','四種配對，別漏掉！','Try all four\npairs!'),
'quiz1':('cards','think','我只會輸出哪些值？','Which values\ncan I output?'),
'quiz2':('dial','think','把反號代進去。','Substitute\nopposite signs.'),
'quiz3':('slope','think','先分清兩個函數。','Two functions.\nKeep them distinct.'),
'quiz4':('routes','think','兩條路，怎麼會合？','How do the\nroutes meet?'),
}
def draw(K,kind):
 P=K.P;tr=K.tr;c=K.BLUE;pu=K.PURPLE;g=K.GREEN;red=K.RED;gold=K.ORANGE
 prop,mood,zh,en=STORY[kind]
 # Approx. 20% of the content area is a role-based mini-scene.
 P.ellipse(1140,400,110,173,fill='#f1f6fc',z=1)
 # Dialogue appears with the relevant derivation beat, rather than before the condition.
 P.rect(1025,162,230,94,fill='#fff5da',stroke='#dfba65',sw=2,radius=19,z=10)
 P.poly([(1110,256),(1120,276),(1140,256)],fill='#fff5da',stroke='#dfba65',sw=2,z=10)
 K.text(1036,179,208,70,tr(zh,en),23 if K.LANG=='zh' else 20,K.INK,True,'center',name='concept-dialogue')
 # Fixed mascot: sky-blue head, purple body, single orange antenna, expressive face.
 P.ellipse(1141,546,72,11,fill='#dce6f1',z=2)
 for x in (1112,1170):
  P.line([(x,485),(x,528)],pu,13,z=12);P.rect(x-16,524,35,13,fill=pu,radius=5,z=13)
 P.rect(1090,399,101,100,fill='#e3d9f5',stroke=pu,sw=3,radius=24,z=12)
 P.rect(1121,422,38,34,fill='white',stroke=pu,sw=2,radius=8,z=13)
 # A pulse line denotes the model, never a numerical claim.
 P.line([(1127,439),(1134,439),(1140,429),(1146,448),(1152,439)],pu,3,z=14)
 P.line([(1141,311),(1141,292)],c,4,z=12);P.ellipse(1141,289,8,fill='#efb84c',z=13)
 P.rect(1076,316,130,87,fill='#d9effa',stroke=c,sw=3,radius=28,z=14)
 P.ellipse(1075,356,8,16,fill='#b0d4ed',z=13);P.ellipse(1207,356,8,16,fill='#b0d4ed',z=13)
 if mood=='happy':
  for x in (1113,1169):P.line([(x-8,349),(x,341),(x+8,349)],K.INK,3,z=16)
  P.line([(1126,370),(1141,380),(1156,370)],K.INK,3,z=16)
 elif mood=='confused':
  P.ellipse(1114,350,5,8,fill=K.INK,z=16);P.line([(1160,343),(1179,349)],K.INK,4,z=16)
  P.ellipse(1141,377,8,6,fill='white',stroke=K.INK,sw=2,z=16)
  P.poly([(1209,314),(1202,330),(1215,330)],fill='#78bfdf',z=16)
 else:
  for x in (1114,1169):P.ellipse(x,350,5,7,fill=K.INK,z=16)
  P.line([(1129,377),(1154,377 if mood=='work' else 372)],K.INK,3,z=16)
  if mood=='think':P.line([(1103,335),(1122,332)],K.INK,3,z=16)
 # Hands aim toward a concept-specific prop.
 left=[(1094,419),(1061,451),(1078,478)]
 right=[(1185,421),(1221,447),(1207,478)]
 if prop in ('ruler','square','slope'):
  left=[(1094,419),(1060,395),(1039,369)]
 elif prop=='warning':
  left=[(1094,419),(1045,390),(1069,340)]
 elif prop in ('routes','bridge'):
  left=[(1094,419),(1060,405),(1038,418)]
  right=[(1185,421),(1217,405),(1236,418)]
 for arm in (left,right):
  P.line(arm,c,10,z=11)
  P.ellipse(*arm[-1],10,fill='#d9effa',stroke=c,sw=2,z=18)
 if prop in ('tags','cards'):
  for x,val,col in [(1066,'-1',c),(1163,'+1',g)]:
   if prop=='cards':val='?'
   P.line([(x+30,473),(x+30,508)],col,3,z=18)
   P.rect(x,444,66,51,fill='white',stroke=col,sw=3,radius=10,z=19)
   K.eq(x,448,66,val,23,h=43,c=col)
 elif prop=='ruler':
  n=P.rect(1046,463,191,32,fill='#ffe4a7',stroke=gold,sw=2,radius=5,z=19)
  for i in range(11):P.line([(1054+i*17,465),(1054+i*17,480 if i%5 else 488)],gold,2,z=20)
 elif prop=='warning':
  P.poly([(1140,437),(1186,511),(1094,511)],fill='#fff0ce',stroke=gold,sw=3,z=19)
  K.label(1120,459,40,50,'!',36,gold)
 elif prop=='square':
  P.poly([(1080,448),(1105,448),(1105,496),(1204,496),(1204,520),(1080,520)],fill='#ffe4a7',stroke=gold,sw=2,z=19)
  for i in range(6):P.line([(1120+i*15,498),(1120+i*15,509)],gold,2,z=20)
 elif prop=='rate':
  P.ellipse(1141,484,38,fill='#eee7f8',stroke=pu,sw=3,z=19)
  for a in range(0,360,45):
   t=math.radians(a);P.line([(1141+29*math.cos(t),484+29*math.sin(t)),(1141+35*math.cos(t),484+35*math.sin(t))],pu,2,z=20)
  P.line([(1141,484),(1162,463)],pu,5,z=21)
 elif prop=='dial':
  P.rect(1049,451,189,67,fill='#f8fbfe',stroke=pu,sw=2,radius=13,z=19)
  for j in range(3):
   x=1083+j*61;P.ellipse(x,483,19,fill='#eee7f8',stroke=pu,sw=2,z=20)
   P.line([(x,483),(x+10,473 if j else 483)],pu,3,z=21)
 elif prop=='basket':
  P.poly([(1080,464),(1200,464),(1183,521),(1097,521)],fill='#fff0e8',stroke=red,sw=3,z=19)
  for x in (1112,1165):P.poly([(x,446),(x-11,465),(x+11,465)],fill=g,z=20)
  K.label(1110,479,60,34,'?',25,red)
 elif prop=='slope':
  P.poly([(1060,454),(1225,520),(1060,520)],fill='#e8e0f6',stroke=pu,sw=2,z=19)
  P.ellipse(1110,463,11,fill=g,z=20);K.arrow(1132,455,1215,488,g,4)
 elif prop=='steps':
  pts=[(1060,520),(1111,520),(1111,498),(1162,498),(1162,476),(1226,476)]
  P.line(pts,pu,4,z=19)
  K.arrow(1080,495,1206,450,g,4)
 elif prop in ('routes','bridge'):
  if prop=='routes':
   K.arrow(1066,459,1140,501,c,4);K.arrow(1217,459,1140,501,pu,4)
   P.ellipse(1140,501,10,fill=g,z=20)
  else:
   P.line([(1060,501),(1089,475),(1192,475),(1225,501)],g,4,z=19)
   K.eq(1100,487,80,'0',22,h=40,c=K.MUT);K.arrow(1090,453,1190,453,g,4)
 elif prop=='gate':
  P.rect(1080,449,118,67,fill='#f4faf6',stroke=g,sw=2,radius=10,z=19)
  P.ellipse(1111,480,19,fill='#e7f4ec',stroke=g,sw=2,z=20)
  P.line([(1111,480),(1173,455)],g,7,z=21)
  P.ellipse(1173,480,6,fill=g,z=20)
