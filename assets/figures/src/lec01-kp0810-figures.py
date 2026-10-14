"""Rebuild Ch01 KP08–KP10 lecture figures. Fixed teaching values only."""
from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "svg.fonttype": "none", "figure.facecolor": "#f6f4ee", "axes.facecolor": "#f6f4ee", "savefig.facecolor": "#f6f4ee"})
ROOT = Path(__file__).resolve().parents[1]
INK, BLUE, TEAL, CORAL, MUTED = "#263849", "#315f85", "#158b84", "#d16b50", "#677782"

def save(fig, name):
    fig.savefig(ROOT / f"{name}.png", dpi=175, bbox_inches="tight", facecolor=fig.get_facecolor())
    svg = ROOT / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    svg.write_text(re.sub(r"(?m)[ \t]+$", "", svg.read_text(encoding="utf-8")), encoding="utf-8")
    plt.close(fig)

def canvas(w=10, h=4.0):
    fig, ax = plt.subplots(figsize=(w,h))
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    return fig,ax

def box(ax,x,y,w,h,text,edge=BLUE,fc="#ffffff",size=12):
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.025",linewidth=2,edgecolor=edge,facecolor=fc)
    ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha="center",va="center",color=INK,fontsize=size,weight="bold")

def arrow(ax,x1,y1,x2,y2,color=MUTED):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=17,linewidth=2,color=color))

# KP08: milestones (question changes, not a causal lineage).
fig,ax=canvas(12,3.8)
dates=[1943,1949,1958,1960,1969,1986]
labels=["M–P\nlogic", "Hebb\nplasticity", "Perceptron\nexamples", "ADALINE\nLMS", "Perceptrons\nlimits", "Backprop\nmultilayer"]
xpos=[.10,.25,.41,.56,.72,.89]
ax.plot([.08,.91],[.5,.5],color=MUTED,lw=2)
for i,(x,d,l) in enumerate(zip(xpos,dates,labels)):
    ax.scatter([x],[.5],s=160,color=TEAL if i in (2,5) else BLUE,zorder=3)
    yy=.73 if i%2==0 else .26
    ax.plot([x,x],[.5,yy-.04 if yy>.5 else yy+.11],color=MUTED,lw=1.5)
    ax.text(x,yy,f"{d}\n{l}",ha="center",va="center",fontsize=11,color=INK)
ax.text(.5,.95,"What changed: logic → adaptable weights → multilayer learning",ha="center",va="top",color=INK,fontsize=15,weight="bold")
save(fig,"lec01-kp08-early-timeline")

# KP08: four XOR points. Candidate line is illustrative, inequalities in lecture prove the claim.
fig,ax=plt.subplots(figsize=(6,5.2));fig.patch.set_facecolor("#f6f4ee");ax.set_facecolor("#f6f4ee")
for x,y,v in [(0,0,0),(1,0,1),(0,1,1),(1,1,0)]:
    ax.scatter(x,y,s=850,c=CORAL if v else BLUE,edgecolor="white",lw=3,zorder=3)
    ax.text(x,y,str(v),ha="center",va="center",color="white",fontsize=20,weight="bold",zorder=4)
ax.plot([-.2,1.2],[1.45,.05],linestyle="--",color=TEAL,lw=2)
ax.set_xlim(-.25,1.25);ax.set_ylim(-.25,1.25);ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xlabel("x1");ax.set_ylabel("x2");ax.set_title("XOR: opposite corners share a label",color=INK)
ax.grid(alpha=.15);ax.set_aspect("equal")
save(fig,"lec01-kp08-xor-four-points")

fig,ax=plt.subplots(figsize=(8,3.8));fig.patch.set_facecolor("#f6f4ee");ax.set_facecolor("#f6f4ee")
ax.barh([1,0],[0,.08],color=[BLUE,TEAL],height=.45)
ax.set_yticks([1,0],["Perceptron", "ADALINE LMS"]);ax.set_xlim(0,.105);ax.set_xlabel("Update to w1 (x=(1,0), y=1, a=0.2, eta=0.1)")
ax.text(.005,1,"0: sign correct",va="center",fontsize=12,color=INK)
ax.text(.082,0,"0.08",va="center",fontsize=12,color=INK)
ax.spines[["top","right"]].set_visible(False)
save(fig,"lec01-kp08-update-comparison")

# KP09: structural diagram in notes (native elements in slides).
fig,ax=canvas(9.5,4.5)
ax.text(.5,.94,"One 2×2 filter is reused at nine 4×4 locations",ha="center",fontsize=15,weight="bold",color=INK)
x0,y0=.07,.19;s=.12
for r in range(4):
    for c in range(4):
        rect=Rectangle((x0+c*s,y0+(3-r)*s),s-.005,s-.005,edgecolor="#bcc6ca",facecolor="#bfe1db" if r<2 and c<2 else "white",lw=1)
        ax.add_patch(rect)
ax.text(.30,.08,"4×4 input",ha="center",color=INK)
box(ax,.61,.51,.14,.16,"2×2\nfilter",TEAL,size=12)
box(ax,.79,.51,.14,.16,"3×3\noutput",BLUE,size=12)
arrow(ax,.55,.58,.60,.58);arrow(ax,.76,.58,.78,.58)
ax.text(.75,.28,"2 filters × (4 weights + 1 bias) = 10",ha="center",color=INK,fontsize=12)
save(fig,"lec01-kp09-shared-kernel")

fig,ax=plt.subplots(figsize=(8,3.6));fig.patch.set_facecolor("#f6f4ee");ax.set_facecolor("#f6f4ee")
ax.barh([1,0],[306,10],height=.5,color=[CORAL,TEAL]);ax.set_yticks([1,0],["Dense, 16 → 18","Conv, 2 filters"])
ax.set_xlim(0,345);ax.set_xlabel("Parameters in one stipulated layer")
for y,n in [(1,306),(0,10)]:ax.text(n+5,y,str(n),va="center",color=INK,fontsize=13,weight="bold")
ax.spines[["top","right"]].set_visible(False)
save(fig,"lec01-kp09-parameter-count")

fig,ax=canvas(12,4)
years=[1980,1986,1989,1998,2006,2012,2017]
xx=[.08,.21,.34,.47,.60,.74,.89]
names=["Neocognitron", "Backprop", "Early CNN", "LeNet-5", "Layer-wise", "AlexNet", "Transformer"]
ax.plot([.06,.92],[.51,.51],color=MUTED,lw=2)
for i,(x,y,name) in enumerate(zip(xx,years,names)):
    ax.scatter(x,.51,s=155,c=TEAL if i in (4,5,6) else BLUE,zorder=3)
    yy=.74 if i%2==0 else .27
    ax.text(x,yy,f"{y}\n{name}",ha="center",va="center",fontsize=10.5,color=INK)
    ax.plot([x,x],[.51, .61 if yy>.5 else .39],color=MUTED,lw=1)
ax.text(.5,.96,"Selected milestones, not a single replacement chain",ha="center",va="top",fontsize=15,color=INK,weight="bold")
save(fig,"lec01-kp09-revival-timeline")

fig,ax=canvas(10,4.4)
left=[("Image",.74),("Sequence",.55),("Graph",.36),("Generation",.17)]
right=[("CNN",.74),("Attention / RNN",.55),("GNN",.36),("VAE / GAN / diffusion",.17)]
for (a,y),(b,yr) in zip(left,right):
    box(ax,.06,y,.29,.11,a,BLUE,size=12);box(ax,.65,yr,.30,.11,b,TEAL,size=12);arrow(ax,.36,y+.055,.64,yr+.055)
ax.text(.5,.94,"Data and objective suggest candidates",ha="center",fontsize=15,color=INK,weight="bold")
save(fig,"lec01-kp09-task-architecture-map")

fig,ax=canvas(10,4.6)
box(ax,.04,.41,.2,.2,"Feedback?",CORAL,size=14)
branches=[("Labels → class",.69),("Labels → value",.52),("No label → structure",.35),("Action → reward",.18)]
for name,y in branches:
    box(ax,.39,y,.35,.13,name,BLUE if y>.39 else TEAL,size=12)
    arrow(ax,.25,.51,.38,y+.065)
box(ax,.79,.41,.17,.2,"Evaluate\nnew cases",TEAL,size=12)
for _,y in branches:arrow(ax,.75,y+.065,.78,.51)
ax.text(.5,.96,"Input + feedback + output → task → evaluation",ha="center",va="top",fontsize=15,color=INK,weight="bold")
save(fig,"lec01-kp10-task-feedback-map")

fig,ax=canvas(10,4.1)
headers=["Task","Feedback","Output","Candidate","Check"]
rows=[("Spam","Labels","Class","Logistic / tree","New mail"),("Sensors","None","Structure","PCA / clustering","Stability"),("Robot","Rewards","Action","Policy model","New trials")]
xs=[.03,.20,.39,.56,.80];ws=[.16,.18,.16,.23,.17]
for x,w,h in zip(xs,ws,headers):box(ax,x,.72,w,.12,h,BLUE,size=10.5)
for j,row in enumerate(rows):
    y=.53-j*.19
    for i,(x,w,val) in enumerate(zip(xs,ws,row)):
        box(ax,x,y,w,.13,val,TEAL if i==4 else "#b8c8cf",size=10.5)
ax.text(.5,.97,"Same questions across different tasks",ha="center",va="top",fontsize=15,color=INK,weight="bold")
save(fig,"lec01-kp10-method-matrix")

print("Generated 9 PNG/SVG pairs")
