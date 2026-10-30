from pathlib import Path
import json,math,importlib.util,sys
import numpy as np
R=Path(__file__).parent

def calculate():
 out={}
 for k in range(2,10):
  p=R/f'KP{k:02d}/原始建頁來源/kp{k:02d}_nums.py';sp=importlib.util.spec_from_file_location(f'original_nums_{k}',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.save();out[f'KP{k:02d}_source']='rerun original numeric script and assertions in isolated new source copy'
 assert sum([1,0,1])==2
 assert [int(2>=t) for t in [1,2,3]]==[1,1,0]
 assert [int(a+b>=2) for a,b in [(0,0),(0,1),(1,0),(1,1)]]==[0,0,0,1]
 assert [int(a+b>=1) for a,b in [(0,0),(0,1),(1,0),(1,1)]]==[0,1,1,1]
 assert np.array_equal(np.array([1,1,-1])-np.array([1,0,1]),[0,1,-2])
 assert math.isclose(math.hypot(3,4),5) and abs(-5)/5==1
 assert np.array_equal(np.array([1,2])@np.array([[1,0],[1,1]]),[3,2])
 sig=1/(1+math.exp(-1));der=sig*(1-sig);assert abs(sig-.7310585786300049)<1e-15
 residual=np.array([2,-1,.5,6]);sq=float((residual**2).sum());ab=float(np.abs(residual).sum());h=np.where(np.abs(residual)<=1,.5*residual**2,np.abs(residual)-.5);assert sq==41.25 and ab==9.5 and h.sum()==7.625
 x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]],float);y=np.array([0,1,1,1]);w=np.array([.2,.2,.3]);res=y-x@w;g=-x.T@res;new=w-.1*g;loss=.5*np.sum((y-x@new)**2);assert np.allclose(g,[-.8,-.8,-1]) and np.allclose(new,[.28,.28,.4]) and abs(loss-.1832)<1e-15
 assert np.allclose(np.array([1,1])+.5*(np.array([.8,.6])-np.array([1,1])),[.9,.8])
 out['added_practice']=dict(kp01=dict(sum=2,threshold_interval='1<theta<=2',not_weights=[-1,.5]),kp02=dict(updated_weights=[0,1,-2],first_reverse_update_visit=4),kp03=dict(new_weights=[2,1,-.5],new_score=1.5,distance=1),kp04=dict(AND_weights=[1,1,-1.5],XNOR='inequality contradiction'),kp05=dict(merged_matrix=[3,2],xor10=dict(s1=0,h1=1,s2=.5,h2=1,s3=0,o=1)),kp06=dict(sigmoid=sig,derivative=der,uniform=[1/3]*3,negative_leaky_slope=.01),kp07=dict(squared_sum=sq,absolute_sum=ab,huber_terms=h.tolist(),huber_sum=float(h.sum()),means=[sq/4,ab/4,float(h.sum()/4)],uniform_ce=math.log(3),hinge=1.8),kp08=dict(prediction=(x@w).tolist(),residual=res.tolist(),gradient=g.tolist(),new_weights=new.tolist(),new_loss=loss,probability=1/(1+math.exp(-.4))),kp09=dict(midpoint_phi=math.exp(-.5),midpoint_sum=2*math.exp(-.5),winner_distance=[1,math.sqrt(.2)],winner=2,new_weights=[.9,.8]))
 (R/'新增數值驗算.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('PASS: 8 source numeric scripts plus all new practice assertions.');return out
if __name__=='__main__':calculate()
