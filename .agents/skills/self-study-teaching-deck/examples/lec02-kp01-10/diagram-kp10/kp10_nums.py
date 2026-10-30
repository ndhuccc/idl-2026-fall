from pathlib import Path
import numpy as np,json
R=Path(__file__).parent
XS=np.linspace(0,1,4001);NS=[3,5,10,20,40,80]
def sigmoid(x):return 0.5*(1+np.tanh(0.5*x))
def approx(n,x=XS,k=2000):
 t=np.linspace(0,1,n+1);v=np.sin(2*np.pi*(t[:-1]+t[1:])/2)
 return v[0]+sum((v[i]-v[i-1])*sigmoid(k*(x-t[i])) for i in range(1,n))
def save():
 errs=[float(np.max(np.abs(np.sin(2*np.pi*XS)-approx(n)))) for n in NS]
 for got,target in zip(errs,[.866,.588,.309,.156,.0785,.0393]):assert abs(got-target)<.0005,(got,target)
 # Endpoint error is a rigorous lower bound on true uniform error; sampled maxima are not sup certificates.
 endpoint=[float(abs(approx(n,np.array([0.]))[0])) for n in NS]
 v=(np.arange(4)+.5)/4;alpha=np.diff(v);assert np.allclose(v,[.125,.375,.625,.875]) and np.allclose(alpha,.25)
 data=dict(source_grid_points=4001,sigmoid_k=2000,N_means='number of cells; N-1 transition sigmoids plus constant bias, or N hidden units when representing the constant by a constant unit',N=NS,sampled_max_errors=errs,endpoint_errors=endpoint,ideal_staircase_lipschitz_bound=[float(np.pi/n) for n in NS],N320_sampled_error=float(np.max(np.abs(np.sin(2*np.pi*XS)-approx(320)))),practice1=dict(midpoints=v.tolist(),jumps=alpha.tolist(),thresholds=[.25,.5,.75],ideal_value_at_point6=.625,ideal_max_error=.125),practice2=dict(strict_ideal_bound_N_min=int(np.floor(np.pi/.05))+1),limitations=['Finite sigmoid sums are smooth, not exactly cellwise constant.','A maximum on 4001 samples is not a certified supremum.','pi/N bounds the ideal midpoint staircase, not automatically every finite-k approximation.'])
 (R/'數值驗算.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));return data
if __name__=='__main__':print(json.dumps(save(),ensure_ascii=False,indent=2))
