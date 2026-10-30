from pathlib import Path
import ast,numpy as np,json
from kp10_nums import XS,NS,approx,save
R=Path(__file__).parent;mod=ast.parse((R/'source-figures.py').read_text());fn=next(n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name=='uat_staircase');sig=next(n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name=='sigmoid');start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='f');end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.FunctionDef) and n.name=='approx');selected=ast.Module(body=[sig]+fn.body[start:end+1],type_ignores=[]);env={'np':np};exec(compile(ast.fix_missing_locations(selected),'unmodified source numeric AST','exec'),env)
checks=[]
for n in NS:
 a=env['approx'](n);b=approx(n);assert np.array_equal(a,b),n;checks.append(dict(N=n,all_4001_values_bitwise_equal=True,sampled_error=float(np.abs(env['f'](XS)-a).max())))
(R/'output/source-reproduction.json').write_text(json.dumps(dict(source_ast_executed=True,grid_points=4001,k=2000,checks=checks,warning='Grid maxima are not certified suprema.'),indent=2));print('PASS: all six source arrays match bit for bit.')
