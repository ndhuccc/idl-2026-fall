from pathlib import Path
import ast,numpy as np,json
from kp09_nums import GRID,GRID0,PHI,B,BOUND
R=Path(__file__).parent
module=ast.parse((R/'source-figures.py').read_text());f=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='som_grid');start=next(i for i,n in enumerate(f.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='rng');stop=next(i for i,n in enumerate(f.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))
selected=ast.Module(body=f.body[start:stop],type_ignores=[]);env={'np':np};exec(compile(ast.fix_missing_locations(selected),'unmodified source simulation','exec'),env)
assert np.array_equal(env['w0'],GRID0);assert np.array_equal(env['w'],GRID)
assert np.max(np.abs(np.column_stack([PHI,np.ones(4)])@B-np.array([0,1,1,0])))<1e-12
assert np.isclose(BOUND,(1+np.exp(-2)+2*np.exp(-1))/2)
(R/'output/source-reproduction.json').write_text(json.dumps({'original_source_ast_executed':True,'initial_arrays_bitwise_identical':True,'final_arrays_bitwise_identical':True,'updates':4000,'xor_max_abs_error':float(np.max(np.abs(np.column_stack([PHI,np.ones(4)])@B-np.array([0,1,1,0]))))},indent=2));print('PASS: original simulation and independent reproduction match bit for bit; XOR residual <1e-12.')
