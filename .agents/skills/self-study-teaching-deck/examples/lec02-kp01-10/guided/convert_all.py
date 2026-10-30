from pathlib import Path
import subprocess,concurrent.futures
R=Path(__file__).parent;script=R/'toolchain/.agents/skills/slides-to-html/scripts/jbd_to_html.mjs'
def one(k):
 u=R/f'KP{k:02d}';out=u/'output';files=sorted(out.glob('*.jbd.json'));p=subprocess.run(['node',str(script),'--strict','--out-dir',str(out),*[str(f) for f in files]],capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr;return p.stdout.strip()
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 for result in ex.map(one,range(1,10)):print(result)

subprocess.run(["node",str(R/"enhance_notes.mjs")],check=True)
