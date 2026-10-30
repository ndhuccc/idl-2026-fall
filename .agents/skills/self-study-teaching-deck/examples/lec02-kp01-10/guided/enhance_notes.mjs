// Local output-only enhancement: source JBD and upstream converter remain unchanged.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const katex=require('./toolchain/.agents/skills/slides-to-html/node_modules/katex');
const root=path.dirname(new URL(import.meta.url).pathname);
const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
function render(s){
 let out='',pos=0;const rx=/\$\$([\s\S]*?)\$\$|\$([^$]+)\$/g;let m;
 while((m=rx.exec(s))){out+=esc(s.slice(pos,m.index));out+=katex.renderToString(m[1]??m[2],{displayMode:m[1]!==undefined,throwOnError:false,strict:'ignore',trust:false});pos=rx.lastIndex;}
 return out+esc(s.slice(pos));
}
for(let k=1;k<=9;k++){
 const folder=path.join(root,`KP${String(k).padStart(2,'0')}`,'output');
 for(const f of fs.readdirSync(folder).filter(f=>f.endsWith('.jbd.json'))){
  const doc=JSON.parse(fs.readFileSync(path.join(folder,f),'utf8'));const hp=path.join(folder,f.replace('.jbd.json','.html'));let h=fs.readFileSync(hp,'utf8');
  const data=JSON.stringify(doc.steps.map(s=>render(s.speakerNotes||''))).replaceAll('<','\\u003c');
  h=h.replace('const AUD=',`const STUDY_NOTES=${data};\nconst AUD=`);
  const old="notes.innerHTML=(s.notes?'<h4>講者備註</h4>'+esc(s.notes):'')+(s.tr?'<h4>語音逐字稿</h4>'+esc(s.tr):'');";
  if(!h.includes(old))throw new Error('Unexpected notes renderer');
  h=h.replace(old,`notes.innerHTML='<h4>${f.includes('-en.')?'Study notes':'自學導讀'}</h4>'+STUDY_NOTES[cur];`);
  h=h.replace('font-size:15px;line-height:1.65;padding:12px 18px','font-size:18px;line-height:1.65;padding:18px 24px');
  const oldLookup="const nodeEls=id=>document.querySelectorAll('[data-n=\"'+(window.CSS&&CSS.escape?CSS.escape(id):id)+'\"]');";
  if(!h.includes(oldLookup))throw new Error('Unexpected node lookup');
  h=h.replace(oldLookup,`const NODE_CACHE=new Map();document.querySelectorAll('[data-n]').forEach(el=>{const id=el.dataset.n;if(!NODE_CACHE.has(id))NODE_CACHE.set(id,[]);NODE_CACHE.get(id).push(el)});const nodeEls=id=>NODE_CACHE.get(id)||[];`);
  fs.writeFileSync(hp,h);
 }
}
console.log('Enhanced 18 offline HTML files with pre-rendered study-note math.');
