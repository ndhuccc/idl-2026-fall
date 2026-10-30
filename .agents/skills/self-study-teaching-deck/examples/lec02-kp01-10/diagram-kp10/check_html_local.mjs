#!/usr/bin/env node
// 驗證轉出的 HTML 與來源 .jbd.json 一致。需要 playwright-core 與一個 Chromium:
//   npm i --no-save playwright-core        (不列入本技能的相依,避免拖慢網站建置)
//   node check_html.mjs <HTML 資料夾> <來源 .jbd.json 資料夾> [chromium 執行檔路徑]
//
// 逐份檢查:
//   1. 沒有頁面錯誤,也沒有對外網路請求(離線可用)。
//   2. 所有 <img> 都成功載入;沒有 KaTeX 錯誤。
//   3. 每個文字/圖形節點實際顯示的字級、粗細、對齊,都和 JSON 內的設定相同
//      (因超出方框而被縮小的節點另外計數,數量應該很少)。
// 全部通過才以 0 結束。
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require('/opt/codex/runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages/playwright/driver/package')); } catch { console.error('請先執行:npm i --no-save playwright-core'); process.exit(2); }

const [, , htmlDir, jsonDir, exe] = process.argv;
if (!htmlDir || !jsonDir) { console.error('用法:node check_html.mjs <HTML 資料夾> <JSON 資料夾> [chromium 路徑]'); process.exit(2); }
const browser = await chromium.launch(exe ? { executablePath: exe, args: ['--no-sandbox'] } : {});
const pg = await browser.newPage({ viewport: { width: 1280, height: 760 } });
let files = 0, nodes = 0, problems = 0, shrunkAll = 0;
for (const f of fs.readdirSync(htmlDir).filter((x) => x.endsWith('.html')).sort()) {
  const jf = path.join(jsonDir, f.replace(/\.html$/, '.jbd.json'));
  if (!fs.existsSync(jf)) continue;
  const d = JSON.parse(fs.readFileSync(jf, 'utf8'));
  const exp = Object.values(d.nodes).filter((n) => (n.type === 'text' || n.type === 'shape') && n.text).map((n) => ({
    id: n.id, size: n.style.fontSize ?? 20, weight: String(n.style.fontWeight ?? (n.type === 'shape' ? 720 : 400)),
    align: n.style.textAlign ?? (n.type === 'shape' ? 'center' : 'left'),
  }));
  await pg.goto('about:blank');
  const errs = [], ext = [];
  const onErr = (e) => errs.push(String(e)), onMsg = (m) => { if (m.type() === 'error') errs.push(m.text()); };
  const onReq = (r) => { const u = r.url(); if (!/^(file|data|blob):/.test(u)) ext.push(u.slice(0, 60)); };
  pg.on('pageerror', onErr); pg.on('console', onMsg); pg.on('request', onReq);
  await pg.setContent(fs.readFileSync(path.resolve(htmlDir, f), 'utf8'), {waitUntil: 'load'});
  const res = await pg.evaluate(async (exp) => {
    await document.fonts.ready; await new Promise((r) => setTimeout(r, 400));
    const imgs = [...document.querySelectorAll('img')];
    await Promise.all(imgs.map((i) => i.decode().catch(() => {})));
    const out = []; let shr = 0;
    for (const e of exp) {
      const el = document.querySelector(`[data-n="${CSS.escape(e.id)}"] .copy`);
      if (!el) { out.push([e.id, '找不到節點']); continue; }
      const cs = getComputedStyle(el);
      const k = el.dataset.shrunk ? parseFloat(el.dataset.shrunk) : 1; if (el.dataset.shrunk) shr++;
      const sz = parseFloat(cs.fontSize) / k;
      if (Math.abs(sz - e.size) > 0.6) out.push([e.id, `字級 ${sz.toFixed(1)} ≠ ${e.size}`]);
      else if (cs.fontWeight !== e.weight) out.push([e.id, `粗細 ${cs.fontWeight} ≠ ${e.weight}`]);
      else if (cs.textAlign !== e.align && !(e.align === 'left' && cs.textAlign === 'start')) out.push([e.id, `對齊 ${cs.textAlign} ≠ ${e.align}`]);
    }
    for (let i = 0; i < STEPS.length; i++) go(i, true); // 走完全部步驟,確認不丟例外
    return { out, shr, badImg: imgs.filter((i) => i.naturalWidth === 0).length, kerr: document.querySelectorAll('.katex-error').length };
  }, exp);
  pg.off('pageerror', onErr); pg.off('console', onMsg); pg.off('request', onReq);
  files++; nodes += exp.length; shrunkAll += res.shr;
  const n = res.out.length + res.badImg + res.kerr + errs.length + ext.length;
  if (n) {
    problems += n;
    console.log('問題', f, JSON.stringify({ 字型: res.out.slice(0, 2), 圖片失敗: res.badImg, 公式錯誤: res.kerr, 頁面錯誤: errs.slice(0, 2), 外部請求: ext.slice(0, 2) }));
  }
}
console.log(`檢查 ${files} 份、${nodes} 個節點:${problems} 個問題;${shrunkAll} 個節點因超出方框而縮小`);
await browser.close();
process.exit(problems ? 1 : 0);
