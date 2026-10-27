"""Publish one finished unit (slides + lecture figures + captions + records) under a repo-wide lock, so several agents can publish safely in parallel.
    python3 publish_unit.py --unit-dir <dir> --kp kp09 --pages 10 --lecture lec05-kp09-cost-functions --map map.json --caps caps.json \
        --remove "lec05-kp09-old.png,lec05-kp09-old2.png" --en-before "延伸=Extension" \
        --changelog "[更新] KP09 簡報以 view-from-text-v3 重做並發佈:..." --state "KP09 簡報以 v3 重做並發佈 ...;下一步 ..." --log "依教授指示 ... 重做 KP09 並發佈 ..." [--dry-run]
<dir> holds kpNN_pages.py, kpNN_text.py, kpNN_nums.py (+ data), out/ (the HTML-designed pages), W_zh/draft.jbd.json, W_en/draft.jbd.json (from run_unit.sh), map.json, caps.json.
--lecture is the file stem under lectures/ (zh) and lectures/en/; slides are slides/<stem>.jbd.json and <stem>-en.jbd.json.
--en-before maps a Chinese heading prefix used in caps.json 'before' to the English heading prefix (second pass for the English lecture).
--dry-run stops after the checks (nothing is committed; run `git -C <repo> checkout -- . && git -C <repo> clean -fdq assets` to discard)."""
import argparse, os, re, shutil, subprocess, sys, time, json, glob
REPO = "/home/user/pr-2026-fall"; OTHERS = ["/home/user/course-template", "/home/user/idl-2026-fall"]; JARVIS = "/home/user/my-jarvis"
SK = f"{REPO}/.agents/skills/view-from-text-v3"; LOCK = "/tmp/pr-publish.lock"
ap = argparse.ArgumentParser()
for a in ("unit-dir", "kp", "pages", "lecture", "map", "caps", "remove", "en-before", "changelog", "state", "log"): ap.add_argument("--" + a, default="")
ap.add_argument("--dry-run", action="store_true"); A = ap.parse_args()
def sh(cmd, cwd=REPO, check=True):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode: raise SystemExit(f"FAILED: {cmd}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.stdout + r.stderr
def acquire():
    for _ in range(360):
        try: os.mkdir(LOCK); return
        except FileExistsError:
            if time.time() - os.path.getmtime(LOCK) > 1800: shutil.rmtree(LOCK, ignore_errors=True)
            time.sleep(5)
    raise SystemExit("could not get the publish lock")
def main():
    d = os.path.abspath(A.unit_dir); kp = A.kp; T = subprocess.check_output("TZ=Asia/Taipei date '+%Y-%m-%d %H:%M'", shell=True, text=True).strip()
    for f in (f"{d}/W_zh/draft.jbd.json", f"{d}/W_en/draft.jbd.json", f"{d}/{A.map}", f"{d}/{A.caps}", f"{d}/{kp}_pages.py", f"{d}/{kp}_text.py"):
        if not os.path.exists(f): raise SystemExit(f"missing {f}")
    if sh("git status --porcelain").strip(): raise SystemExit("repo working tree is not clean:\n" + sh("git status --porcelain"))
    sh("git pull --rebase origin main")
    stem = A.lecture; zh_md = f"lectures/{stem}.md"; en_md = f"lectures/en/{stem}.md"
    shutil.copy(f"{d}/W_zh/draft.jbd.json", f"{REPO}/slides/{stem}.jbd.json"); shutil.copy(f"{d}/W_en/draft.jbd.json", f"{REPO}/slides/{stem}-en.jbd.json")
    sh(f"python3 {SK}/tools/make_specs.py {d} {kp} {A.pages}")
    print(sh(f"python3 -W ignore {SK}/tools/gen_figs.py {d}/{kp}_specs.py {d}/{A.map} assets/figures").splitlines()[-1])
    rm = [x for x in A.remove.split(",") if x]
    for x in rm:   # drop caption blocks of removed figures from both lectures
        fid = x[:-4] if x.endswith(".png") else x
        for path in (zh_md, en_md):
            s = open(f"{REPO}/{path}").read(); s2 = re.sub(r"(圖說|Figure) `%s`.*?%s(-en)?\.png\)\n\n?" % (re.escape(fid), re.escape(fid.split(kp + "-")[-1])), "", s, flags=re.S)
            open(f"{REPO}/{path}", "w").write(s2)
    ins = f"python3 {SK}/tools/insert_figs.py"; pre = f"lec05-{kp}"
    print(sh(f"{ins} {d}/{A.caps} {zh_md} {en_md} --prefix {pre}", check=False).strip().splitlines()[-1])   # zh done; en may stop at a heading it cannot find
    caps = open(f"{d}/{A.caps}").read()
    for pair in [p for p in A.en_before.split(";") if p]:
        z, e = pair.split("="); caps = re.sub(r'("before"\s*:\s*")' + re.escape(z) + '"', lambda m: m.group(1) + e + '"', caps)
    open("/tmp/_caps_en.json", "w").write(caps)
    print(sh(f"{ins} /tmp/_caps_en.json {zh_md} {en_md} --prefix {pre}").strip().splitlines()[-1])
    for x in rm:
        if os.path.exists(f"{REPO}/assets/figures/{x}"): sh(f"git rm -q -f assets/figures/{x}")
    print(sh(f"python3 scripts/check_slides.py 2>&1 | grep -i '{stem}' || true", check=False))
    out = sh("python3 scripts/build_docs.py >/dev/null 2>&1; mkdocs build 2>&1 | grep -i 'error\\|warning:' || true", check=False); print("mkdocs:", out.strip() or "ok")
    if A.dry_run: print(sh("git status --short")); return
    ex = f"{SK}/examples"
    for f in (f"{kp}_nums.py", f"{kp}_pages.py", f"{kp}_text.py"): shutil.copy(f"{d}/{f}", f"{ex}/{f}")
    for g in glob.glob(f"{d}/{kp}_data.py"): shutil.copy(g, f"{ex}/{kp}_data.py")
    shutil.copy(f"{d}/{A.caps}", f"{ex}/{kp}_caps.json"); shutil.copy(f"{d}/{A.map}", f"{ex}/{kp}_map.json")
    s = open("CHANGELOG.md").read().replace("## 2026 秋季\n\n", f"## 2026 秋季\n\n- {T} — {A.changelog}\n", 1); open("CHANGELOG.md", "w").write(s)
    open("COURSE_STATE.md", "a").write(f"- {T} — {A.state}\n")
    lp = f"{JARVIS}/logs/2026-10-24-lec05-投影片重做.md"; s = open(lp).read().replace("- 時間為推算,可能差幾分鐘。", f"- {T} {A.log}\n- 時間為推算,可能差幾分鐘。", 1); open(lp, "w").write(s)
    sh(f"git add CHANGELOG.md COURSE_STATE.md assets/figures lectures slides .agents/skills/view-from-text-v3/examples")
    names = sh("git diff --cached --name-status HEAD"); bad = [l for l in names.splitlines() if l[0] not in "AM" and not (l[0] == "D" and os.path.basename(l.split()[-1]) in rm)]
    if bad: raise SystemExit("unexpected changes staged:\n" + "\n".join(bad))
    msg = f"{kp.upper()} 簡報以 v3 重做並發佈,講義換圖\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VtCC9aDbSiytjfhjv3FCFV"
    open("/tmp/_msg.txt", "w").write(msg); sh("git commit -q -F /tmp/_msg.txt")
    for i in range(4):
        r = subprocess.run("git push -q origin main", shell=True, cwd=REPO, capture_output=True, text=True)
        if r.returncode == 0: break
        sh("git pull --rebase origin main"); time.sleep(2 ** i)
    else: raise SystemExit("push failed")
    for R in OTHERS:
        for f in glob.glob(f"{ex}/{kp}_*"):
            if "__pycache__" not in f: shutil.copy(f, f"{R}/.agents/skills/view-from-text-v3/examples/")
        sh(f"git pull -q --rebase origin main; git add .agents/skills/view-from-text-v3/examples; git commit -q -m '同步 {kp.upper()} 範例檔\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VtCC9aDbSiytjfhjv3FCFV'; git push -q origin main", cwd=R)
    sh(f"git pull -q --rebase origin main; git add logs; git commit -q -m '記錄 {kp.upper()} 簡報重做發佈\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VtCC9aDbSiytjfhjv3FCFV'; git push -q origin main", cwd=JARVIS)
    print("PUBLISHED", kp, sh("git status -sb | head -1").strip())
os.chdir(REPO)
acquire()
try: main()
finally: shutil.rmtree(LOCK, ignore_errors=True)
