"""Insert figure captions + images into a lecture (zh and en) from a JSON file.
    python3 insert_figs.py <captions.json> <lecture_zh.md> <lecture_en.md> [--prefix lec06-kp04]
captions.json: [{"id": "cost", "before": "延伸", "zh": ["圖說文字（含〔功能：N〕）", "alt 文字"], "en": ["Figure text [Function: N]", "alt text"]}, ...]
  "before": the start of the heading to insert before (matched against ### headings; use "END" for the end, before the terms section).
An existing block `圖說 `<prefix>-<id>`` / `Figure `<prefix>-<id>`` is replaced. The English file uses the -en PNG. Paths are ../assets (zh) and ../../assets (en)."""
import re, sys, json
cap_file, zh_path, en_path = sys.argv[1:4]; prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "lec06-kp"
caps = json.load(open(cap_file))
for path, lang in ((zh_path, "zh"), (en_path, "en")):
    s = open(path).read(); en = lang == "en"
    for c in caps:
        txt, alt = c[lang]; fid = f"{prefix}-{c['id']}"
        blk = (f"Figure `{fid}`: {txt}\n\n![{alt}](../../assets/figures/{fid}-en.png)\n\n" if en else f"圖說 `{fid}`：{txt}\n\n![{alt}](../assets/figures/{fid}.png)\n\n")
        m = re.search(r"(圖說|Figure) `%s`.*?%s(-en)?\.png\)\n\n" % (re.escape(fid), re.escape(c["id"])), s, flags=re.S)
        if m: s = s[:m.start()] + blk + s[m.end():]; continue
        if c["before"] == "END":
            j = s.index("## 術語與來源" if not en else "## Terms and source"); s = s[:j].rstrip("\n") + "\n\n" + blk.rstrip("\n") + "\n\n" + s[j:]; continue
        heads = [h for h in re.findall(r"^### .*$", s, flags=re.M) if c["before"] in h]; j = s.index(heads[0])
        q = max(s.rfind("接下來的問題", 0, j), s.rfind("The next question", 0, j)); pos = q if q != -1 and j - q < 400 else j
        s = s[:pos] + blk + s[pos:]
    if en: s = re.sub(r"(assets/figures/%s-[a-z0-9]+)\.png" % re.escape(prefix), r"\1-en.png", s); s = s.replace("-en-en.png", "-en.png")
    open(path, "w").write(s)
print("inserted", len(caps))
