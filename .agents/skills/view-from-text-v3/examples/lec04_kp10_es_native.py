"""KP10 'curvature spectrum comparison & early-stopping recipe' page, rebuilt from a picture with image-to-jbd-page.
    python3 build_kp10_es.py <out_dir>      -> <out_dir>/nodes_zh.json, nodes_en.json   (then insert_page.py ... --after 3)"""
import math, os, sys
SKILL = os.environ.get("JBD_PAGE_SKILL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "image-to-jbd-page")
sys.path.insert(0, os.path.join(SKILL, "tools"))
from jbd_page import Page

INK, NAVY, GREYTXT, PANEL = "#222a33", "#2f3c52", "#4a5565", "#ffffff"
GREEN, DKGREEN, RED, DKRED = "#34c26a", "#2f8f57", "#e0483c", "#d9453a"
# numbers: retention 1-(1-eta*lam)^t vs lam/(lam+alpha); eta=0.1, t=10, alpha=1
ROWS = []
for lam in (0.1, 0.25, 0.5, 1.0, 4.0):
    es, wd = 1 - (1 - 0.1 * lam) ** 10, lam / (lam + 1.0)
    ROWS.append((f"{lam:g}" if lam != 1.0 and lam != 4.0 else f"{lam:.1f}", f"{es:.4f}", f"{wd:.4f}", f"{abs(round(es, 4) - round(wd, 4)):.4f}"))   # the picture subtracts the rounded values (exact 0.5 -> 0.0679)
assert [r[1:] for r in ROWS] == [("0.0956", "0.0909", "0.0047"), ("0.2237", "0.2000", "0.0237"), ("0.4013", "0.3333", "0.0680"), ("0.6513", "0.5000", "0.1513"), ("0.9940", "0.8000", "0.1940")], ROWS

T = {"zh": dict(t1="KP10 Section 4: Curvature Spectrum Comparison & Practical Recipe", t2="對應精確度光譜與早停實務五部曲",
                lt="各曲率保留比例對比表 (η=0.1, t=10, α=1)", h=["曲率 $\\lambda$", "早停", "權重衰減", "相差", "精準度\n評級"],
                h2=["", "(Early Stopping)", "(Weight Decay)", "(Difference)", ""], star="★ 極準!", tag="Analysis",
                an=["平坦方向 (小 $\\lambda$) 近乎精準吻合，", "曲率越大差距漸增；但整體趨勢高度一致！"],
                rt="早停法實務 SOP 五部曲 (Engineering Checklist)",
                steps=[["劃分獨立 Validation Set (切分不可逆)。"], ["記錄", "定期評估驗證損失，\n錄歷史最佳 Checkpoint。"], ["設定合理 Patience\n(例如 10~20 次評估無改善)。"],
                       ["觸發早停時立即回退 (Rollback)\n至最佳權重。"], ["最終必須用完全獨立的 Test Set\n做無偏驗收 (不可拿驗證集自吹)！"]],
                kt="實務準則 (Key Takeaway)", kx="早停法是免費的高效正則化工具，搭配 Patience 與 Checkpoint 即可輕鬆防範過擬合！"),
     "en": dict(t1="KP10 Section 4: Curvature Spectrum Comparison & Practical Recipe", t2="The precision spectrum and the five-step early-stopping recipe",
                lt="Retained fraction by curvature (η=0.1, t=10, α=1)", h=["Curv.\n$\\lambda$", "Early stopping", "Weight decay", "Difference", "Accuracy\ngrade"],
                h2=["", "", "", "", ""], star="★ Exact!", tag="Analysis",
                an=["Flat directions (small $\\lambda$): nearly exact match;", "gap grows with curvature, same overall trend."],
                rt="Early-stopping SOP: five steps",
                steps=[["Split off an independent Validation Set (the split is irreversible)."], ["Record the validation loss periodically and keep the best Checkpoint so far."],
                       ["Set a sensible Patience (e.g. 10-20 evaluations without improvement)."], ["On triggering early stopping, roll back (Rollback) to the best weights."],
                       ["Finally, an independent Test Set for an unbiased check (never the validation set)!"]],
                kt="Key Takeaway", kx="Early stopping is a free, effective regularizer: Patience + Checkpoint guard against overfitting!")}


def build(lang):
    t = T[lang]; P = Page(src_w=1024, src_h=572, prefix="k10es")
    P.rect(0, 0, 1024, 572, fill="#f3f5f9", z=0)
    P.label(30, 22, 964, 38, t["t1"], size=26, color=INK)
    P.label(30, 62, 964, 34, t["t2"], size=26, color=INK)
    shadow = dict(shadowColor="rgba(40,50,70,0.18)", shadowOpacity=1, shadowBlur=6, shadowOffsetX=0, shadowOffsetY=2)
    # ---- left panel: table
    P.rect(28, 114, 502, 369, fill=PANEL, stroke="#e1e5ec", sw=1, radius=12, **shadow)
    P.label(36, 124, 486, 28, t["lt"], size=18, color=INK)
    xs = [36.7, 104.6, 228, 350, 442, 523]; ys = [162, 221, 256, 291, 327, 362, 397]
    for c in range(5):
        P.rect(xs[c], ys[0], xs[c + 1] - xs[c], ys[1] - ys[0], fill="#d5dbe3", stroke="#ffffff", sw=1, z=2)
        if t["h2"][c]:
            P.label(xs[c], ys[0] + 6, xs[c + 1] - xs[c], 28, t["h"][c], size=17, color=INK, z=4)
            P.label(xs[c], ys[0] + 32, xs[c + 1] - xs[c], 22, t["h2"][c], size=13, color=INK, bold=False, z=4)
        elif "\n" in t["h"][c]:
            P.text(xs[c], ys[0] + 8, xs[c + 1] - xs[c], 46, t["h"][c], size=17, color=INK, bold=True, align="center", line=1.2, z=4)
        else:
            P.label(xs[c], ys[0], xs[c + 1] - xs[c], ys[1] - ys[0], t["h"][c], size=17, color=INK, z=4)
    for r in range(5):
        for c in range(5):
            fill = GREEN if r == 0 else ("#e3e6eb" if c == 0 else "#f8f9fa")
            P.rect(xs[c], ys[r + 1], xs[c + 1] - xs[c], ys[r + 2] - ys[r + 1], fill=fill, stroke="#d9dde4", sw=1, z=2)
            txt = ROWS[r][c] if c < 4 else (t["star"] if r == 0 else "")
            if txt: P.label(xs[c], ys[r + 1], xs[c + 1] - xs[c], ys[r + 2] - ys[r + 1], txt, size=17, color=INK, bold=(c == 4), z=4)
    # analysis box
    P.rect(37.5, 410, 485, 66, fill="#f7f8fa", stroke=NAVY, sw=1.5, radius=8)
    P.rect(50, 401, 72, 19, fill="#34425e", radius=4, z=3); P.label(50, 401, 72, 19, t["tag"], size=12, color="#ffffff", bold=False, z=5)
    P.text(50, 424, 380, 24, t["an"][0], size=17, color=INK, bold=True); P.text(50, 450, 385, 24, t["an"][1], size=17, color=INK, bold=True)
    P.line([(436, 471), (436, 417)], NAVY, 1.5, arrow="end", head=6); P.line([(436, 471), (512, 471)], NAVY, 1.5, arrow="end", head=6)
    P.line([(441, 466), (462, 448), (486, 433), (508, 421)], "#1b2a2a", 1.8, smooth=True)
    P.line([(441, 467), (462, 445), (486, 430), (508, 420)], "#2f9d62", 1.8, smooth=True)
    P.line([(470, 458), (508, 456)], "#2f9d62", 1, dash=True)
    # ---- right panel: checklist
    P.rect(545, 114, 451, 369, fill=PANEL, stroke="#e1e5ec", sw=1, radius=12, **shadow)
    P.label(552, 124, 437, 28, t["rt"], size=17, color=INK)
    cy = [189, 253, 317, 380, 445]
    for i in range(4): P.line([(571, cy[i] + 15), (571, cy[i + 1] - 15)], NAVY, 3)
    top = [163, 225, 288, 352.5, 417]; hh = [52, 53, 54.5, 53.5, 56]
    for i in range(5):
        red = i == 4; col = DKRED if red else NAVY
        P.line([(584, cy[i]), (596, cy[i])], col, 2)
        P.rect(596, top[i], 390, hh[i], fill="#fbdcd8" if red else "#e8ebf2", stroke=col, sw=1.5, radius=9)
        P.ellipse(571, cy[i], 15, 15, fill=RED if red else NAVY); P.label(556, cy[i] - 14, 30, 28, str(i + 1), size=17, color="#ffffff", z=5)
        s = t["steps"][i]
        if lang == "zh" and i == 1:
            P.text(606, 243, 40, 22, s[0], size=17, color=INK, bold=True); P.text(645, 230, 300, 50, s[1], size=17, color=INK, bold=True)
        elif lang == "zh" and i in (2, 3, 4):
            P.text(645, cy[i] - 25, 300, 52, s[0], size=17, color=INK, bold=True)
        elif lang == "zh":
            P.text(606, cy[i] - 12, 340, 24, s[0], size=17, color=INK, bold=True)
        else:
            P.text(645 if i in (2, 3) else 606, top[i] + 6, 295 if i in (2, 3) else 335, hh[i] - 8, s[0], size=15, color=INK, bold=True)
        ic = 964
        if i == 0:      # database
            for k in (0, 1, 2): P.rect(ic - 12, cy[i] - 13 + k * 9, 24, 8, fill="transparent", stroke=NAVY, sw=2, radius=4, z=3)
        elif i == 1:    # line chart
            P.line([(ic - 13, cy[i] - 13), (ic - 13, cy[i] + 13), (ic + 13, cy[i] + 13)], NAVY, 2)
            P.line([(ic - 10, cy[i] + 5), (ic - 3, cy[i] - 3), (ic + 3, cy[i] + 3), (ic + 12, cy[i] - 8)], NAVY, 2)
        elif i == 2:    # clock + list icon
            P.ellipse(ic, cy[i], 14, 14, stroke=NAVY, sw=2.2); P.line([(ic, cy[i] - 8), (ic, cy[i]), (ic + 6, cy[i] + 4)], NAVY, 2)
            for k in (0, 1, 2): P.ellipse(610, cy[i] - 10 + k * 10, 2, 2, fill=NAVY); P.line([(618, cy[i] - 10 + k * 10), (632, cy[i] - 10 + k * 10)], NAVY, 1.5)
        elif i == 3:    # undo arrow + small icon
            P.line([(ic + 12, cy[i] + 10), (ic + 12, cy[i] - 2), (ic - 8, cy[i] - 2)], NAVY, 2.5, arrow="end", head=9)
            P.rect(607, cy[i] - 4, 14, 10, fill=NAVY, radius=5, z=3); P.line([(622, cy[i] + 6), (630, cy[i] + 2), (630, cy[i] - 4)], NAVY, 2)
        else:           # shield
            P.poly([(ic - 12, cy[i] - 14), (ic, cy[i] - 18), (ic + 12, cy[i] - 14), (ic + 12, cy[i] + 2), (ic, cy[i] + 15), (ic - 12, cy[i] + 2)], fill=RED, stroke=DKRED, sw=2)
    # ---- takeaway banner
    P.rect(27.5, 502, 969, 58, fill="#e8f7ef", stroke=DKGREEN, sw=1.8, radius=10)
    P.rect(40.6, 489, 211.5 if lang == "zh" else 150, 27, fill="#1f6b43", radius=6, z=3); P.label(40.6, 489, 211.5 if lang == "zh" else 150, 27, t["kt"], size=16, color="#ffffff", z=5)
    P.label(36, 519, 952, 38, t["kx"], size=24 if lang == "zh" else 19, color=INK)
    P.ellipse(975, 502, 8, 9, stroke=DKGREEN, sw=1.6); P.line([(971, 512), (979, 512)], DKGREEN, 1.6); P.line([(972, 516), (978, 516)], DKGREEN, 1.6)
    return P


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    for lang in ("zh", "en"): build(lang).dump(os.path.join(out, f"nodes_{lang}.json"))
