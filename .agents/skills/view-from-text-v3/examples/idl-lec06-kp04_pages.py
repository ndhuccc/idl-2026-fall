"""KP04 (lec06) pages designed in HTML (view-from-text-v3, route A). pageN(lang) returns the body HTML of a 1160x460 figure.
    python3 kp04_pages.py OUTDIR [p1 p2 ...]
Every number comes from kp04_nums.py (N)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, HERE)
from kp04_kit import *
import kp04_nums as N
N.check()
X = [int(v) for v in N.x]; Wk = [int(v) for v in N.w]
WROLE = {4: "orange", 5: "blue"}          # weight 4 orange, weight 5 blue (same as the lecture figures)
def vec(a): return "[" + ",".join(str(v) for v in a) + "]"
def rows(M): return "、".join(vec(r) for r in M)
def rows_en(M): return ", ".join(vec(r) for r in M[:-1]) + " and " + vec(M[-1]) if len(M) > 1 else vec(M[0])

# ---------------------------------------------------------------- 1: weighted sum
def page1(lang):
    t = tr(lang); o = card(0, 0, 690, 400, "grey", .08) + head(0, 0, 690, "grey", t("把核蓋在訊號上：對齊的格子相乘再相加", "Lay the kernel on the signal: multiply aligned cells, then add"), 36, 18)
    cx = [200 + 76 * i for i in range(3)]
    o += mt(20, 64, 170, t("訊號 $x$", "signal $x$"), 18, MUT)
    for i, v in enumerate(X): o += cell(cx[i], 56, v)
    o += mt(cx[0], 112, 3 * 76, t("位置 $0$、$1$、$2$", "positions $0$, $1$, $2$"), 16, MUT, ha="center", h=26)
    for r, (lab_zh, lab_en) in enumerate((("核在第一個位置", "first position"), ("核在第二個位置", "second position"))):
        y = 160 + r * 74
        o += mt(20, y + 12, 175, t(lab_zh, lab_en), 16, MUT)
        o += cell(cx[r], y, Wk[0], WROLE[Wk[0]]) + cell(cx[r + 1], y, Wk[1], WROLE[Wk[1]])
    o += mchip(20, 318, 650, "green", t(f"核要完整落在訊號內：只有 $3-2+1={N.valid_n}$ 個位置", f"The kernel must lie fully inside: only $3-2+1={N.valid_n}$ positions"), False, 36, 18)
    o += mt(440, 56, 240, t("核（也叫濾波器）：\n一組固定的權重，\n這裡是 $4$ 和 $5$。", "Kernel (or filter):\na fixed set of weights,\nhere $4$ and $5$."), 16, TXT, name="term:" + t("核", "kernel"), lh=26)
    o += mt(440, 150, 240, t("有效輸出：核完整落在\n訊號內的位置算出的數。\n長 $n$ 的訊號配長 $k$ 的核，\n共有 $n-k+1$ 個。", "Valid output: numbers from\npositions where the kernel\nlies fully inside the signal;\nthere are $n-k+1$."), 16, TXT, name="term:" + t("有效輸出", "valid output"), lh=26)
    o += card(706, 0, 454, 400, "yellow", .08) + head(706, 0, 454, "yellow", t("第一個位置：照原樣蓋，還是先翻面？", "First position: as it is, or flipped first?"), 36, 18)
    for r, (lab, kern, tex, val) in enumerate(((t("照原樣", "as it is"), Wk, r"2\cdot4+1\cdot5", N.p1_as_is), (t("先翻面", "flipped first"), Wk[::-1], r"2\cdot5+1\cdot4", N.p1_flip))):
        y = 58 + r * 150
        o += box(722, y, 422, 132, rgba("grey", .10), rgba("grey", .5), 2, 12)
        o += mt(738, y + 10, 150, lab, 18, TXT, True) + cell(900, y + 8, kern[0], WROLE[kern[0]], 54, 44, 20) + cell(962, y + 8, kern[1], WROLE[kern[1]], 54, 44, 20)
        o += mt(738, y + 70, 390, f"${tex}={val}$", 24, TXT, ha="center", h=44)
    o += chip(806, 356, 290, "red", t("兩個數字不同", "the two numbers differ"), True, 34, 18)
    o += take(t, "每個輸出是附近幾格的加權和；核的方向會改變算出的數字。", "Each output is a weighted sum of nearby cells; the direction of the kernel changes the number.")
    return o

# ---------------------------------------------------------------- 2: the subscript t - tau
def page2(lang):
    t = tr(lang); o = card(0, 0, 760, 400, "grey", .08)
    o += k(0, 10, 760, r"(x*w)[t]=\sum_{\tau=-\infty}^{\infty}x[\tau]\,w[t-\tau]", 28, 70)
    o += mchip(70, 92, 300, "violet", t("$-\\tau$：把核沿軸鏡射", "$-\\tau$: mirror the kernel"), False, 34, 18)
    o += mchip(390, 92, 300, "teal", t("$+t$：再挪到位置 $t$", "$+t$: then move it to $t$"), False, 34, 18)
    xs = {tau: 300 + 90 * (tau + 1) for tau in (-1, 0, 1, 2)}
    o += mt(20, 146, 200, t("位置 $\\tau$", "position $\\tau$"), 16, MUT)
    for tau, xx in xs.items(): o += mt(xx, 146, 64, f"${tau}$", 18, MUT, ha="center", h=26)
    rowsd = ((r"w[\tau]", N.pos_w, t("原本的核", "the kernel as given")), (r"w[-\tau]", N.pos_wm, t("鏡射：$5$ 跑到 $\\tau=-1$", "mirrored: $5$ moves to $\\tau=-1$")),
             (r"w[1-\tau]", N.pos_w1, t("挪到 $t=1$：$[5,4]$", "moved to $t=1$: $[5,4]$")))
    for r, (lab, pos, note) in enumerate(rowsd):
        y = 180 + r * 62
        o += k(10, y + 4, 200, lab, 22, 44)
        for tau, xx in xs.items():
            v = pos.get(tau)
            o += cell(xx, y, v if v is not None else "", WROLE.get(v, "grey"), 64, 52, 22, dim=v is None)
    o += mt(20, 368, 720, t("訊號與核以外的格子都當成 $0$（灰格）。", "Cells outside the signal and the kernel count as $0$ (grey cells)."), 16, MUT)
    o += card(776, 0, 384, 400, "violet", .08) + head(776, 0, 384, "violet", t("三條軸上的核", "The kernel on three axes"), 36, 18)
    for r, (lab, pos, note) in enumerate(rowsd):
        o += mt(792, 52 + r * 34, 360, f"${lab}$：" + note if lang == "zh" else f"${lab}$: " + note, 16, TXT)
    o += mt(792, 162, 360, t("求和指標 $\\tau$：掃過輸入各格的\n位置，和輸出位置 $t$ 無關。", "Summation index $\\tau$: sweeps the\ninput cells; it is not the output $t$."), 16, TXT, name="term:" + t("求和指標", "summation index"), lh=26)
    o += mt(792, 226, 360, t("卷積 $*$：把核先翻面，再滑到\n每個位置 $t$ 做加權和。", "Convolution $*$: flip the kernel,\nthen slide it to each $t$ and add."), 16, TXT, name="term:" + t("卷積", "convolution"), lh=26)
    o += mt(792, 290, 360, t("翻轉核：把核的順序倒過來，\n$[4,5]$ 變成 $[5,4]$。", "Flipping the kernel: reverse its\norder, so $[4,5]$ becomes $[5,4]$."), 16, TXT, name="term:" + t("翻轉核", "flipping the kernel"), lh=26)
    o += take(t, "下標 $t-\\tau$ 是相減：先把核翻面，再滑到位置 $t$。", "The subscript $t-\\tau$ subtracts: flip the kernel first, then slide it to $t$.")
    return o

# ---------------------------------------------------------------- 3: full convolution term by term
def page3(lang):
    t = tr(lang); o = card(0, 0, 720, 400, "grey", .08)
    xs = {tau: 150 + 70 * (tau + 1) for tau in range(-1, 4)}
    o += mt(16, 14, 130, t("訊號 $x$", "signal $x$"), 18, MUT)
    for tau, xx in xs.items(): o += cell(xx, 8, X[tau] if 0 <= tau < 3 else 0, "grey", 60, 44, 20, dim=not 0 <= tau < 3)
    for tau, xx in xs.items(): o += mt(xx - 20, 54, 100, f"$\\tau={tau}$" if tau == -1 else f"${tau}$", 16, MUT, ha="center", h=24)
    for t_ in range(4):
        y = 92 + t_ * 74; valid = t_ in (1, 2)
        o += box(8, y - 6, 704, 64, rgba("green" if valid else "grey", .14 if valid else .06), rgba("green", .6) if valid else None, 2, 10)
        o += mt(16, y + 8, 110, f"$t={t_}$", 20, TXT, True)
        for tau, wi, val in ((t_ - 1, 1, Wk[1]), (t_, 0, Wk[0])):       # flipped kernel [5,4]: w[1] at tau=t-1, w[0] at tau=t
            inside = 0 <= tau < 3
            o += cell(xs[tau] if tau in xs else 0, y + 4, val, WROLE[val], 60, 44, 20, dim=not inside)
        terms = N.full_terms[t_]
        tex = "+".join(f"{xv}\\cdot{wv}" for _, xv, _, wv in terms) + f"={N.full[t_]}"
        o += mt(510, y + 10, 200, f"${tex}$", 20, TXT, ha="left", h=32)
    o += card(736, 0, 424, 400, "green", .08) + head(736, 0, 348, "green", t("完整卷積與有效輸出", "Full and valid output"), 36, 18) + added(1092, 5, lang, 60 if lang == "zh" else 62)
    o += mt(752, 50, 392, t("完整卷積：核和訊號至少重疊一格的\n所有位置，共 $n+k-1=4$ 個數。", "Full convolution: every position where\nkernel and signal overlap in at least\none cell; $n+k-1=4$ numbers."), 16, TXT, name="term:" + t("完整卷積", "full convolution"), lh=26, h=78)
    o += mchip(752, 136, 392, "grey", t(f"$t=0,3$：一格懸在訊號外（淡色）", f"$t=0,3$: one cell hangs outside (faint)"), False, 34, 16)
    o += mchip(752, 180, 392, "green", t(f"$t=1,2$：有效輸出 ${vec(N.valid_conv)}$", f"$t=1,2$: valid output ${vec(N.valid_conv)}$"), False, 34, 18)
    o += mt(752, 232, 392, t("看 $t=1$：$x[0]$ 乘的是 $w[1]=5$，\n$x[1]$ 乘的是 $w[0]=4$。", "Look at $t=1$: $x[0]$ is multiplied by\n$w[1]=5$, and $x[1]$ by $w[0]=4$."), 16, TXT, lh=26)
    o += mt(752, 292, 392, t("等於把核翻成 $[5,4]$ 之後才對齊。", "That is the kernel flipped to $[5,4]$."), 16, TXT, True)
    o += mt(752, 330, 392, t(f"有效輸出 $n-k+1={N.valid_n}$ 個數。", f"Valid output: $n-k+1={N.valid_n}$ numbers."), 16, MUT)
    o += take(t, f"完整卷積是 ${vec(N.full)}$；核整個在訊號內的只有 $t=1,2$，有效輸出 ${vec(N.valid_conv)}$。", f"The full convolution is ${vec(N.full)}$; only $t=1,2$ have the kernel fully inside: ${vec(N.valid_conv)}$.")
    return o

# ---------------------------------------------------------------- 4: cross-correlation
def page4(lang):
    t = tr(lang); o = card(0, 0, 720, 400, "grey", .08)
    o += k(0, 8, 720, r"(x\star w)[t]=\sum_{m=0}^{k-1}x[t+m]\,w[m]", 28, 66)
    cx = [210 + 76 * i for i in range(3)]
    o += mt(20, 98, 170, t("訊號 $x$", "signal $x$"), 18, MUT)
    for i, v in enumerate(X): o += cell(cx[i], 90, v)
    tex = (r"2\cdot4+1\cdot5", r"1\cdot4+3\cdot5")
    for t_ in range(2):
        y = 176 + t_ * 96
        o += mt(20, y + 12, 170, f"$t={t_}$", 20, TXT, True)
        o += cell(cx[t_], y, Wk[0], WROLE[Wk[0]]) + cell(cx[t_ + 1], y, Wk[1], WROLE[Wk[1]])
        o += mt(cx[t_] - 10, y + 56, 80, "$w[0]$", 16, MUT, ha="center", h=26) + mt(cx[t_ + 1] - 10, y + 56, 80, "$w[1]$", 16, MUT, ha="center", h=26)
        o += mt(450, y + 10, 260, f"${tex[t_]}={N.corr[t_]}$", 22, TXT, h=34)
    o += card(736, 0, 424, 400, "red", .08) + head(736, 0, 424, "red", t("不翻核，指標相加", "No flip: the indices add"), 36, 18)
    o += mt(752, 50, 392, t("互相關 $\\star$：不翻核，直接把核蓋上去；\n$w[0]$ 永遠對齊 $x[t]$（橘色格）。", "Cross-correlation $\\star$: lay the kernel down\nwithout flipping; $w[0]$ always lines up\nwith $x[t]$ (the orange cell)."), 16, TXT, name="term:" + t("互相關", "cross-correlation"), lh=26, h=78)
    o += mt(752, 136, 392, t("$t$ 從 $0$ 數到 $n-k$，正好是有效輸出。", "$t$ runs from $0$ to $n-k$: the valid output."), 16, TXT)
    o += mchip(752, 180, 190, "green", t(f"卷積 ${vec(N.valid_conv)}$", f"convolution ${vec(N.valid_conv)}$"), False, 36, 17)
    o += mchip(954, 180, 190, "red", t(f"互相關 ${vec(N.corr)}$", f"correlation ${vec(N.corr)}$"), False, 36, 17)
    o += chip(752, 226, 392, "red", t("同一組 x、w，數字不同", "same x and w, different numbers"), True, 32, 17)
    o += box(752, 276, 392, 108, rgba("yellow", .10), rgba("yellow", .5), 2, 10)
    o += mt(764, 284, 370, t("常見錯誤：把 $w[t-\\tau]$ 讀成 $w[\\tau-t]$，\n以為核照原樣滑過去；\n這樣算的其實是互相關。", "Common error: reading $w[t-\\tau]$ as $w[\\tau-t]$\nand sliding the kernel as it is;\nthat computes cross-correlation."), 16, TXT, lh=30, h=92)
    o += take(t, f"互相關不翻核：同一組 $x$、$w$ 得到 ${vec(N.corr)}$，和卷積的 ${vec(N.valid_conv)}$ 不同。", f"Cross-correlation does not flip: the same $x$, $w$ give ${vec(N.corr)}$, not the convolution's ${vec(N.valid_conv)}$.")
    return o

# ---------------------------------------------------------------- 5: flipped kernel = convolution (epiphany)
def page5(lang):
    t = tr(lang); o = ""
    cols = ((t("卷積 $x*w$（核翻成 $[5,4]$）", "convolution $x*w$ (flip to $[5,4]$)"), Wk[::-1], (1, 2), N.valid_conv, "green", r"2\cdot5+1\cdot4", r"1\cdot5+3\cdot4"),
            (t("互相關 $x\\star w$（$w=[4,5]$）", "correlation $x\\star w$ ($w=[4,5]$)"), Wk, (0, 1), N.corr, "red", r"2\cdot4+1\cdot5", r"1\cdot4+3\cdot5"),
            (t("互相關 $x\\star\\tilde w$（$\\tilde w=[5,4]$）", "correlation $x\\star\\tilde w$ ($\\tilde w=[5,4]$)"), Wk[::-1], (0, 1), N.corr_flip, "green", r"2\cdot5+1\cdot4", r"1\cdot5+3\cdot4"))
    for c, (hd, kern, ts, out, role, e0, e1) in enumerate(cols):
        x0 = c * 393; cw = 374
        o += card(x0, 0, cw, 318, role, .08) + box(x0, 0, cw, 40, rgba(role, .30), None, 0, 12) + mt(x0, 3, cw, hd, 17, TXT, True, "center", lh=34, h=34)
        cx = [x0 + 92 + 66 * i for i in range(3)]
        for i, v in enumerate(X): o += cell(cx[i], 54, v, "grey", 56, 44, 20)
        for r in range(2):
            y = 112 + r * 66
            o += mt(x0 + 10, y + 8, 78, f"$t={ts[r]}$", 18, TXT, True)
            o += cell(cx[r], y, kern[0], WROLE[kern[0]], 56, 44, 20) + cell(cx[r + 1], y, kern[1], WROLE[kern[1]], 56, 44, 20)
        o += mt(x0 + 10, 226, cw - 20, f"${e0}={out[0]}$", 17, MUT, ha="center", h=26) + mt(x0 + 10, 252, cw - 20, f"${e1}={out[1]}$", 17, MUT, ha="center", h=26)
        o += mchip(x0 + 77, 282, 220, role, t(f"輸出 ${vec(out)}$", f"output ${vec(out)}$"), False, 30, 18)
    o += box(0, 330, W, 70, rgba("green", .10), rgba("green", .55), 2, 12)
    o += k(10, 336, 470, r"(x\star\tilde w)[t]=(x*w)[t+k-1]", 26, 56)
    o += mt(490, 336, 660, t("$\\tilde w[m]=w[k-1-m]$：把核左右翻；左欄與右欄的輸出相同。", "$\\tilde w[m]=w[k-1-m]$ flips the kernel; the left and right outputs match."), 16, TXT, lh=28, h=28)
    o += mt(490, 366, 660, t("$k=2$：卷積的 $t=1,2$ 對應互相關的 $t=0,1$，只是編號平移。", "$k=2$: $t=1,2$ of convolution are $t=0,1$ of correlation, a shift of numbering."), 16, MUT, lh=28, h=28)
    o += take(t, "兩者只差核的方向：把核翻面再做互相關，就是卷積。", "They differ only in the direction of the kernel: correlate with the flipped kernel and you get convolution.")
    return o

# ---------------------------------------------------------------- 6: 2-D: flip twice = rotate 180
def grid(x0, y0, M, cs=56, gap=6, role="orange", shade=True, bold_vals=(1, 9), size=22, vmin=None, vmax=None, roles=None):
    o = ""; flat = [v for r in M for v in r]; vmin = min(flat) if vmin is None else vmin; vmax = max(flat) if vmax is None else vmax
    for i, r in enumerate(M):
        for j, v in enumerate(r):
            rl = roles(v) if roles else role
            a = (.12 + .70 * (v - vmin) / max(vmax - vmin, 1)) if shade else .30
            o += cell(x0 + j * (cs + gap), y0 + i * (cs + gap), v, rl, cs, cs, size, a=a, bold=v in bold_vals)
    return o

def page6(lang):
    t = tr(lang); o = card(0, 0, W, 400, "grey", .06)
    o += k(0, 6, 700, r"(I*K)[i,j]=\sum_{p}\sum_{q}I[p,q]\,K[i-p,\,j-q]", 26, 60)
    o += mt(700, 10, 450, t("$p,q$：掃過影像各格的求和指標", "$p,q$: summation indices over the image"), 16, TXT)
    o += mt(700, 38, 450, t("$(i,j)$：輸出的位置", "$(i,j)$: the output position"), 16, TXT)
    o += mchip(250, 74, 660, "violet", t("兩個下標都相減：列方向翻一次，行方向也翻一次", "Both subscripts subtract: flip once along rows and once along columns"), False, 34, 17)
    gx = (110, 480, 850); gy = 124
    for g, M in zip(gx, (N.K9.tolist(), N.K9_lr, N.K9_rot)): o += grid(g, gy, M)
    o += svg(0, 0, W, 400, arrow(310, 215, 460, 215, MUT, 3) + arrow(680, 215, 830, 215, MUT, 3))
    o += mt(310, 178, 150, t("左右翻", "left-right"), 17, TXT, True, "center", h=28) + mt(680, 178, 150, t("上下翻", "up-down"), 17, TXT, True, "center", h=28)
    o += k(110, 308, 180, "K", 22, 32) + mt(450, 308, 240, t("左右翻之後", "after left-right"), 17, MUT, ha="center", h=30) + mt(790, 312, 300, t("$\\tilde K$：轉了 $180^\\circ$", "$\\tilde K$: rotated by $180^\\circ$"), 18, TXT, True, "center", h=30)
    o += k(60, 358, 520, r"\tilde K[m,n]=K[k-1-m,\,k-1-n]", 24, 40)
    o += mt(600, 362, 550, t("$m,n$ 掃過核的各格；$k$ 是核的邊長，這裡 $k=3$。", "$m,n$ sweep the kernel cells; $k$ is its side, here $k=3$."), 16, MUT, h=34)
    o += take(t, "左右翻再上下翻，合起來就是把核轉 $180^\\circ$：$1$ 從左上跑到右下。", "Flip left-right, then up-down: together a $180^\\circ$ rotation, and $1$ moves from top left to bottom right.")
    return o

# ---------------------------------------------------------------- 7: 2-D cross-correlation worked example
def page7(lang):
    t = tr(lang); o = card(0, 0, 760, 400, "grey", .06)
    o += k(0, 6, 760, r"s[i,j]=(I\star K)[i,j]=\sum_{m=0}^{k-1}\sum_{n=0}^{k-1}I[i+m,\,j+n]\,K[m,n]", 22, 60)
    I = N.I.tolist(); Kk = N.K.tolist(); S = N.s
    gx, gy, cs, gp = 40, 84, 52, 6
    for i in range(3):
        for j in range(3):
            inwin = i in (0, 1) and j in (1, 2)
            o += cell(gx + j * (cs + gp), gy + i * (cs + gp), I[i][j], "yellow" if inwin else "grey", cs, cs, 22, a=.30 if inwin else .18)
    o += box(gx + (cs + gp) - 5, gy - 5, 2 * cs + gp + 10, 2 * cs + gp + 10, None, YE, 3, 10)
    o += k(gx, gy + 3 * (cs + gp) + 2, 3 * cs + 2 * gp, "I", 20, 34)
    o += k(252, 128, 50, r"\star", 30, 50)
    kx = 308
    for i in range(2):
        for j in range(2): o += cell(kx + j * (cs + gp), gy + 31 + i * (cs + gp), Kk[i][j], "teal", cs, cs, 22, a=.30 if Kk[i][j] else .12)
    o += k(kx, gy + 31 + 2 * (cs + gp) + 2, 2 * cs + gp, "K", 20, 34)
    o += k(444, 128, 40, "=", 30, 50)
    sx = 494
    for i in range(2):
        for j in range(2): o += cell(sx + j * (cs + gp), gy + 31 + i * (cs + gp), S[i][j], "green", cs, cs, 22, a=.30)
    o += box(sx + (cs + gp) - 5, gy + 26, cs + 10, cs + 10, None, YE, 3, 10)
    o += k(sx, gy + 31 + 2 * (cs + gp) + 2, 2 * cs + gp, "s", 20, 34)
    o += mt(40, 304, 720, "$s[0,1]=I[0,1]K[0,0]+I[0,2]K[0,1]+I[1,1]K[1,0]+I[1,2]K[1,1]$", 20, TXT, h=36)
    o += mt(102, 344, 600, "$=2\\cdot1+0\\cdot0+1\\cdot0+2\\cdot1=4$", 20, TXT, h=36)
    o += card(776, 0, 384, 400, "green", .08) + mhead(776, 0, 384, "green", t("核的左上角對齊 $(i,j)$", "Kernel's top-left corner at $(i,j)$"))
    o += mt(792, 50, 352, t("影像 $I$ 三列：$[1,2,0]$、$[1,1,2]$、$[0,1,1]$。", "Image $I$ rows: $[1,2,0]$, $[1,1,2]$, $[0,1,1]$."), 16, TXT)
    o += mt(792, 82, 352, t("核 $K$：$2\\times2$ 單位矩陣，\n對角線是 $1$，其餘是 $0$。", "Kernel $K$: the $2\\times2$ identity,\n$1$ on the diagonal, $0$ elsewhere."), 16, TXT, lh=26)
    o += mt(792, 142, 352, t("特徵圖 $s$：二維的輸出，\n$s[i,j]$ 是位置 $(i,j)$ 的值。", "Feature map $s$: the 2-D output;\n$s[i,j]$ is its value at $(i,j)$."), 16, TXT, name="term:" + t("特徵圖", "feature map"), lh=26)
    o += mt(792, 202, 352, t("黃框：$s[0,1]$ 蓋住第 $0$、$1$ 列\n與第 $1$、$2$ 行。", "Yellow frames: $s[0,1]$ covers rows\n$0$, $1$ and columns $1$, $2$."), 16, TXT, lh=26)
    o += box(792, 262, 352, 62, rgba("green", .20), None, 0, 10) + mt(792, 264, 352, t("輸出大小\n$(3-2+1)\\times(3-2+1)=2\\times2$", "output size\n$(3-2+1)\\times(3-2+1)=2\\times2$"), 16, TXT, True, "center", lh=28, h=58)
    o += mt(792, 338, 352, t("四格都算完：輸出兩列是 $[2,4]$、$[2,2]$。", "All four: output rows $[2,4]$ and $[2,2]$."), 16, TXT, True)
    o += take(t, "核框住四格逐格乘加：$s[0,1]=4$，輸出兩列是 $[2,4]$ 與 $[2,2]$。", "The kernel frames four cells, multiply and add: $s[0,1]=4$; the output rows are $[2,4]$ and $[2,2]$.")
    return o

# ---------------------------------------------------------------- 8: symmetric kernel trap + 2-D numbering
def page8(lang):
    t = tr(lang); o = card(0, 0, 500, 400, "yellow", .07) + head(0, 0, 500, "yellow", t("陷阱：這個核轉了不變", "Trap: this kernel does not change"), 36, 18)
    Kk = N.K.tolist(); cs, gp = 56, 6
    for g, x0 in enumerate((60, 300)):
        for i in range(2):
            for j in range(2): o += cell(x0 + j * (cs + gp), 62 + i * (cs + gp), Kk[i][j], "teal", cs, cs, 22, a=.30 if Kk[i][j] else .12)
    o += k(60, 192, 118, "K", 20, 34) + k(300, 192, 118, r"\tilde K", 20, 34) + k(196, 92, 90, "=", 30, 50)
    o += mt(16, 236, 468, t("單位矩陣轉 $180^\\circ$ 還是自己：$\\tilde K=K$。", "The identity rotated by $180^\\circ$ is itself: $\\tilde K=K$."), 17, TXT, ha="center")
    o += mt(16, 270, 468, t("所以用卷積算，也得到 $[2,4]$ 與 $[2,2]$。", "So convolution also gives $[2,4]$ and $[2,2]$."), 17, TXT, ha="center")
    o += box(16, 312, 468, 74, rgba("yellow", .12), rgba("yellow", .5), 2, 10)
    o += mt(28, 318, 444, t("要看出差別，必須換一個轉了會變的核：\n留給練習第 $3$ 題。", "To see a difference, use a kernel that\nchanges under rotation: Exercise $3$."), 16, TXT, ha="center", lh=30, h=62)
    o += card(516, 0, 644, 400, "green", .07) + head(516, 0, 568, "green", t("二維也只差編號", "In 2-D only the numbering shifts"), 36, 18) + added(1092, 5, lang, 60 if lang == "zh" else 62)
    o += k(530, 46, 616, r"(I\star\tilde K)[i,j]=(I*K)[i+k-1,\,j+k-1]", 22, 44)
    fx, fy, c2, g2 = 590, 124, 44, 5
    for i in range(4):
        o += mt(fx - 40, fy + i * (c2 + g2) + 10, 30, f"${i}$", 16, MUT, ha="center", h=26)
        o += mt(fx + i * (c2 + g2) - 4, fy - 30, c2 + 8, f"${i}$", 16, MUT, ha="center", h=26)
        for j in range(4):
            inb = 1 <= i <= 2 and 1 <= j <= 2
            o += cell(fx + j * (c2 + g2), fy + i * (c2 + g2), N.full_block[i - 1][j - 1] if inb else "", "green" if inb else "grey", c2, c2, 18, a=.30 if inb else .08, stroke_a=.9 if inb else .35)
    o += mt(fx - 50, fy + 4 * (c2 + g2) + 4, 4 * (c2 + g2) + 60, t("$I*K$ 的位置 $i,j=0,\\dots,3$", "$I*K$ at $i,j=0,\\dots,3$"), 16, MUT, ha="center", h=28)
    o += svg(0, 0, W, 400, arrow(fx + 4 * (c2 + g2) + 10, fy + 2 * (c2 + g2), 930, fy + 2 * (c2 + g2), MUT, 3))
    o += mt(fx + 4 * (c2 + g2) - 34, fy + 2 * (c2 + g2) - 64, 160, t("有效區塊\n重新從 $0$ 編號", "valid block\nrenumbered from $0$"), 16, TXT, ha="center", lh=24)
    sx, sy = 950, fy + (c2 + g2) - 0
    for i in range(2):
        o += mt(sx - 34, sy + i * (c2 + g2) + 10, 30, f"${i}$", 16, MUT, ha="center", h=26)
        o += mt(sx + i * (c2 + g2) - 4, sy - 30, c2 + 8, f"${i}$", 16, MUT, ha="center", h=26)
        for j in range(2): o += cell(sx + j * (c2 + g2), sy + i * (c2 + g2), N.s_conv[i][j], "green", c2, c2, 18)
    o += k(sx, sy + 2 * (c2 + g2) + 2, 2 * c2 + g2, "s", 20, 32)
    o += mt(532, 366, 612, t("卷積的有效輸出從 $i,j=k-1=1$ 開始；改成從 $0$ 數，就是互相關的編號。", "Convolution's valid output starts at $i,j=k-1=1$; counted from $0$ it matches."), 16, MUT, ha="center")
    o += take(t, "核轉 $180^\\circ$ 不變時兩者相同；一般情況下，用 $\\tilde K$ 做互相關就是卷積，只差編號平移 $k-1$。", "Equal when the kernel is unchanged by rotation; in general correlating with $\\tilde K$ is convolution, shifted by $k-1$.")
    return o

# ---------------------------------------------------------------- 9: the same set of functions
def page9(lang):
    t = tr(lang)
    o = card(0, 4, 330, 296, "blue", .08) + mhead(0, 4, 330, "blue", t("卷積網路：核 $K$", "Convolution networks: kernel $K$"))
    o += card(830, 4, 330, 296, "orange", .08) + mhead(830, 4, 330, "orange", t("互相關網路：核 $\\tilde K$", "Correlation networks: kernel $\\tilde K$"))
    conv54 = [int(v) for v in N.conv1d_valid(N.x, N.w[::-1])]; assert conv54 == N.corr
    pairs = ((vec(Wk), vec(Wk[::-1]), vec(N.valid_conv)), (vec(Wk[::-1]), vec(Wk), vec(conv54)), ("K", "\\tilde K", t("相同", "same")))
    for r, (a, b, out) in enumerate(pairs):
        y = 62 + r * 76
        o += mchip(40, y, 250, "blue", f"${a}$", False, 44, 22) + mchip(870, y, 250, "orange", f"${b}$", False, 44, 22)
        o += svg(0, 0, W, 300, f'<line x1="292" y1="{y+22}" x2="868" y2="{y+22}" stroke="{GR}" stroke-width="3"/>')
        lab = t(f"輸出 ${out}$", f"output ${out}$") if r < 2 else t("一對一，輸出相同", "one to one, same output")
        o += box(430, y + 4, 300, 36, BG, None, 0, 10) + mchip(430, y + 4, 300, "green", lab, r == 2, 36, 18)
    o += mt(20, 264, 290, t("例：$x=[2,1,3]$", "example: $x=[2,1,3]$"), 16, MUT, ha="center") + mt(850, 264, 290, t("$\\tilde K$：$K$ 轉 $180^\\circ$", "$\\tilde K$: $K$ rotated $180^\\circ$"), 16, MUT, ha="center")
    fy = 316
    o += chip(0, fy, 300, "green", t("兩群是同一群函數", "the two sets are the same set"), True, 40, 18)
    o += svg(0, 0, W, 400, arrow(304, fy + 20, 356, fy + 20, MUT, 3) + arrow(664, fy + 20, 716, fy + 20, MUT, 3))
    o += chip(360, fy, 300, "green", t("所以表達能力相同", "so the expressive power is equal"), True, 40, 18)
    o += box(720, fy - 6, 440, 92, rgba("violet", .12), rgba("violet", .55), 2, 10)
    o += mt(732, fy, 416, t("梯度下降：沿損失下降的方向反覆調整\n核的數字；它只在這群函數裡找損失小的，\n不在乎慣例叫什麼。", "Gradient descent: repeatedly adjusts the\nkernel numbers downhill on the loss; it only\nsearches this set, whatever its name."), 16, TXT, name="term:" + t("梯度下降", "gradient descent"), lh=26, h=80)
    o += mt(0, 364, 700, t("表達能力：一種網路能表示的所有函數。", "Expressive power: all functions a kind of network can represent."), 16, TXT, name="term:" + t("表達能力", "expressive power"), ha="center", h=30)
    o += take(t, "兩群函數一樣，所以表達能力相同；核是學出來的，不翻也學得到。", "The two sets of functions are the same, so the expressive power is equal; the kernel is learned, so no flip is needed.")
    return o

# ---------------------------------------------------------------- 10: when the flip matters
def page10(lang):
    t = tr(lang); o = card(0, 0, 600, 330, "red", .07) + head(0, 0, 600, "red", t("手設計的核：翻轉後正負號全反", "A hand-designed kernel: flipping reverses every sign"), 36, 18)
    sign = lambda v: "blue" if v > 0 else ("red" if v < 0 else "grey")
    Hp = N.Hp.tolist()
    o += grid(70, 60, Hp, 52, 6, shade=False, bold_vals=(), size=20, roles=sign) + grid(360, 60, N.Hp_rot, 52, 6, shade=False, bold_vals=(), size=20, roles=sign)
    o += svg(0, 0, 600, 330, arrow(258, 147, 346, 147, MUT, 3))
    o += mt(244, 108, 116, t("轉 $180^\\circ$", "rotate $180^\\circ$"), 16, TXT, True, "center", h=28)
    o += mt(40, 240, 236, t("水平 Prewitt 核 $H$", "horizontal Prewitt $H$"), 17, TXT, True, "center", h=30) + k(330, 238, 236, r"\mathrm{rot}_{180}(H)=-H", 20, 36)
    o += mt(16, 284, 568, t("藍色是正、紅色是負：偵測到的邊緣也跟著變號。", "Blue is positive, red negative: the detected edges change sign too."), 16, MUT, ha="center")
    o += card(616, 0, 544, 330, "violet", .07) + head(616, 0, 468, "violet", t("跨框架搬權重", "Moving weights between frameworks"), 36, 18) + added(1092, 5, lang, 60 if lang == "zh" else 62)
    o += mchip(640, 70, 210, "blue", t("框架 $A$ 學到的核 $K$", "kernel $K$ from $A$"), False, 64, 17)
    o += mchip(926, 70, 210, "orange", t("另一種慣例的框架 $B$", "other framework $B$"), False, 64, 17)
    o += svg(0, 0, W, 330, arrow(852, 102, 922, 102, MUT, 3))
    o += mt(640, 150, 496, t("先把核轉 $180^\\circ$，再放進框架 $B$。", "Rotate the kernel by $180^\\circ$ before moving it to $B$."), 17, TXT, True, "center")
    o += mt(640, 186, 496, t("沒有轉的話，輸出會錯。", "Without the rotation the output is wrong."), 16, MUT, ha="center")
    o += box(632, 232, 512, 82, rgba("grey", .10), rgba("grey", .45), 2, 10)
    o += mt(644, 240, 488, t("練習第 $2$ 題：把 $H$ 用在一個小塊上，\n比較兩種運算的正負號。", "Exercise $2$: apply $H$ to a small patch and\ncompare the signs of the two operations."), 16, TXT, ha="center", lh=30, h=64)
    o += box(0, 344, W, 56, rgba("yellow", .12), rgba("yellow", .55), 2, 12)
    o += mt(16, 352, W - 32, t("前提：核的數字必須是學出來的。核固定、或要換慣例時，翻不翻就有差別。", "The assumption: the kernel numbers are learned. With a fixed kernel or a change of convention, the flip matters."), 18, TXT, True, "center", lh=40, h=40)
    o += take(t, "核要學，才不必管翻轉；固定的核或搬權重時，必須先轉 $180^\\circ$。", "Only a learned kernel can ignore the flip; a fixed kernel or moved weights must be rotated by $180^\\circ$ first.")
    return o

# ---------------------------------------------------------------- 11: names in code (table)
def page11(lang):
    t = tr(lang); o = ""
    cw = (440, 300, 380); xs = (0, 450, 760); rh = 56; y0 = 0
    hdr = (t("函式", "function"), t("名稱看起來像", "the name suggests"), t("實際做的", "what it computes"))
    for c in range(3): o += chip(xs[c], y0, cw[c], "grey", hdr[c], True, 44, 18)
    rowsd = (("torch.nn.functional.conv2d", t("卷積", "convolution"), t("互相關（不翻核）", "cross-correlation (no flip)"), "red", False),
             ("tf.nn.conv2d", t("卷積", "convolution"), t("互相關（不翻核）", "cross-correlation (no flip)"), "red", False),
             ("np.convolve", t("卷積", "convolution"), t("卷積（翻核）", "convolution (flips)"), "green", True),
             ("np.correlate", t("互相關", "cross-correlation"), t("互相關（不翻核）", "cross-correlation (no flip)"), "green", True))
    for r, (fn, looks, does, role, add) in enumerate(rowsd):
        y = y0 + 54 + r * (rh + 8)
        o += box(xs[0], y, cw[0], rh, rgba("grey", .10), rgba("grey", .45), 2, 10) + txt(xs[0] + 16, y + 15, cw[0] - 100, fn, 20, TXT, True, "left", 26).replace("font-weight:700", "font-weight:700;font-family:monospace")
        if add: o += added(xs[0] + cw[0] - 76, y + 15, lang, 62)
        o += box(xs[1], y, cw[1], rh, rgba("grey", .10), rgba("grey", .45), 2, 10) + txt(xs[1], y + 15, cw[1], looks, 18, TXT, False, "center", 26)
        o += box(xs[2], y, cw[2], rh, rgba(role, .22), rgba(role, .7), 2, 10) + txt(xs[2], y + 15, cw[2], does, 18, TXT, True, "center", 26)
    o += mt(0, 316, W, t("論文的公式常寫 $*$（卷積），程式裡多半算的是 $\\star$（互相關）。", "Papers often write $*$ (convolution); the code mostly computes $\\star$ (cross-correlation)."), 18, TXT, True, "center", h=32)
    o += box(0, 356, W, 46, rgba("teal", .12), rgba("teal", .55), 2, 12)
    o += mt(16, 362, W - 32, t("留給 KP05：同一個乘加也能寫成巨大的全連接層，卷積到底省下了什麼？", "Left for KP05: the same multiply-and-add could be one huge fully connected layer; what does convolution save?"), 17, TXT, False, "center", lh=34, h=34)
    o += take(t, "框架的 conv2d 名叫卷積，實際做的是互相關；要翻核的只有 np.convolve。", "A framework's conv2d is called convolution but computes cross-correlation; only np.convolve flips.")
    return o

# ---------------------------------------------------------------- 12: summary and misconceptions
def page12(lang):
    t = tr(lang); o = ""
    fs = ((t("一維卷積", "1-D convolution"), r"(x*w)[t]=\sum_{\tau}x[\tau]\,w[t-\tau]", vec(N.valid_conv), "blue"),
          (t("一維互相關", "1-D cross-correlation"), r"(x\star w)[t]=\sum_{m=0}^{k-1}x[t+m]\,w[m]", vec(N.corr), "red"),
          (t("翻核就相等", "flip and they match"), r"(x\star\tilde w)[t]=(x*w)[t+k-1]", vec(N.corr_flip), "green"))
    for c, (lab, tex, out, role) in enumerate(fs):
        x0 = c * 393
        o += card(x0, 0, 374, 150, role, .08) + head(x0, 0, 374, role, lab, 34, 17)
        o += k(x0, 42, 374, tex, 20, 56)
        o += mt(x0, 108, 374, t(f"主例子：${out}$", f"main example: ${out}$"), 17, TXT, True, "center", h=30)
    myths = ((t("核一定要先轉 $180^\\circ$，否則網路無法訓練", "The kernel must be rotated by $180^\\circ$ or it cannot train"), t("不翻也能訓練：網路會學到翻轉後的核", "It trains without the flip and learns the flipped kernel")),
             (t("式子右邊的求和指標就是輸出", "The summation index on the right is the output"), t("求和指標只負責掃過各格，和輸出無關", "The summation index only sweeps the cells")),
             (t("函式叫 conv2d，所以是真正的卷積", "A function named conv2d computes true convolution"), t("它多半在做互相關", "It mostly computes cross-correlation")),
             (t("試一個例子結果一樣，兩者就相同", "One example agreed, so the two are the same"), t("核轉 $180^\\circ$ 不變時才一樣，例如單位矩陣", "Only if the kernel is unchanged by rotation, like the identity")))
    for r, (bad, good) in enumerate(myths):
        y = 164 + r * 60
        o += box(0, y, 560, 52, rgba("red", .12), rgba("red", .55), 2, 10) + mt(12, y + 8, 540, "✗ " + bad, 17, TXT, h=36)
        o += svg(0, 0, W, 410, arrow(566, y + 26, 594, y + 26, MUT, 3))
        o += box(600, y, 560, 52, rgba("green", .12), rgba("green", .55), 2, 10) + mt(612, y + 8, 540, "✓ " + good, 17, TXT, h=36)
    o += take(t, "卷積先翻核、互相關不翻；換個核就相等，而核是學出來的，所以不翻也能訓練。", "Convolution flips and correlation does not; a flipped kernel makes them equal, and a learned kernel needs no flip.")
    return o

# ---------------------------------------------------------------- 13: extension: commutativity
def strip_cells(x0, y, vals, role, cs=58, gap=8, size=20):
    return "".join(cell(x0 + i * (cs + gap), y, v, role, cs, 46, size) for i, v in enumerate(vals))

def page13(lang):
    t = tr(lang); o = card(0, 0, 720, 190, "green", .07) + head(0, 0, 720, "green", t("完整卷積：交換也一樣", "Full convolution: swapping changes nothing"), 36, 18)
    o += k(20, 54, 200, "x*w", 22, 46) + strip_cells(230, 54, N.full, "green") + k(20, 120, 200, "w*x", 22, 46) + strip_cells(230, 120, N.full_wx, "green")
    o += chip(530, 86, 170, "green", t("相同", "same"), True, 36, 18)
    o += card(0, 206, 720, 194, "red", .07) + head(0, 206, 720, "red", t("完整互相關：交換後順序倒過來", "Full cross-correlation: swapping reverses the order"), 36, 18)
    cs, gp = 58, 8; x0 = 230
    o += k(20, 254, 200, r"x\star w", 22, 46) + strip_cells(x0, 254, N.fcorr_xw, "red") + k(20, 344, 200, r"w\star x", 22, 46) + strip_cells(x0, 344, N.fcorr_wx, "red")
    lines = "".join(f'<line x1="{x0 + i*(cs+gp) + cs/2:.0f}" y1="300" x2="{x0 + (3-i)*(cs+gp) + cs/2:.0f}" y2="344" stroke="{RD}" stroke-width="2" opacity="0.8"/>' for i in range(4))
    o += svg(0, 0, 720, 400, lines)
    o += chip(530, 300, 170, "red", t("順序倒過來", "order reversed"), True, 36, 18)
    o += card(736, 0, 424, 400, "violet", .07) + head(736, 0, 348, "violet", t("延伸：交換律", "Extension: commutativity"), 36, 18) + added(1092, 5, lang, 60 if lang == "zh" else 62, "yellow")
    o += mt(752, 50, 392, t("交換律：兩個運算元對調，結果不變。\n卷積滿足 $x*w=w*x$；互相關一般不滿足。", "Commutativity: swapping the two operands\nkeeps the result. Convolution has $x*w=w*x$;\ncross-correlation in general does not."), 16, TXT, name="term:" + t("交換律", "commutativity"), lh=26, h=78)
    o += mt(752, 140, 392, t("完整互相關：列出核和訊號有重疊的\n所有位置，訊號外補 $0$；\n$t$ 從 $-(k-1)$ 到 $n-1$。", "Full cross-correlation: every position\nwhere kernel and signal overlap, zeros\noutside; $t$ from $-(k-1)$ to $n-1$."), 16, TXT, name="term:" + t("完整互相關", "full cross-correlation"), lh=26, h=78)
    o += mt(752, 230, 392, t("$w\\star x$：把較長的 $x$ 當成核。", "$w\\star x$: the longer $x$ acts as the kernel."), 16, TXT)
    o += mt(752, 270, 392, t("傅立葉域：卷積是逐點相乘，\n互相關多一步複共軛。\n（需要訊號與系統的背景）", "Fourier domain: convolution is pointwise\nmultiplication; correlation adds a complex\nconjugate. (Needs signals and systems.)"), 16, MUT, lh=26, h=78)
    o += mt(752, 358, 392, t("對「學一個核」的網路沒有影響。", "No effect on a network that learns its kernel."), 16, TXT, True)
    o += take(t, f"卷積可交換：${vec(N.full)}$；互相關交換後是 ${vec(N.fcorr_wx)}$，順序倒過來。", f"Convolution commutes: ${vec(N.full)}$; swapped cross-correlation gives ${vec(N.fcorr_wx)}$, reversed.")
    return o

# ---------------------------------------------------------------- 14: extension: exercises (hints only, no answers)
def page14(lang):
    t = tr(lang); o = ""
    qs = ((t("第 1 題", "Exercise 1"), False, t("$x=[1,0,2]$、$w=[3,4]$：\n算出卷積與互相關的有效輸出。", "$x=[1,0,2]$, $w=[3,4]$:\ncompute the valid outputs of convolution\nand of cross-correlation."), t(f"提示：卷積先把核翻成 ${vec(N.e1_flip)}$。", f"Hint: convolution first flips the kernel to ${vec(N.e1_flip)}$.")),
          (t("第 2 題", "Exercise 2"), True, t("$3\\times3$ 小塊 $X$ 三列：$[0,0,0]$、$[1,1,1]$、$[2,2,2]$；\n水平 Prewitt 核 $H$ 三列：$[1,1,1]$、$[0,0,0]$、$[-1,-1,-1]$。\n算出 $X\\star H$ 與 $X*H$（各 $1$ 個數），說明正負號為何相反。", "Patch $X$ rows: $[0,0,0]$, $[1,1,1]$, $[2,2,2]$;\nPrewitt $H$ rows: $[1,1,1]$, $[0,0,0]$, $[-1,-1,-1]$.\nCompute $X\\star H$ and $X*H$ (one number each)\nand explain why the signs are opposite."), t("提示：先把核轉 $180^\\circ$。", "Hint: first rotate the kernel by $180^\\circ$.")),
          (t("第 3 題", "Exercise 3"), False, t("影像 $I$ 三列：$[1,2,0]$、$[1,1,2]$、$[0,1,1]$；\n核 $K$ 兩列：$[1,2]$、$[0,0]$。\n算出互相關與卷積的 $2\\times2$ 有效輸出。", "Image $I$ rows: $[1,2,0]$, $[1,1,2]$, $[0,1,1]$;\nkernel $K$ rows: $[1,2]$, $[0,0]$.\nCompute the $2\\times2$ valid outputs of\ncross-correlation and of convolution."), t(f"提示：轉 $180^\\circ$ 後的核兩列是 ${vec(N.e3_rot[0])}$、${vec(N.e3_rot[1])}$。", f"Hint: the rotated kernel has rows ${vec(N.e3_rot[0])}$ and ${vec(N.e3_rot[1])}$.")),
          (t("第 4 題", "Exercise 4"), True, t("訊號 $[3,1,4,1,5]$，核 $[1,2,1]$ 與 $[1,0,-1]$，\n各做互相關與卷積。哪一個核兩種結果相同？\n為什麼？輸出各有幾個數？", "Signal $[3,1,4,1,5]$, kernels $[1,2,1]$ and $[1,0,-1]$,\neach with correlation and convolution. For which\nkernel do the two agree, and why? How many\nnumbers does each output have?"), t("提示：比較核翻轉前後是否相同。", "Hint: compare each kernel before and after flipping.")))
    for i, (lab, add, q, hint) in enumerate(qs):
        x0 = (i % 2) * 590; y0 = (i // 2) * 202; cw = 570
        o += card(x0, y0, cw, 192, "blue", .07) + head(x0, y0, cw - (76 if add else 0), "blue", lab, 34, 18)
        if add: o += added(x0 + cw - 68, y0 + 4, lang, 62)
        nl = q.count("\n") + 1; lh = 25 if nl > 3 else 27
        o += mt(x0 + 14, y0 + 42, cw - 28, q, 16, TXT, lh=lh, h=nl * lh + 8)
        o += box(x0 + 14, y0 + 150, cw - 28, 34, rgba("yellow", .14), None, 0, 8) + mt(x0 + 14, y0 + 150, cw - 28, hint, 16, TXT, True, "center", lh=34, h=34)
    o += take(t, "先判斷要不要翻核，再逐格乘加；題目數字都可以用 np.convolve 與 np.correlate 自己核對。", "First decide whether the kernel is flipped, then multiply and add; check your numbers with np.convolve and np.correlate.")
    return o
