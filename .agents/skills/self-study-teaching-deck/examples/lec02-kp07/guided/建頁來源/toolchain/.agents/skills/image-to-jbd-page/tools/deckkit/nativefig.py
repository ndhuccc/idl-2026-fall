"""Only smooth_outline() (Catmull-Rom curve through JianBauDau `smooth` points); used by preview_deck.py. Copied from view-from-text/tools/deckkit/nativefig.py."""


def smooth_outline(pts, n=10, closed=True):
    """Outline through pts = [(x, y, smooth_flag), ...]: sharp vertices stay corners, smooth ones are joined by a Catmull-Rom (cubic Hermite) curve
    (JianBauDau `smooth: true`). closed=True wraps around (polygon); closed=False is an open polyline whose end points take the tangent from their only neighbour.
    Returns a polyline [(x, y), ...] sampled n points per curved segment."""
    m = len(pts); out = []
    P = [(p[0], p[1]) for p in pts]; S = [bool(p[2]) if len(p) > 2 else False for p in pts]
    if closed: tan = [((P[(i + 1) % m][0] - P[i - 1][0]) / 2, (P[(i + 1) % m][1] - P[i - 1][1]) / 2) if S[i] else (0.0, 0.0) for i in range(m)]
    else:
        def tg(i):
            a, b = P[max(i - 1, 0)], P[min(i + 1, m - 1)]; d = 2 if 0 < i < m - 1 else 1
            return ((b[0] - a[0]) / d, (b[1] - a[1]) / d)
        tan = [tg(i) if S[i] else (0.0, 0.0) for i in range(m)]
    for i in range(m if closed else m - 1):
        j = (i + 1) % m; p0, p1, m0, m1 = P[i], P[j], tan[i], tan[j]; out.append(p0)
        if S[i] or S[j]:
            for k in range(1, n):
                t = k / n; h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
                out.append((h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0], h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1]))
    if not closed: out.append(P[-1])
    return out
