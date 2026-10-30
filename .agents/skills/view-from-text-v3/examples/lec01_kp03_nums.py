"""KP03 數字(講義 §2–§3)。"""
from lec01_kp02_nums import pts
fruit = [("apple", 7.2), ("cucumber", 6.8), ("apple", 6.5)]       # 講義 §2 的 3 筆假想教學樣本(第 2 筆是黃瓜)
assert [round(a - b, 1) for (_, a), (_, b) in zip(fruit, fruit[1:])] == [0.4, 0.3] and round(7.2 - 6.5, 1) == 0.7
s = [x + y - 5 for x, y in pts]                                   # s(x)=x1+x2-5, b=-5
assert s[0] == -1.5 and s[6] == 10.5                              # 講義的兩個代入
cls = ["l2" if v >= 0 else "l1" for v in s]
assert cls[0] == "l1" and cls[6] == "l2" and cls.count("l1") == 1 and cls.count("l2") == 11
assert [pts[i] for i, v in enumerate(s) if v == 0] == [(2.5, 2.5), (1.0, 4.0)]     # 落在邊界上的點歸 l2(分數 >= 0)
assert (min(s), max(s)) == (-1.5, 11.0)
f = lambda v: f"{v:.1f}".replace("-", "-")
