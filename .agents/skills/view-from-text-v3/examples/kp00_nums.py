"""KP00 numbers (recomputed): the XOR table used as the chapter's starting point."""
PTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
def xor_class(a, b): return "A" if a != b else "B"
if __name__ == "__main__":
    for p in PTS: print(p, xor_class(*p))
    print("A on one diagonal:", [p for p in PTS if xor_class(*p) == "A"], "B on the other:", [p for p in PTS if xor_class(*p) == "B"])
