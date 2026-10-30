"""KP09 數字(講義 §1 的一層參數數目)。"""
n_in, k, filters = 4, 2, 2
out = n_in - k + 1                       # stride 1、不補邊:3
assert out == 3
n_out = out * out * filters; assert n_out == 18
n_pix = n_in * n_in; assert n_pix == 16
fc = n_pix * n_out + n_out; assert fc == 306                   # 每個輸出各連全部 16 個像素並各有偏置
conv = filters * (k * k + 1); assert conv == 10                # 兩個 2×2 核在各位置重用,各一個偏置
assert out * out == 9                                           # 每個濾波器在 9 個位置重用同樣 4 個權重
