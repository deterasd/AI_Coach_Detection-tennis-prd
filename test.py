import numpy as np
import cv2

# 你的投影矩陣
P1 = np.array([
    [916.626242,   0.000000, 960.250417, 0.000000],
    [  0.000000, 921.951283, 523.154606, 0.000000],
    [  0.000000,   0.000000,   1.000000, 0.000000],
])

P2 = np.array([
    [782.909772, -18.152980, 1066.677600, -255341.954492],
    [ -25.104948, 925.678666,  514.730223,   46851.486878],
    [  -0.122625,   0.020539,    0.992241,      90.876653],
])

# ===== 測試用像素點 =====
# 你可以換成真的 p1, p2（像素座標）
p1 = np.array([960, 523], dtype=np.float32).reshape(2, 1)  # 左相機中央附近
p2 = np.array([1000, 520], dtype=np.float32).reshape(2, 1) # 右相機視差 slightly shifted

# ===== Triangulation =====
point_4d = cv2.triangulatePoints(P1, P2, p1, p2)

# 轉為 3D 座標
point_4d /= point_4d[3]
X, Y, Z = point_4d[:3].flatten()

print("Triangulated 3D Point:")
print("X =", X)
print("Y =", Y)
print("Z =", Z)

# ===== Z 正負檢查 =====
if Z > 0:
    print("\n✔ Z > 0 → 3D 點在相機前方，方向正確")
else:
    print("\n❌ Z < 0 → 3D 點在相機後方，Z 軸反向")
