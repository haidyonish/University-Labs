import numpy as np
from PIL import Image


file = open("../data/model_1.obj")
v = []
for s in file:
    sp = s.split()
    if sp[0] == "v":
        v.append(list(map(float, sp[1:-1])))

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)
k = 10000
for i in range(len(v)):
    x, y = v[i]
    x = int(x*k+1000)
    y = -int(y*k+500)
    img_mat[y, x] = 255

img = Image.fromarray(img_mat, mode="RGB")
img.save("../images/img-task4.png")