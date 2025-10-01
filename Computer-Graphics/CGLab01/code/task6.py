import numpy as np
from PIL import Image


def draw_line_8(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2.0*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        if xchange:
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color

        derror += dy
        if derror > (x1-x0):
            derror -= 2.0*(x1-x0)
            y += y_update


file = open("../data/model_1.obj")
v = []
f = []
for s in file:
    sp = s.split()
    if sp[0] == "v":
        v.append(list(map(float, sp[1:-1])))
    elif sp[0] == "f":
        f_current = []
        for i in sp[1:]:
            f_current.append(int(i.split("/")[0]))
        f.append(f_current)


img_mat = np.zeros((4000, 4000, 3), dtype=np.uint8)
k = 30000
for i1, i2, i3 in f:
    x1, y1 = v[i1-1]
    x1 = int(x1 * k + 2000)
    y1 = int(y1 * k + 500)
    x2, y2 = v[i2 - 1]
    x2 = int(x2 * k + 2000)
    y2 = int(y2 * k + 500)
    x3, y3 = v[i3 - 1]
    x3 = int(x3 * k + 2000)
    y3 = int(y3 * k + 500)
    y1 = -y1
    y2 = -y2
    y3 = -y3
    draw_line_8(img_mat, x1, y1, x2, y2, [255,255,255])
    draw_line_8(img_mat, x2, y2, x3, y3, [255,255,255])
    draw_line_8(img_mat, x3, y3, x1, y1, [255,255,255])

img = Image.fromarray(img_mat, mode="RGB")
img.save("../images/img-task6.png")