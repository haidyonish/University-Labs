import math
import random

import numpy as np
from PIL import Image, ImageOps

def getCoords(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def getCos(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    normal = np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 - y0, z1 - z0])
    l = np.array([0, 0, 1])
    cos = np.dot(normal, l) / (np.linalg.norm(normal) * np.linalg.norm(l))
    return cos


def drawTriangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, cos):
    intensity = max(0, -cos)

    if intensity < 0.5:
        r = int(510 * intensity)
        g = 0
        b = 0
    else:
        r = 255
        g = int(510 * (intensity - 0.5))
        b = 0

    color = [r, g, b]


    xmin = min(x0, x1, x2) if (min(x0, x1, x2) >= 0) else 0
    xmax = max(x0, x1, x2) if (max(x0, x1, x2) <= widht) else widht
    ymin = min(y0, y1, y2) if (min(y0, y1, y2) >= 0) else 0
    ymax = max(y0, y1, y2) if (max(y0, y1, y2) <= height) else height
    for x in range(math.ceil(xmin), math.ceil(xmax)):
        for y in range(math.ceil(ymin), math.ceil(ymax)):
            coords = getCoords(x, y, x0, y0, x1, y1, x2, y2)
            if all(coord >= 0 for coord in coords):
                zCoord = z0*coords[0] + z1*coords[1] + z2*coords[2]
                if zCoord < z_buff[y][x]:
                    img_mat[y, x] = color
                    z_buff[y][x] = zCoord



file = open("../data/model_1.obj")
v = []
f = []
for s in file:
    sp = s.split()
    if sp[0] == "v":
        v.append(list(map(float, sp[1:])))
    elif sp[0] == "f":
        f_current = []
        for i in sp[1:]:
            f_current.append(int(i.split("/")[0]))
        f.append(f_current)

height = 1500
widht = 1500
z_buff = np.full((height, widht), np.inf)
img_mat = np.zeros((height, widht, 3), dtype=np.uint8)
k = 10000
for i1, i2, i3 in f:
    x1, y1, z1 = v[i1 - 1]
    x2, y2, z2 = v[i2 - 1]
    x3, y3, z3 = v[i3 - 1]
    cos = getCos(x1, y1, z1, x2, y2, z2, x3, y3, z3)
    if cos < 0:
        x1 = x1 * k + 700
        y1 = y1 * k + 300
        x2 = x2 * k + 700
        y2 = y2 * k + 300
        x3 = x3 * k + 700
        y3 = y3 * k + 300

        drawTriangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, cos)


img = Image.fromarray(img_mat, mode="RGB")
img = ImageOps.flip(img)
img.save("../img/img.png")