import numpy as np
from PIL import Image


def draw_line_1(img_mat, x0, y0, x1, y1, color):
    count = 100
    step = 1.0/count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t)*x0 + t*x1)
        y = round((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = color


def draw_line_2(img_mat, x0, y0, x1, y1, color):
    count = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        img_mat[y, x] = color


def draw_line_3(img_mat, x0, y0, x1, y1, color):
    for x in range(x0, x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t)*y0 + t * y1)
        img_mat[y, x] = color


def draw_line_4(img_mat, x0, y0, x1, y1, color):
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t)*y0 + t * y1)
        img_mat[y, x] = color


def draw_line_5(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0-x1) < abs(y0-y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    for x in range(x0, x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t)*y0 + t * y1)
        if xchange:
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color


def draw_line_6(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color



def draw_line_7(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2.0*(x1-x0)*abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        if xchange:
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color

        derror += dy
        if derror > 2.0*(x1-x0)*0.5:
            derror -= 2.0*(x1-x0)*1.0
            y += y_update


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


line_functions = [
    draw_line_1, draw_line_2, draw_line_3, draw_line_4,
    draw_line_5, draw_line_6, draw_line_7, draw_line_8
]

for i, draw_line_func in enumerate(line_functions, 1):
    img_mat = np.zeros((200, 200, 3), dtype=np.uint8)

    for y in range(200):
        for x in range(200):
            img_mat[y, x] = [(x + y) % 16, (x + y) % 32, (x + y) % 64]

    count_of_lines = 21
    for k in range(count_of_lines):
        color = [255, 255, 255]
        x0, y0 = 100, 100
        x1 = int(100 + np.cos(2 * np.pi / count_of_lines * k) * 95)
        y1 = int(100 + np.sin(2 * np.pi / count_of_lines * k) * 95)
        draw_line_func(img_mat, x0, y0, x1, y1, color)

    img = Image.fromarray(img_mat, mode="RGB")
    img.save(f"../images/img-task2-{i}.png")
