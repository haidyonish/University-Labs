import numpy as np
from PIL import Image

def mat_1(height, width):
    img_mat = np.zeros((height, width), dtype=np.uint8)
    return img_mat


def mat_2(height, width):
    img_mat = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img_mat[i, j] = 255
    return img_mat


def mat_3(height, width):
    img_mat = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img_mat[i, j] = [255, 0, 0]
    return img_mat


def mat_4(height, width):
    img_mat = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            intensity = int((i + j) / (height + width - 2) * 255)
            img_mat[i, j] = [intensity, intensity, intensity]
    return img_mat

images = []
images.append(Image.fromarray(mat_1(400, 800), mode="L"))
images.append(Image.fromarray(mat_2(400, 800), mode="L"))
images.append(Image.fromarray(mat_3(400, 800), mode="RGB"))
images.append(Image.fromarray(mat_4(400, 800), mode="RGB"))
for i in range(len(images)):
    images[i].save(f"../images/img-task1-{i+1}.png")