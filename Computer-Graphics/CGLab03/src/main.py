import math
import sys
import numpy as np
from PIL import Image, ImageOps


def parse_obj(path):
    with open(path) as file:
        vertices = []
        faces = []
        for s in file:
            sp = s.split()
            if sp[0] == "v":
                vertices.append(list(map(float, sp[1:])))
            elif sp[0] == "f":
                faces_current = []
                for i in sp[1:]:
                    faces_current.append(int(i.split("/")[0]))
                faces.append(faces_current)
    return [faces, vertices]


def transform_vertices(vertices, rotation_angles, translation):
    x_angle, y_angle, z_angle = rotation_angles

    rotation_x = np.array([
        [1, 0, 0],
        [0, np.cos(x_angle), np.sin(x_angle)],
        [0, -np.sin(x_angle), np.cos(x_angle)]
    ])

    rotation_y = np.array([
        [np.cos(y_angle), 0, np.sin(y_angle)],
        [0, 1, 0],
        [-np.sin(y_angle), 0, np.cos(y_angle)]
    ])

    rotation_z = np.array([
        [np.cos(z_angle), np.sin(z_angle), 0],
        [-np.sin(z_angle), np.cos(z_angle), 0],
        [0, 0, 1]
    ])

    rotation_matrix = rotation_x @ rotation_y @ rotation_z
    transformed_vertices = []

    for vertex in vertices:
        rotated_vertex = rotation_matrix @ vertex
        translated_vertex = rotated_vertex + translation
        transformed_vertices.append(translated_vertex)

    return transformed_vertices


def calculate_barycentric_coordinates(x, y, triangle_vertices_2d):
    x0, y0 = triangle_vertices_2d[0]
    x1, y1 = triangle_vertices_2d[1]
    x2, y2 = triangle_vertices_2d[2]

    denominator = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    if abs(denominator) < 1e-10:
        return 0, 0, 0

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denominator
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denominator
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2


def project_vertex_to_screen(vertex, scale, image_center_x, image_center_y):
    x_3d, y_3d, z_3d = vertex

    x_2d = x_3d * scale / z_3d + image_center_x
    y_2d = y_3d * scale / z_3d + image_center_y

    return (x_2d, y_2d)


def calculate_light_intensity(triangle_vertices_3d, light_direction=np.array([0, 0, 1])):
    vertex1, vertex2, vertex3 = triangle_vertices_3d
    v1 = np.array(vertex1)
    v2 = np.array(vertex2)
    v3 = np.array(vertex3)

    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = np.cross(edge1, edge2)

    normal_length = np.linalg.norm(normal)
    light_length = np.linalg.norm(light_direction)

    if normal_length == 0 or light_length == 0:
        return 0

    cos_angle = np.dot(normal, light_direction) / (normal_length * light_length)
    return max(-1, min(1, cos_angle))


def get_rgb_from_hsv(h, s, v):
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c

    if h < 1 / 6:
        r, g, b = c, x, 0
    elif h < 2 / 6:
        r, g, b = x, c, 0
    elif h < 3 / 6:
        r, g, b = 0, c, x
    elif h < 4 / 6:
        r, g, b = 0, x, c
    elif h < 5 / 6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return [int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)]


def get_gold_shade(triangle_light_intensity):
    base_r, base_g, base_b = 184, 115, 51

    r = int(base_r * triangle_light_intensity)
    g = int(base_g * triangle_light_intensity * 0.9)
    b = int(base_b * triangle_light_intensity * 0.7)

    if triangle_light_intensity > 0.8:
        highlight = (triangle_light_intensity - 0.8) * 25
        r = min(255, r + int(70 * highlight))
        g = min(255, g + int(50 * highlight))
        b = min(255, b + int(20 * highlight))
    return r, g, b


def draw_triangle(triangle_vertices_2d, triangle_vertices_3d, color, z_buff, img_mat, width, height):
    z_coords_3d = [v[2] for v in triangle_vertices_3d]

    x_coords_2d = [v[0] for v in triangle_vertices_2d]
    y_coords_2d = [v[1] for v in triangle_vertices_2d]

    x_min = max(0, math.floor(min(x_coords_2d)))
    x_max = min(width, math.ceil(max(x_coords_2d)))
    y_min = max(0, math.floor(min(y_coords_2d)))
    y_max = min(height, math.ceil(max(y_coords_2d)))

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            barycentric = calculate_barycentric_coordinates(x, y, triangle_vertices_2d)
            if all(coord >= 0 for coord in barycentric):
                depth = sum(barycentric[i] * z_coords_3d[i] for i in range(3))
                if depth < z_buff[y, x]:
                    img_mat[y, x] = color
                    z_buff[y, x] = depth


def print_progress_bar(iteration, total, length=50):
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    sys.stdout.write(f'\rПрогресс: |{bar}| {iteration}/{total} ({percent:.1f}%)')
    sys.stdout.flush()


def draw_object(obj, scale, width, height, z_buff, img_mat):
    faces, vertices = obj
    total_faces = len(faces)

    image_center_x = width / 2
    image_center_y = height / 2

    for idx, face in enumerate(faces):
        if idx % 10 == 0 or idx == total_faces - 1:
            print_progress_bar(idx + 1, total_faces)

        i1, i2, i3 = face
        triangle_vertices_3d = [vertices[i1 - 1], vertices[i2 - 1], vertices[i3 - 1]]

        cos_angle = calculate_light_intensity(triangle_vertices_3d)

        if cos_angle < 0:
            triangle_vertices_2d = [
                project_vertex_to_screen(vertex, scale, image_center_x, image_center_y)
                for vertex in triangle_vertices_3d
            ]
            color = get_gold_shade(-cos_angle)
            draw_triangle(triangle_vertices_2d, triangle_vertices_3d, color, z_buff, img_mat, width, height)

    print()


def render_img(obj_path, width, height, scale, rotation, translation):
    model = parse_obj(obj_path)
    faces, vertices = model

    z_buff = np.full((height, width), np.inf)
    img_mat = np.zeros((height, width, 3), dtype=np.uint8)

    transformed_vertices = transform_vertices(vertices, rotation, translation)
    transformed_model = [faces, transformed_vertices]

    print("Начинается рендеринг модели...")
    draw_object(transformed_model, scale, width, height, z_buff, img_mat)
    print("Рендеринг завершен!")

    img = Image.fromarray(img_mat, mode="RGB")
    img = ImageOps.flip(img)
    return img


if __name__ == "__main__":
    rotation = (np.radians(0), np.radians(180), np.radians(0))
    translation = np.array([0, -0.05, 0.3])

    img = render_img(
        "../data/model_1.obj",
        width=2000,
        height=2000,
        scale=3000,
        rotation=rotation,
        translation=translation
    )

    img.save("../img/img.png")
    print("Изображение сохранено!")