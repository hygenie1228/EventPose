"""CPU perspective-correct mesh z-buffer for diagnostic video panels."""
import cv2
import numpy as np
from numba import njit


@njit(cache=True)
def _triangles(image, depth, uv, z, faces, colors):
    height, width = depth.shape
    for face in range(len(faces)):
        a, b, c = faces[face]
        if min(z[a], z[b], z[c]) <= .02: continue
        x0, y0 = uv[a]; x1, y1 = uv[b]; x2, y2 = uv[c]
        low_x = max(0, int(np.floor(min(x0, x1, x2))))
        high_x = min(width - 1, int(np.ceil(max(x0, x1, x2))))
        low_y = max(0, int(np.floor(min(y0, y1, y2))))
        high_y = min(height - 1, int(np.ceil(max(y0, y1, y2))))
        determinant = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(determinant) < 1e-9: continue
        for y in range(low_y, high_y+1):
            for x in range(low_x, high_x+1):
                wa = ((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/determinant
                wb = ((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/determinant
                wc = 1-wa-wb
                if min(wa, wb, wc) < -1e-6: continue
                inverse_z = wa/z[a]+wb/z[b]+wc/z[c]
                if inverse_z <= 0: continue
                value = 1/inverse_z
                if value < depth[y, x]:
                    depth[y, x] = value
                    image[y, x, 0] = colors[face, 0]
                    image[y, x, 1] = colors[face, 1]
                    image[y, x, 2] = colors[face, 2]


def mesh(image, vertices, faces, K, alpha=1., background_depth=None):
    rendered = image.copy()
    depth = np.full(image.shape[:2], np.inf, np.float32) if background_depth is None else background_depth.copy()
    before = depth.copy()
    palette = ((215, 145, 75), (160, 95, 220), (75, 175, 230))
    for person, points in enumerate(vertices):
        points = np.asarray(points, np.float64)
        projected = points @ K.T
        uv = projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)
        triangles = points[faces]
        normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-10)
        brightness = .35 + .65*np.abs(normal @ np.array([.25, -.35, -.902]))
        colors = np.clip(brightness[:, None]*np.array(palette[person % len(palette)])[None], 0, 255).astype(np.uint8)
        _triangles(rendered, depth, uv, points[:, 2], np.asarray(faces, np.int32), colors)
    mask = depth < before
    if alpha != 1:
        rendered[mask] = np.rint(alpha*rendered[mask]+(1-alpha)*image[mask]).astype(np.uint8)
    return rendered, depth, int(mask.sum())


def pointcloud(points, colors, eye, rotation, K, size=(360, 640)):
    image = np.full((*size, 3), 24, np.uint8); depth = np.full(size, np.inf, np.float32)
    value = (points-eye) @ rotation.T; projected = value @ K.T
    uv = np.rint(projected[:, :2]/np.maximum(projected[:, 2:], 1e-8)).astype(np.int64)
    keep = (value[:, 2]>.05)&(uv[:, 0]>=0)&(uv[:, 0]<size[1])&(uv[:, 1]>=0)&(uv[:, 1]<size[0])
    uv, z, color = uv[keep], value[keep, 2], colors[keep]
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        u, v = uv[:, 0]+dx, uv[:, 1]+dy
        valid = (u<size[1])&(v<size[0]); u, v, zz, cc = u[valid], v[valid], z[valid], color[valid]
        order = np.argsort(zz)
        _, first = np.unique(v[order]*size[1]+u[order], return_index=True)
        chosen = order[first]; u, v, zz, cc = u[chosen], v[chosen], zz[chosen], cc[chosen]
        front = zz < depth[v, u]
        image[v[front], u[front]] = cc[front]; depth[v[front], u[front]] = zz[front]
    return image, depth


def labelled_panel(image, title):
    result = np.full((400, 640, 3), 24, np.uint8)
    if image.shape[:2] != (360, 640): image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
    result[34:394] = image
    cv2.putText(result, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .52, (240, 240, 240), 1, cv2.LINE_AA)
    return result
