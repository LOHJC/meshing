import os
import numpy as np
import pyvista as pv
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_voxels import show_shape
from utils import *


def load_voxel(filename):
    return np.load(filename)


def generate_terrain_voxel(
    size=(15, 15, 15),
    padding=(1, 1, 1),
    max_height=10,
    min_height=1,
    max_value=2,
    base_area=(2, 2),
):
    terrain = np.zeros(size, dtype=np.int8)
    for x in range(padding[0], size[0] - padding[0], base_area[0]):
        for y in range(padding[1], size[1] - padding[1], base_area[1]):
            height = np.random.randint(min_height, max_height + 1)
            material = np.random.randint(1, max_value + 1)

            # Prevent slicing outside the padding boundaries
            x_end = min(x + base_area[0], size[0] - padding[0])
            y_end = min(y + base_area[1], size[1] - padding[1])

            # Fill the entire 2x2 columns up to the chosen height
            terrain[x:x_end, y:y_end, :height] = material
    return terrain


def show_mesh(mesh):
    # Visualize per-face voxel labels when available.
    plotter = pv.Plotter()
    if "voxel_value" in mesh.cell_data:
        plotter.add_mesh(mesh, scalars="voxel_value", show_edges=True, cmap="tab20")
    else:
        plotter.add_mesh(mesh, show_edges=True)
    plotter.show_axes()
    plotter.camera.azimuth = -100
    plotter.camera.elevation = -25
    plotter.show()


def greedy_mesh(voxel_data):
    # Grid dimensions are in (x, y, z) order.
    width, height, depth = voxel_data.shape

    # Any value > 0 is treated as filled voxel space.
    occupied = voxel_data > 0

    # Incremental mesh buffers.
    vertices = []
    faces = []
    face_values = []
    vertex_map = {}

    def add_vertex(coord):
        # Reuse identical coordinates to avoid duplicate vertices.
        key = (int(coord[0]), int(coord[1]), int(coord[2]))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append([key[0], key[1], key[2]])
        return vertex_map[key]

    def add_quad(v0, v1, v2, v3, value):
        # Store one quad face and its source voxel value (used for coloring).
        i0 = add_vertex(v0)
        i1 = add_vertex(v1)
        i2 = add_vertex(v2)
        i3 = add_vertex(v3)
        faces.extend([4, i0, i1, i2, i3])
        face_values.append(value)

    def greedy_sweep(axis):
        # Sweep planes perpendicular to one axis and greedily merge exposed faces.
        dims = (width, height, depth)

        # sometimes it better to swap u <-> v for better meshing
        # u = (axis + 1) % 3
        # v = (axis + 2) % 3
        v = (axis + 1) % 3
        u = (axis + 2) % 3

        for d in range(dims[axis] + 1):
            # mask: +1/-1 encodes face orientation, 0 means no visible face.
            mask = np.zeros((dims[u], dims[v]), dtype=np.int8)
            # values tracks voxel label so different materials are not merged.
            values = np.zeros((dims[u], dims[v]), dtype=voxel_data.dtype)

            for i in range(dims[u]):
                for j in range(dims[v]):
                    left_filled = False
                    right_filled = False

                    if d > 0:
                        left = [0, 0, 0]
                        left[axis] = d - 1
                        left[u] = i
                        left[v] = j
                        left_filled = occupied[left[0], left[1], left[2]]

                    if d < dims[axis]:
                        right = [0, 0, 0]
                        right[axis] = d
                        right[u] = i
                        right[v] = j
                        right_filled = occupied[right[0], right[1], right[2]]

                    if left_filled and not right_filled:
                        mask[i, j] = 1
                        values[i, j] = voxel_data[left[0], left[1], left[2]]
                    elif right_filled and not left_filled:
                        mask[i, j] = -1
                        values[i, j] = voxel_data[right[0], right[1], right[2]]

            # Greedy rectangle merge on each 2D mask slice.
            visited = np.zeros_like(mask, dtype=bool)
            for i in range(dims[u]):
                for j in range(dims[v]):
                    if visited[i, j] or mask[i, j] == 0:
                        continue

                    sign = mask[i, j]
                    value = values[i, j]

                    rect_w = 1
                    while (
                        i + rect_w < dims[u]
                        and (not visited[i + rect_w, j])
                        and mask[i + rect_w, j] == sign
                        and values[i + rect_w, j] == value
                    ):
                        rect_w += 1

                    # Expand height only if all cells in the next row still match.
                    rect_h = 1
                    can_grow = True
                    while j + rect_h < dims[v] and can_grow:
                        for ii in range(i, i + rect_w):
                            if (
                                visited[ii, j + rect_h]
                                or mask[ii, j + rect_h] != sign
                                or values[ii, j + rect_h] != value
                            ):
                                can_grow = False
                                break
                        if can_grow:
                            rect_h += 1

                    # Mark consumed cells so each visible face is emitted once.
                    visited[i : i + rect_w, j : j + rect_h] = True

                    # Convert merged rectangle back into 3D quad corners.
                    base = np.array([0, 0, 0], dtype=int)
                    base[axis] = d
                    base[u] = i
                    base[v] = j

                    du = np.array([0, 0, 0], dtype=int)
                    dv = np.array([0, 0, 0], dtype=int)
                    du[u] = rect_w
                    dv[v] = rect_h

                    c0 = base
                    c1 = base + du
                    c2 = base + du + dv
                    c3 = base + dv

                    # Keep winding consistent with face orientation.
                    if sign == 1:
                        add_quad(c0, c1, c2, c3, value)
                    else:
                        add_quad(c0, c3, c2, c1, value)

    # Sweep depth-facing planes first, then complete other orientations.
    greedy_sweep(axis=2)
    greedy_sweep(axis=0)
    greedy_sweep(axis=1)

    if not vertices:
        return pv.PolyData()

    # Build final PolyData and attach per-face voxel labels for rendering.
    mesh = pv.PolyData(np.array(vertices, dtype=float), np.array(faces, dtype=np.int64))
    mesh.cell_data["voxel_value"] = np.array(face_values)
    return mesh


# https://gist.github.com/dwilliamson/c041e3454a713e58baf6e4f8e5fffecd
# refer to marching_cube_utils.py
def marching_cube(voxel_data, iso_value=0.5):
    # voxel_data axes are:
    #
    #      z
    #      |     y
    #      |   /
    #      | /
    #      +----- x
    #
    # Axes are:
    #
    #      y
    #      |     z
    #      |   /
    #      | /
    #      +----- x
    #
    # Vertex and edge layout:
    #
    #            6             7
    #            +-------------+               +-----6-------+
    #          / |           / |             / |            /|
    #        /   |         /   |          11   7         10   5
    #    2 +-----+-------+  3  |         +-----+2------+     |
    #      |   4 +-------+-----+ 5       |     +-----4-+-----+
    #      |   /         |   /           3   8         1   9
    #      | /           | /             | /           | /
    #    0 +-------------+ 1             +------0------+
    #
    # Triangulation cases are generated prioritising rotations over inversions, which can introduce non-manifold geometry.
    #
    vertex_map = {}
    vertices = []
    faces = []
    face_values = []  # for coloring

    voxel_data = np.pad(voxel_data, pad_width=1, mode='constant', constant_values=0)
    width, height, depth = voxel_data.shape
    cube_x, cube_y, cube_z = width - 1, height - 1, depth - 1
    for i in range(cube_x):
        for j in range(cube_y):
            for k in range(cube_z):
                # get the corner X positions for the current cube
                corners_pos = np.array(
                    [
                        [i, j, k],  # bottom left front
                        [i + 1, j, k],  # bottom right front
                        [i, j, k + 1],  # top left front
                        [i + 1, j, k + 1],  # top right front
                        [i, j + 1, k],  # bottom left back
                        [i + 1, j + 1, k],  # bottom right back
                        [i, j + 1, k + 1],  # top left back
                        [i + 1, j + 1, k + 1],  # top right back
                    ]
                )

                # check if corner X is hit
                corners_hit = []
                for corner_pos in corners_pos:
                    x, y, z = corner_pos
                    corners_hit.append(voxel_data[x, y, z] > iso_value)

                # based on the corner hits, determine the case index for the marching cubes algorithm
                case_index = (
                    (1 if corners_hit[0] else 0)
                    | (2 if corners_hit[1] else 0)
                    | (4 if corners_hit[2] else 0)
                    | (8 if corners_hit[3] else 0)
                    | (16 if corners_hit[4] else 0)
                    | (32 if corners_hit[5] else 0)
                    | (64 if corners_hit[6] else 0)
                    | (128 if corners_hit[7] else 0)
                )
                triangles = TriangleTable[case_index]
                for t in range(0, len(triangles), 3):
                    if triangles[t] == -1:
                        break
                    face = [3]  # Face with 3 vertices (triangle)
                    edge_pts = []
                    for _t in range(t, t + 3):
                        edge_pts.append(EdgeVertexIndices[triangles[_t]])
                    for edge in edge_pts:
                        # Compute the vertex position for this edge
                        v0_idx, v1_idx = edge
                        v0 = corners_pos[v0_idx]
                        v1 = corners_pos[v1_idx]
                        vertex_pos = (v0 + v1) / 2.0  # Midpoint for simplicity

                        # Add vertex to the mesh
                        vertex_key = tuple(vertex_pos)
                        if vertex_key not in vertex_map:
                            vertex_map[vertex_key] = len(vertices)
                            vertices.append(vertex_pos)

                        # Add vertex index to face
                        face_idx = vertex_map[vertex_key]
                        face.append(face_idx)

                    faces.append(face)

                    # choose the max voxel value from the corners of the cube for coloring
                    color = 0
                    for corner_hit in corners_hit:
                        if corner_hit:
                            color = max(
                                color,
                                voxel_data[
                                    tuple(corners_pos[corners_hit.index(corner_hit)])
                                ],
                            )
                    face_values.append(color)

    print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
    mesh = pv.PolyData(np.array(vertices, dtype=float), np.array(faces, dtype=np.int64))
    mesh.cell_data["voxel_value"] = np.array(face_values)
    return mesh


def surface_nets(voxel_data, iso_value=0.5):
    #
    # Vertex and edge layout:
    #
    #            6             7
    #            +-------------+               +-----6-------+
    #          / |           / |             / |            /|
    #        /   |         /   |          11   7         10   5
    #    2 +-----+-------+  3  |         +-----+2------+     |
    #      |   4 +-------+-----+ 5       |     +-----4-+-----+
    #      |   /         |   /           3   8         1   9
    #      | /           | /             | /           | /
    #    0 +-------------+ 1             +------0------+
    #
    # Triangulation cases are generated prioritising rotations over inversions, which can introduce non-manifold geometry.
    #
    vertex_map = {}
    vertices = []
    faces = []
    face_values = []  # for coloring
    cube_vertex_idx = {}

    width, height, depth = voxel_data.shape
    cube_x, cube_y, cube_z = width - 1, height - 1, depth - 1
    for i in range(cube_x):
        for j in range(cube_y):
            for k in range(cube_z):
                # get the corner X positions for the current cube
                corners_pos = np.array(
                    [
                        [i, j, k],  # bottom left front
                        [i + 1, j, k],  # bottom right front
                        [i, j, k + 1],  # top left front
                        [i + 1, j, k + 1],  # top right front
                        [i, j + 1, k],  # bottom left back
                        [i + 1, j + 1, k],  # bottom right back
                        [i, j + 1, k + 1],  # top left back
                        [i + 1, j + 1, k + 1],  # top right back
                    ]
                )

                # check if corner X is hit
                corners_hit = []
                for corner_pos in corners_pos:
                    x, y, z = corner_pos
                    corners_hit.append(voxel_data[x, y, z] > iso_value)

                intersection_points = []
                for edges in EdgeVertexIndices:
                    v0_idx, v1_idx = edges
                    v0 = corners_pos[v0_idx]
                    v1 = corners_pos[v1_idx]
                    if corners_hit[v0_idx] != corners_hit[v1_idx]:
                        intersection_points.append((v0 + v1) / 2.0)

                vertex_pos = (
                    np.mean(intersection_points, axis=0)
                    if intersection_points
                    else None
                )
                if vertex_pos is not None:
                    # Add vertex to the mesh
                    vertex_key = tuple(vertex_pos)
                    if vertex_key not in vertex_map:
                        vertex_map[vertex_key] = len(vertices)
                        cube_vertex_idx[(i, j, k)] = len(vertices)
                        vertices.append(vertex_pos)

    # face generation
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                # Check X-Axis edge crossing: (i, j, k) -> (i + 1, j, k)
                if i < width - 1 and j < height and k < depth:
                    val_current = voxel_data[i, j, k] > iso_value
                    val_next = voxel_data[i + 1, j, k] > iso_value
                    if val_current != val_next:
                        # Fetch indices of the 4 dual cells surrounding this edge
                        c0 = cube_vertex_idx.get((i, j, k))
                        c1 = cube_vertex_idx.get((i, j, k - 1))
                        c2 = cube_vertex_idx.get((i, j - 1, k - 1))
                        c3 = cube_vertex_idx.get((i, j - 1, k))

                        if all(c is not None for c in [c0, c1, c2, c3]):
                            if val_current:
                                faces.append([4, c0, c1, c2, c3])
                            else:
                                faces.append([4, c3, c2, c1, c0])

                            face_values.append(
                                max(voxel_data[i, j, k], voxel_data[i + 1, j, k])
                            )

                # Check Y-Axis edge crossing: (i, j, k) -> (i, j + 1, k)
                if i < width and j < height - 1 and k < depth:
                    val_current = voxel_data[i, j, k] > iso_value
                    val_next = voxel_data[i, j + 1, k] > iso_value
                    if val_current != val_next:
                        # Fetch indices of the 4 dual cells surrounding this edge
                        c0 = cube_vertex_idx.get((i, j, k))
                        c1 = cube_vertex_idx.get((i - 1, j, k))
                        c2 = cube_vertex_idx.get((i - 1, j, k - 1))
                        c3 = cube_vertex_idx.get((i, j, k - 1))

                        if all(c is not None for c in [c0, c1, c2, c3]):
                            if val_current:
                                faces.append([4, c3, c2, c1, c0])
                            else:
                                faces.append([4, c0, c1, c2, c3])

                            face_values.append(
                                max(voxel_data[i, j, k], voxel_data[i, j + 1, k])
                            )

                # Check Z-Axis edge crossing: (i, j, k) -> (i, j, k + 1)
                if i < width and j < height and k < depth - 1:
                    val_current = voxel_data[i, j, k] > iso_value
                    val_next = voxel_data[i, j, k + 1] > iso_value
                    if val_current != val_next:
                        # Fetch indices of the 4 dual cells surrounding this edge
                        c0 = cube_vertex_idx.get((i, j, k))
                        c1 = cube_vertex_idx.get((i, j - 1, k))
                        c2 = cube_vertex_idx.get((i - 1, j - 1, k))
                        c3 = cube_vertex_idx.get((i - 1, j, k))

                        if all(c is not None for c in [c0, c1, c2, c3]):
                            if val_current:
                                faces.append([4, c0, c1, c2, c3])
                            else:
                                faces.append([4, c3, c2, c1, c0])

                            face_values.append(
                                max(voxel_data[i, j, k], voxel_data[i, j, k + 1])
                            )

    print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
    mesh = pv.PolyData(np.array(vertices, dtype=float), np.array(faces, dtype=np.int64))
    mesh.cell_data["voxel_value"] = np.array(face_values)
    return mesh


def dual_contouring(voxel_data, iso_value=0.5):
    #
    # Vertex and edge layout:
    #
    #            6             7
    #            +-------------+               +-----6-------+
    #          / |           / |             / |            /|
    #        /   |         /   |          11   7         10   5
    #    2 +-----+-------+  3  |         +-----+2------+     |
    #      |   4 +-------+-----+ 5       |     +-----4-+-----+
    #      |   /         |   /           3   8         1   9
    #      | /           | /             | /           | /
    #    0 +-------------+ 1             +------0------+
    #
    # Triangulation cases are generated prioritising rotations over inversions, which can introduce non-manifold geometry.
    #
    vertex_map = {}
    vertices = []
    faces = []
    face_values = []  # for coloring
    cube_vertex_idx = {}

    width, height, depth = voxel_data.shape
    cube_x, cube_y, cube_z = width - 1, height - 1, depth - 1
    for i in range(cube_x):
        for j in range(cube_y):
            for k in range(cube_z):
                # get the corner X positions for the current cube
                corners_pos = np.array(
                    [
                        [i, j, k],  # bottom left front
                        [i + 1, j, k],  # bottom right front
                        [i, j, k + 1],  # top left front
                        [i + 1, j, k + 1],  # top right front
                        [i, j + 1, k],  # bottom left back
                        [i + 1, j + 1, k],  # bottom right back
                        [i, j + 1, k + 1],  # top left back
                        [i + 1, j + 1, k + 1],  # top right back
                    ]
                )

                # check if corner X is hit
                corners_hit = []
                for corner_pos in corners_pos:
                    x, y, z = corner_pos
                    corners_hit.append(voxel_data[x, y, z] > iso_value)

                intersection_points = []
                intersection_normals = []
                for edges in EdgeVertexIndices:
                    v0_idx, v1_idx = edges
                    v0 = corners_pos[v0_idx]
                    v1 = corners_pos[v1_idx]
                    if corners_hit[v0_idx] != corners_hit[v1_idx]:
                        intersection_points.append((v0 + v1) / 2.0)
                vertex_pos = None

                # TODO:
                # dual contour vertex pos calculation
                # calculate the normals at each intersection points
                def calc_normal(point, eps=1.0):
                    x, y, z = map(int, point)
                    max_x, max_y, max_z = np.array(voxel_data.shape) - 1
                    x_pos = voxel_data[int(min(max(x + eps, 0), max_x)), y, z]
                    x_neg = voxel_data[int(max(min(x - eps, max_x), 0)), y, z]
                    y_pos = voxel_data[x, int(min(max(y + eps, 0), max_y)), z]
                    y_neg = voxel_data[x, int(max(min(y - eps, max_y), 0)), z]
                    z_pos = voxel_data[x, y, int(min(max(z + eps, 0), max_z))]
                    z_neg = voxel_data[x, y, int(max(min(z - eps, max_z), 0))]
                    normal = np.array(
                        [x_pos - x_neg, y_pos - y_neg, z_pos - z_neg], dtype=float
                    )
                    norm = np.linalg.norm(normal)
                    if norm == 0:
                        return np.array([0.0, 1.0, 0.0])

                    return normal / norm

                if intersection_points:
                    intersection_normals = [calc_normal(p) for p in intersection_points]
                    if len(intersection_points) != len(intersection_normals):
                        raise ValueError(
                            "intersection_points and intersection_normals must have the same length"
                        )

                    cell_min = np.array([i, j, k])
                    cell_max = np.array([i + 1, j + 1, k + 1])

                    # qef_solver
                    A = np.array(intersection_normals)
                    P = np.array(intersection_points)
                    b = np.sum(A * P, axis=1)

                    # Use NumPy's linear least-squares solver to find the optimal vertex position x
                    vertex_pos, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

                    # Calculate fallback mass center (this is your old Naive Surface Net position!)
                    mass_center = np.mean(P, axis=0)

                    # Safeguard check:
                    # If rank < 3, the equations don't have a unique solution (e.g., all normals point same way).
                    # If any coordinate of vertex_pos is outside the voxel cell, the math overshot.
                    if (
                        rank < 3
                        or np.any(vertex_pos < cell_min)
                        or np.any(vertex_pos > cell_max)
                    ):
                        vertex_pos = mass_center

                if vertex_pos is not None:
                    # Add vertex to the mesh
                    vertex_key = tuple(vertex_pos)
                    if vertex_key not in vertex_map:
                        vertex_map[vertex_key] = len(vertices)
                        cube_vertex_idx[(i, j, k)] = len(vertices)
                        vertices.append(vertex_pos)

    # face generation
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                # Check X-Axis edge crossing: (i, j, k) -> (i + 1, j, k)
                if i < width - 1 and j < height and k < depth:
                    val_current = voxel_data[i, j, k] > iso_value
                    val_next = voxel_data[i + 1, j, k] > iso_value
                    if val_current != val_next:
                        # Fetch indices of the 4 dual cells surrounding this edge
                        c0 = cube_vertex_idx.get((i, j, k))
                        c1 = cube_vertex_idx.get((i, j, k - 1))
                        c2 = cube_vertex_idx.get((i, j - 1, k - 1))
                        c3 = cube_vertex_idx.get((i, j - 1, k))

                        if all(c is not None for c in [c0, c1, c2, c3]):
                            if val_current:
                                faces.append([4, c0, c1, c2, c3])
                            else:
                                faces.append([4, c3, c2, c1, c0])

                            face_values.append(
                                max(voxel_data[i, j, k], voxel_data[i + 1, j, k])
                            )

                # Check Y-Axis edge crossing: (i, j, k) -> (i, j + 1, k)
                if i < width and j < height - 1 and k < depth:
                    val_current = voxel_data[i, j, k] > iso_value
                    val_next = voxel_data[i, j + 1, k] > iso_value
                    if val_current != val_next:
                        # Fetch indices of the 4 dual cells surrounding this edge
                        c0 = cube_vertex_idx.get((i, j, k))
                        c1 = cube_vertex_idx.get((i - 1, j, k))
                        c2 = cube_vertex_idx.get((i - 1, j, k - 1))
                        c3 = cube_vertex_idx.get((i, j, k - 1))

                        if all(c is not None for c in [c0, c1, c2, c3]):
                            if val_current:
                                faces.append([4, c3, c2, c1, c0])
                            else:
                                faces.append([4, c0, c1, c2, c3])

                            face_values.append(
                                max(voxel_data[i, j, k], voxel_data[i, j + 1, k])
                            )

                # Check Z-Axis edge crossing: (i, j, k) -> (i, j, k + 1)
                if i < width and j < height and k < depth - 1:
                    val_current = voxel_data[i, j, k] > iso_value
                    val_next = voxel_data[i, j, k + 1] > iso_value
                    if val_current != val_next:
                        # Fetch indices of the 4 dual cells surrounding this edge
                        c0 = cube_vertex_idx.get((i, j, k))
                        c1 = cube_vertex_idx.get((i, j - 1, k))
                        c2 = cube_vertex_idx.get((i - 1, j - 1, k))
                        c3 = cube_vertex_idx.get((i - 1, j, k))

                        if all(c is not None for c in [c0, c1, c2, c3]):
                            if val_current:
                                faces.append([4, c0, c1, c2, c3])
                            else:
                                faces.append([4, c3, c2, c1, c0])

                            face_values.append(
                                max(voxel_data[i, j, k], voxel_data[i, j, k + 1])
                            )

    print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
    mesh = pv.PolyData(np.array(vertices, dtype=float), np.array(faces, dtype=np.int64))
    mesh.cell_data["voxel_value"] = np.array(face_values)
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert voxels to mesh")
    parser.add_argument(
        "--shape",
        type=str,
        choices=["random", "sphere", "cube"],
        default="random",
        help="Shape to process: random, sphere, or cube",
    )
    args = parser.parse_args()

    if args.shape == "random":
        shape = generate_terrain_voxel()

    elif args.shape == "sphere":
        shape = load_voxel("sphere_2_color.npy")

    elif args.shape == "cube":
        shape = load_voxel("cube_2_color.npy")

    show_shape(shape)
    # mesh = greedy_mesh(shape)
    # mesh = marching_cube(shape)
    # mesh = surface_nets(shape)
    mesh = dual_contouring(shape)
    show_mesh(mesh)
