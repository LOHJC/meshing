
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import argparse
from generate_voxels import show_shape

def load_voxel(filename):
    return np.load(filename)

def show_mesh(mesh):
    # Visualize per-face voxel labels when available.
    plotter = pv.Plotter()
    if "voxel_value" in mesh.cell_data:
        plotter.add_mesh(mesh, scalars="voxel_value", show_edges=True, cmap="tab20")
    else:
        plotter.add_mesh(mesh, show_edges=True)
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
                    visited[i:i + rect_w, j:j + rect_h] = True

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

def marching_cubes(voxel_data, iso_value=0.5, spacing=(1.0, 1.0, 1.0)):
    # Placeholder for the marching cubes algorithm
    # This function would convert voxel data to a mesh representation
    pass

def surface_nets(voxel_data):
    # Placeholder for the surface nets algorithm
    # This function would convert voxel data to a mesh representation
    pass

def dual_contouring(voxel_data):
    # Placeholder for the dual contouring algorithm
    # This function would convert voxel data to a mesh representation
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert voxels to mesh")
    parser.add_argument("--shape", type=str, choices=["sphere", "cube"], default="sphere",
                        help="Shape to process: sphere or cube")
    args = parser.parse_args()
    
    if args.shape == "sphere":
        shape = load_voxel("sphere_2_color.npy")
        
    elif args.shape == "cube":
        shape= load_voxel("cube_2_color.npy")
        
    show_shape(shape)
    mesh = greedy_mesh(shape)
    show_mesh(mesh)
    