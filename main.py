import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from voxel2mesh.voxel_to_mesh import (
    greedy_mesh,
    marching_cube,
    surface_nets,
    dual_contouring,
)


def show_mesh(mesh):
    # Visualize per-face voxel labels when available.
    plotter = pv.Plotter()
    if "voxel_value" in mesh.cell_data:
        plotter.add_mesh(mesh, scalars="voxel_value", show_edges=True, cmap="tab20")
    else:
        plotter.add_mesh(mesh, show_edges=True)
    plotter.show_axes()
    plotter.view_yz()
    plotter.camera.azimuth = 0
    plotter.camera.elevation = 55
    plotter.camera.roll = 0
    plotter.show()


def show_shape(array_3d):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.view_init(elev=40, azim=0, roll=90)

    ax.voxels(array_3d)

    ax.set_box_aspect([1, 1, 1])

    plt.show()


def show_3d_numpy(array_3d: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.voxels(array_3d)
    plt.show()


if __name__ == "__main__":
    voxel = np.load("imgs/faucet.npy")
    print(f"Voxel grid shape: {voxel.shape}")
    # show_shape(voxel)
    mesh = marching_cube(voxel)
    show_mesh(mesh)
