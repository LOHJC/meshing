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


def compare_voxel_to_mesh_algos(voxel):
    algorithms = [
        ("greedy_mesh", greedy_mesh),
        ("marching_cube", marching_cube),
        ("surface_nets", surface_nets),
        ("dual_contouring", dual_contouring),
    ]

    plotter = pv.Plotter(shape=(2, 2), window_size=(1600, 1200))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (name, func) in enumerate(algorithms):
        plotter.subplot(i // 2, i % 2)
        try:
            mesh = func(voxel)
            if mesh is None:
                raise ValueError("returned None")
            plotter.add_mesh(
                mesh, color=colors[i], show_edges=True, smooth_shading=True
            )
            plotter.add_title(name, font_size=16)
        except Exception as exc:
            print(f"{name} failed: {exc}")
            plotter.add_text(f"{name}\nfailed", position="upper_left", font_size=14)

        plotter.camera_position = "iso"
        plotter.view_isometric()

        plotter.show_axes()
        plotter.view_yz()
        plotter.camera.azimuth = 0
        plotter.camera.elevation = 55
        plotter.camera.roll = 0
    plotter.show()


if __name__ == "__main__":
    voxel = np.load("imgs/faucet.npy")
    print(f"Voxel grid shape: {voxel.shape}")
    compare_voxel_to_mesh_algos(voxel)
