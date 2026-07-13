
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from generate_voxels import show_shape

def load_voxel(filename):
    return np.load(filename)

def show_mesh(mesh):
    plotter = pv.Plotter()
    if "voxel_value" in mesh.cell_data:
        plotter.add_mesh(mesh, scalars="voxel_value", show_edges=True, cmap="tab20")
    else:
        plotter.add_mesh(mesh, show_edges=True)
    plotter.show()

def greedy_mesh(voxel_data):
    pass

def marching_cubes(voxel_data):
    # Placeholder for the marching cubes algorithm
    # This function would convert voxel data to a mesh representation
    pass

if __name__ == "__main__":
    sphere = load_voxel("sphere_2_color.npy")
    show_shape(sphere)
    
    mesh = greedy_mesh(sphere)
    show_mesh(mesh)
    
    # diamond = load_voxel("diamond.npy")
    # show_shape(diamond)
    
    # mesh = greedy_mesh(diamond)
    # show_mesh(mesh)
    