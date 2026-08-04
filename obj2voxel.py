import trimesh
import numpy as np

# Load your 3D object file
mesh = trimesh.load("imgs/faucet.obj")

voxel_grid = mesh.voxelized(pitch=1.0)

# Fill the interior of the voxelized mesh (optional)
voxel_grid.fill()

voxel_mesh = voxel_grid.as_boxes()

np.save("imgs/faucet.npy", voxel_grid.matrix)
