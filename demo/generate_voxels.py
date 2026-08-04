import numpy as np
import matplotlib.pyplot as plt


def generate_sphere(radius):
    # Determine the bounds of the 3D grid
    size = radius * 2 + 1
    center = radius

    # Initialize an empty 3D grid/array
    voxels = np.zeros((size, size, size), dtype=bool)

    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Calculate squared distance from the center
                dx = x - center
                dy = y - center
                dz = z - center
                dist_squared = dx**2 + dy**2 + dz**2

                # Check if the voxel falls inside the sphere's radius
                if dist_squared <= radius**2:
                    voxels[x, y, z] = True  # Voxel is filled

    return voxels


def generate_diamond(size):
    # Initialize an empty 3D grid/array
    voxels = np.zeros((size, size, size), dtype=bool)

    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (
                    abs(x - size // 2) + abs(y - size // 2) + abs(z - size // 2)
                    <= size // 2
                ):
                    voxels[x, y, z] = True

    return voxels


def show_shape(array_3d):
    """
    Visualizes a 3D boolean numpy array using Matplotlib voxels.
    """
    # Create the 3D figure environment
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # get the unique values
    unique_values = np.unique(array_3d)
    if len(unique_values) > 2:
        # If there are more than 2 unique values, we can use a colormap
        colors = plt.cm.viridis(
            (array_3d - unique_values.min())
            / (unique_values.max() - unique_values.min())
        )

    # Render the 3D voxel grid
    # facecolors adds color, edgecolor outlines each voxel block, alpha adds transparency
    if len(unique_values) > 2:
        ax.voxels(array_3d, facecolors=colors, edgecolor="black", alpha=0.8)
    else:
        ax.voxels(array_3d, facecolors="crimson", edgecolor="black", alpha=0.8)

    # Configure equal scaling so the shape is not distorted
    ax.set_box_aspect([1, 1, 1])

    # Set labels for context
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Voxel Shape Visualization")

    plt.show()


def save_shape(array_3d, filename):
    """
    Saves a 3D boolean numpy array to a .npy file.
    """
    np.save(filename, array_3d)
    print(f"Shape saved to {filename}")


def generate_sphere_2_color(radius, value1, value2):
    # Determine the bounds of the 3D grid
    size = radius * 2 + 1
    center = radius

    # Initialize an empty 3D grid/array
    voxels = np.zeros((size, size, size), dtype=int)

    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Calculate squared distance from the center
                dx = x - center
                dy = y - center
                dz = z - center
                dist_squared = dx**2 + dy**2 + dz**2

                # Check if the voxel falls inside the sphere's radius
                if dist_squared <= radius**2:
                    # Half the sphere with value1, half with value2
                    if x < center:
                        voxels[x, y, z] = value1
                    else:
                        voxels[x, y, z] = value2

    return voxels


def generate_cube(size, value):
    # Initialize an empty 3D grid/array
    voxels = np.zeros((size, size, size), dtype=int)

    # Fill the entire cube with the specified value
    voxels.fill(value)

    return voxels


def generate_cube_2_color(size_full, size_cube, value1, value2):
    # Initialize an empty 3D grid/array
    voxels = np.zeros((size_full, size_full, size_full), dtype=int)

    start = (size_full - size_cube) // 2
    end = start + size_cube
    mid = (start + end) // 2

    for x in range(size_full):
        for y in range(size_full):
            for z in range(size_full):
                if start <= x < end and start <= y < end and start <= z < end:
                    if x < mid:
                        voxels[x, y, z] = value1
                    else:
                        voxels[x, y, z] = value2

    return voxels


if __name__ == "__main__":
    sphere = generate_sphere(3)
    diamond = generate_diamond(5)
    sphere_2_color = generate_sphere_2_color(3, 1, 2)
    cube = generate_cube(5, 1)
    cube_2_color = generate_cube_2_color(10, 5, 1, 2)

    show_shape(sphere)
    show_shape(diamond)
    show_shape(sphere_2_color)
    show_shape(cube)
    show_shape(cube_2_color)

    save_shape(sphere, "sphere.npy")
    save_shape(diamond, "diamond.npy")
    save_shape(sphere_2_color, "sphere_2_color.npy")
    save_shape(cube, "cube.npy")
    save_shape(cube_2_color, "cube_2_color.npy")
