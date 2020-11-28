import argparse
from multiprocessing import Pool
import itertools

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from stl import mesh


def visualize_voxel(res: np.ndarray, color: str = "blue") -> None:
    """Visualize voxel data in the specified color.
    This function visualize voxel data. We can specify the voxel color. 
    It's default is blue.
    Args:
        res(np.ndarray) : Boolean matrix of voxel data.
        color(str) : Surface color of visualised voxel data.
    Return:
        None 
    """

    # create colot map
    colors = np.full((res.shape[0], res.shape[1],
                      res.shape[2]), "blue", dtype=str)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(res, facecolors=colors, edgecolor='k')
    #plt.savefig('./voxel.jpg', dpi=140)
    plt.show()


def read_stl(filepath: str):
    """Read stl data from file path.

    This function read stl data from your specified filepath.

    Args: 
        filepath(str): stl data filepath you want to read.

    Return: 
        your_mesh(np.array): np.array of stl data you read.
    """
    try:
        stl_mesh = mesh.Mesh.from_file(filepath)
    except:
        print("Error occur when read stl data.")

    return np.array(stl_mesh)


def expand_area(x):
    """Returns the value it's absolute value is greater but closest with given x.

    Example: If x is 1.2, return is 2.0.
             If x is -2.9, return is -3.0. 
    """
    if x > 0:
        return int(np.ceil(x))
    elif x < 0:
        return int(np.ceil(x) - 1)
    else:
        return 0


def get_cross(plane):
    """Returns cross product of given plane

    Args:
        plane: list of float.
               Example, [1.0, 2.0, 4.0, 2.2, 2.6, 4.6, 0.5, 3.3, 2.7] represents plane through given three points,
               (1.0, 2.0, 4.0), (2.2, 2.6, 4.6) and (0.5, 3.3, 2.7).

    Returns:
        Normal vector of given vector.

    """
    vec_a = plane[0:3]
    vec_b = plane[3:6]
    vec_c = plane[6:9]
    vec_ca = vec_c - vec_a
    vec_ba = vec_b - vec_a
    return np.cross(vec_ca, vec_ba)


def get_plain_equation(plane):
    """Returns plane equation through three points.

    Args: 
        plane: list of float.
               Example, [1.0, 2.0, 4.0, 2.2, 2.6, 4.6, 0.5, 3.3, 2.7] represents plane through given three points,
               (1.0, 2.0, 4.0), (2.2, 2.6, 4.6) and (0.5, 3.3, 2.7).

    Returns:
        List of coefficients [a, b, c, d] of plane equation.
        Example, [a, b, c, d] represents plane equation ax + by + cz + d = 0
    """
    vec_cross = get_cross(plane)
    con = np.dot(- plane[0:3], vec_cross)
    return np.append(vec_cross, con)


def get_polygon_section(polygon):
    """Returns 3D section of given polygon.

    For example, if poligon [(1.0, 2.0, 4.0), (2.2, 2.6, 4.6), (0.5, 3.3, 2.7)] is given, 
    this function returns (0.5, 2.2, 2.0, 3.3, 2.7, 4.6).
    """
    x_min = min(polygon[0::3])
    x_max = max(polygon[0::3])
    y_min = min(polygon[1::3])
    y_max = max(polygon[1::3])
    z_min = min(polygon[2::3])
    z_max = max(polygon[2::3])
    return [x_min, x_max, y_min, y_max, z_min, z_max]


def get_3d_object_section(target_object):
    """Returns 3D section includes given object like stl.
    """

    target_object = target_object.flatten()
    x_min = min(target_object[0::3])
    x_max = max(target_object[0::3])
    y_min = min(target_object[1::3])
    y_max = max(target_object[1::3])
    z_min = min(target_object[2::3])
    z_max = max(target_object[2::3])
    return [x_min, x_max, y_min, y_max, z_min, z_max]


def get_voxel_section(object_section, pitch):
    """Inflate a given range to a range rounded up to an integer value.
    """
    tmp = [item/pitch for item in object_section]
    return [expand_area(item) for item in tmp]


def in_voxel(voxel, plain_equation):
    """Check whether a given plane equation crosses a given voxel.

    This function checks whether a given plane equation crosses a given voxel.

    Voxel has 8 vertices.
    If When the coordinates of the vertices are assigned to the ax+by+cz+d of the plane equation, 
    they take on a value of 0 if they are on the plane and positive or negative otherwise.
    If the voxel crosses the plane, the sign of the assignment should be reversed.

    True if the sign is inverted or the assignment result of any vertex is 0, otherwise false.
    """
    x_start, y_start, z_start, pitch = voxel
    x_end = x_start + pitch
    y_end = y_start + pitch
    z_end = z_start + pitch

    # check poligon in voxel
    result = set()
    for x in (x_start, x_end):
        for y in (y_start, y_end):
            for z in (z_start, z_end):
                point = np.array([x, y, z, 1])
                dot = np.dot(point, plain_equation)
                if dot > 0:
                    result.add(1)
                elif dot < 0:
                    result.add(-1)
                else:
                    result.add(0)
                if len(result) >= 2:
                    return True
    return False


def voxelize(traial):
    data = traial[0]
    object_section = traial[1]
    pitch = traial[2]

    result = []

    plain_equation = get_plain_equation(data)
    polygon_section = get_polygon_section(data)

    for i, x_start in enumerate(np.arange(object_section[0], object_section[1]+pitch, pitch)):
        x_end = x_start + pitch
        if not ((x_start <= polygon_section[1]) and (x_end >= polygon_section[0])):
            continue

        for j, y_start in enumerate(np.arange(object_section[2], object_section[3]+pitch, pitch)):
            y_end = y_start + pitch
            if not ((y_start <= polygon_section[3]) and (y_end >= polygon_section[2])):
                continue

            for k, z_start in enumerate(np.arange(object_section[4], object_section[5]+pitch, pitch)):
                z_end = z_start + pitch
                if not ((z_start <= polygon_section[5]) and (z_end >= polygon_section[4])):
                    continue

                if in_voxel([x_start, y_start, z_start, pitch], plain_equation):
                    result.append(str(i) + "," + str(j) + "," + str(k))
    return result


def main():
    # Argments
    parser = argparse.ArgumentParser(description='Simple voxelizer.')
    parser.add_argument('stl_path', help='Stl filepath.')
    parser.add_argument('pitch', help='Voxel pitch. Float.')
    parser.add_argument('parallel', help='Number of process. Integer.')
    args = parser.parse_args()

    # Get argments
    stl_path = args.stl_path
    pitch = float(args.pitch)
    parallel = int(args.parallel)

    #
    obj_data = read_stl(stl_path)
    object_section = get_3d_object_section(obj_data)
    voxel_section = get_voxel_section(object_section, pitch)

    pool = Pool(processes=parallel)
    traials_per_process = [(porigon, object_section, pitch)
                           for porigon in obj_data]
    voxel_idxs = pool.map(voxelize, traials_per_process)

    voxel_idxs = itertools.chain.from_iterable(voxel_idxs)
    voxel_idxs = list(set(voxel_idxs))

    # create voxel map
    voxel_map = np.full((int((voxel_section[1]-voxel_section[0])+1),
                         int((voxel_section[3]-voxel_section[2])+1),
                         int((voxel_section[5]-voxel_section[4])+1)), 0, dtype=int)
    for idxs in voxel_idxs:
        idxs = idxs.split(",")
        idxs = [int(idx) for idx in idxs]
        x, y, z = idxs
        voxel_map[x][y][z] = 1
    #np.save("./voxcelized", voxel_map)

    visualize_voxel(voxel_map)


if __name__ == "__main__":
    main()
