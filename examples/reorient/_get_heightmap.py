import numpy as np


def get_heightmap(points, colors, ids, aabb, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
        points: HxWx3 float array of 3D points in world coordinates.
        colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
        ids: HxW int32 array of values aligned with points.
        defining region in 3D space to generate heightmap in world coordinates.
        pixel_size: float defining size of each pixel in meters.
    Returns:
        heightmap: HxW float array of height (from lower z-bound) in meters.
        colormap: HxWx3 uint8 array of backprojected color aligned with
            heightmap.
    """
    bounds = np.asarray(aabb).T

    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)
    segmmap = np.zeros((height, width), dtype=np.int32)
    pointmap = np.zeros((height, width, 3), dtype=np.float32)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[..., 0] >= bounds[0, 0]) & (points[..., 0] < bounds[0, 1])
    iy = (points[..., 1] >= bounds[1, 0]) & (points[..., 1] < bounds[1, 1])
    iz = (points[..., 2] >= bounds[2, 0]) & (points[..., 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]
    ids = ids[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors, ids = points[iz], colors[iz], ids[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    colormap[py, px] = colors[:]
    segmmap[py, px] = ids[:]
    pointmap[py, px] = points[:]
    return heightmap, colormap, segmmap, pointmap
