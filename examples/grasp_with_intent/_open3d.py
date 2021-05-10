import numpy as np
import open3d
import trimesh


class PointCloud(open3d.geometry.PointCloud):
    def __new__(cls, points, colors, normals=None):
        obj = open3d.geometry.PointCloud()
        obj.points = open3d.utility.Vector3dVector(points)
        if colors is not None:
            colors = np.asarray(colors)
            if colors.ndim == 1:
                colors = np.repeat(colors[None, :], points.shape[0], axis=0)
            obj.colors = open3d.utility.Vector3dVector(colors)
        if normals is not None:
            obj.normals = open3d.utility.Vector3dVector(normals)
        return obj

    @classmethod
    def from_trimesh(cls, geometry):
        return cls(
            points=geometry.vertices,
            colors=None
            if geometry.colors.size == 0
            else geometry.colors[:, :3] / 255,
        )


class LineSet(open3d.geometry.LineSet):
    def __new__(cls, points, lines, colors=None):
        obj = open3d.geometry.LineSet()
        obj.points = open3d.utility.Vector3dVector(points)
        obj.lines = open3d.utility.Vector2iVector(lines)
        if colors is not None:
            obj.colors = open3d.utility.Vector3dVector(colors)
        return obj

    @classmethod
    def from_trimesh(cls, geometry):
        points = geometry.vertices
        lines = np.vstack(
            [
                np.c_[entity.points[:-1], entity.points[1:]]
                for entity in geometry.entities
            ]
        )
        return cls(points=points, lines=lines)


class TriangleMesh(open3d.geometry.TriangleMesh):
    @classmethod
    def from_trimesh(cls, geometry):
        obj = open3d.geometry.TriangleMesh()
        obj.vertices = open3d.utility.Vector3dVector(geometry.vertices)
        obj.vertex_normals = open3d.utility.Vector3dVector(
            geometry.vertex_normals
        )
        obj.triangles = open3d.utility.Vector3iVector(geometry.faces)
        if isinstance(geometry.visual, trimesh.visual.ColorVisuals):
            obj.vertex_colors = open3d.utility.Vector3dVector(
                geometry.visual.vertex_colors[:, :3] / 255
            )
        elif isinstance(geometry.visual, trimesh.visual.TextureVisuals):
            obj.triangle_material_ids = open3d.utility.IntVector(
                np.zeros(geometry.faces.size, dtype=np.int32)
            )
            obj.textures = [
                open3d.geometry.Image(
                    np.asarray(geometry.visual.material.image)
                )
            ]
            uv = geometry.visual.uv.copy()
            uv[:, 1] = 1 - uv[:, 1]
            uv = uv[geometry.faces.flatten()]
            obj.triangle_uvs = open3d.utility.Vector2dVector(uv)
        else:
            raise ValueError
        return obj
