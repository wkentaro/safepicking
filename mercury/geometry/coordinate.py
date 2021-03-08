import numpy as np

from . import transformations as tf


class Coordinate:
    def __init__(
        self, position=None, euler=None, quaternion=None, matrix=None
    ):
        self._position = np.zeros((3,), dtype=float)
        self._euler = np.zeros((3,), dtype=float)
        self._quaternion = np.array([0, 0, 0, 1], dtype=float)
        self._matrix = np.eye(4, dtype=float)

        if matrix is not None and (
            position is not None or euler is not None or quaternion is not None
        ):
            raise ValueError(
                "position, euler and quaternion must be None "
                "when matrix is given"
            )
        elif euler is not None and quaternion is not None:
            raise ValueError(
                "euler and quaternion cannot be given at the same time"
            )
        if position is not None:
            self.position = position
        if euler is not None:
            self.euler = euler
        if quaternion is not None:
            self.quaternion = quaternion
        if matrix is not None:
            self.matrix = matrix

    def __repr__(self):
        with np.printoptions(precision=3, suppress=True):
            position = np.array2string(self.position, separator=", ")
            euler = np.array2string(self.euler, separator=", ")
            quaternion = np.array2string(self.quaternion, separator=", ")
            return (
                f"{self.__class__.__name__}("
                f"\n  position={position},"
                f"\n  euler={euler},"
                f"\n  quaternion={quaternion},"
                f"\n)"
            )

    def translate(self, translation, wrt="local"):
        self.transform(tf.translation_matrix(translation), wrt=wrt)

    def rotate(self, euler, wrt="local"):
        self.rotation_transform(tf.euler_matrix(euler), wrt=wrt)

    def rotation_transform(self, rotation_transform, wrt="local"):
        rotation_transform = rotation_transform[:3, :3]
        rotation_matrix = self.matrix[:3, :3]
        if wrt == "local":
            rotation_matrix = rotation_matrix @ rotation_transform
        else:
            assert wrt == "world"
            rotation_matrix = rotation_transform @ rotation_matrix
        matrix = tf.translation_matrix(self.position)
        matrix[:3, :3] = rotation_matrix
        self.matrix = matrix

    def transform(self, transform, wrt="local"):
        if wrt == "local":
            self.matrix = self.matrix @ transform
        else:
            assert wrt == "world"
            self.matrix = transform @ self.matrix

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = np.asarray(position)
        self._matrix = tf.transformation_matrix(
            self._quaternion, self._position
        )
        self._freeze()

    @property
    def euler(self):
        return self._euler

    @euler.setter
    def euler(self, euler):
        self._euler = np.asarray(euler)
        self._quaternion = tf.quaternion_from_euler(euler)
        self._matrix = tf.euler_matrix(euler)
        self._matrix[:3, 3] = self._position
        self._freeze()

    @property
    def quaternion(self):
        return self._quaternion

    @quaternion.setter
    def quaternion(self, quaternion):
        self._quaternion = np.asarray(quaternion)
        self._euler = tf.euler_from_quaternion(quaternion)
        self._matrix = tf.transformation_matrix(quaternion, self._position)
        self._freeze()

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix = np.asarray(matrix)
        self._position = tf.translation_from_matrix(matrix)
        self._euler = np.asarray(tf.euler_from_matrix(matrix))
        self._quaternion = tf.quaternion_from_matrix(matrix)
        self._freeze()

    def _freeze(self):
        self._position.setflags(write=False)
        self._euler.setflags(write=False)
        self._quaternion.setflags(write=False)
        self._matrix.setflags(write=False)

    def copy(self):
        return Coordinate(matrix=self.matrix.copy())
