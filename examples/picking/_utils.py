import pybullet as p

import safepicking


def get_class_id(object_id):
    visual_shape_data = p.getVisualShapeData(object_id)
    class_name = visual_shape_data[0][4].decode().split("/")[-2]
    class_id = safepicking.datasets.ycb.class_names.tolist().index(class_name)
    return class_id
