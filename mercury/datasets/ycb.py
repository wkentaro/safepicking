import path


def get_visual_file(class_id):
    assert class_id > 0
    root_dir = path.Path("/home/wkentaro/data/ycb_video/YCB_Video_Models")
    class_names = []
    for model_dir in sorted(root_dir.listdir()):
        class_name = str(model_dir.basename())
        class_names.append(class_name)
    class_name = class_names[class_id - 1]
    return root_dir / class_name / "textured_simple.obj"
