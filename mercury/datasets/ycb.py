import gdown
import path


home = path.Path("~").expanduser()


def get_visual_file(class_id):
    root_dir = home / "data/ycb_video/YCB_Video_Models"

    if not root_dir.exists():
        gdown.cached_download(
            url="https://drive.google.com/uc?id=1PKEJ8SVIVLukvmeIBexp6_XmJetHoOf2",  # NOQA
            path=root_dir + ".zip",
            md5="540c37435e4a16546850a83690b2db9b",
            postprocess=gdown.extractall,
        )

    assert class_id > 0
    class_names = []
    for model_dir in sorted(root_dir.listdir()):
        class_name = str(model_dir.basename())
        class_names.append(class_name)
    class_name = class_names[class_id - 1]
    return root_dir / class_name / "textured_simple.obj"
