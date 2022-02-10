#!/usr/bin/env python

import gdown
import path


here = path.Path(__file__).abspath().parent


# Raw+Pose
gdown.cached_download(
    id="14NKsz1OmRjFetbAbewDOVI5DgUpUyFUz",
    path=here / "logs/20210709_005731-fusion_net-noise/hparams.json",
    md5="a1f74f06edb50f0f69d0121a3cdf451c",
)
gdown.cached_download(
    id="1O3_hAuSRXuBQvi3sILH0vpYXl1Ti6V3v",
    path=here / "logs/20210709_005731-fusion_net-noise/weights/84500/q.pth",
    md5="886b36a99c5a44b54c513ec7fee4ae0d",
)

# Raw-only
gdown.cached_download(
    id="1dyvfDHSlM5XQrAiUQBT7imTYcjh7CsoK",
    path=here / "logs/20210706_194543-conv_net/hparams.json",
    md5="deccd200613efdafee47361e01112b97",
)
gdown.cached_download(
    id="15KHlNMVVOifAqgRQ7xTpjfN6GqEsfN4D",
    path=here / "logs/20210706_194543-conv_net/weights/91500/q.pth",
    md5="ebf2e7b874f322fe7f38d0e39375d943",
)
