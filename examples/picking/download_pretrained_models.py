#!/usr/bin/env python

import gdown
import path


here = path.Path(__file__).abspath().parent


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

# gdown.cached_download(
#     id="1mcI34DQunVDbc5F4ENhkk1-MRspQEleH",
#     path=here / "logs/20210706_194543-conv_net/weights/91500/q.pth",
#     md5="ebf2e7b874f322fe7f38d0e39375d943",
# )
