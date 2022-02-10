#!/usr/bin/env python

import gdown
import path


here = path.Path(__file__).abspath().parent

gdown.cached_download(
    id="16sRRjXoFL0h6G6QGQ66V43DYprKnNMNL",
    path=here / "data/pile_generation.zip",
    md5="1ccd67823c0b07355da215d070255030",
    postprocess=gdown.extractall,
)
