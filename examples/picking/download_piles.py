#!/usr/bin/env python

import gdown
import path


home = path.Path("~").expanduser()

gdown.cached_download(
    id="16sRRjXoFL0h6G6QGQ66V43DYprKnNMNL",
    path=home / ".cache/safepicking/pile_generation.zip",
    md5="1ccd67823c0b07355da215d070255030",
    postprocess=gdown.extractall,
)
