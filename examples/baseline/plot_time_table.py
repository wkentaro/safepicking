#!/usr/bin/env python

import json

import imgviz
import numpy as np


def main():
    with open("logs/correct/time_table.json") as f:
        time_table = json.load(f)

    titles = []
    for title, _ in time_table:
        if title not in titles:
            titles.append(title)

    rows = []
    header = np.zeros((200, 100, 3), dtype=np.uint8)
    header[...] = 255
    header = header.transpose(1, 0, 2)
    header = imgviz.draw.text_in_rectangle(
        header,
        loc="lt",
        text=f"object_{len(rows)}",
        color=(0, 0, 0),
        size=40,
        background=(255, 255, 255),
    )
    header = header.transpose(1, 0, 2)[:, ::-1, :]
    row = [header]
    for entry in time_table:
        title = entry[0]
        seconds = entry[1]
        width = int(round(seconds * 100))
        if width == 0:
            continue
        viz_i = np.zeros((200, width, 3), dtype=np.uint8)
        color = imgviz.label_colormap()[titles.index(title) + 1]
        viz_i[...] = color
        viz_i = viz_i.transpose(1, 0, 2)
        viz_i = imgviz.draw.text_in_rectangle(
            viz_i,
            loc="lt",
            text=title,
            color=(255, 255, 255),
            size=30,
            background=color,
        )
        viz_i = viz_i.transpose(1, 0, 2)
        viz_i = viz_i[:, ::-1, :]
        row.append(viz_i)
        if title == "correct":
            rows.append(np.hstack(row))
            header = np.zeros((200, 100, 3), dtype=np.uint8)
            header[...] = 255
            header = header.transpose(1, 0, 2)
            header = imgviz.draw.text_in_rectangle(
                header,
                loc="lt",
                text=f"object_{len(rows)}",
                color=(0, 0, 0),
                size=40,
                background=(255, 255, 255),
            )
            header = header.transpose(1, 0, 2)[:, ::-1, :]
            row = [header]
    max_width = max(row.shape[1] for row in rows)
    rows = [
        imgviz.centerize(row, shape=(row.shape[0], max_width), loc="lt")
        for row in rows
    ]
    rows = imgviz.tile(rows, shape=(-1, 1), border=(0, 0, 0))
    imgviz.io.imsave("logs/correct/time_table.jpg", rows)


if __name__ == "__main__":
    main()
