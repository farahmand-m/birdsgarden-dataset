import argparse
import configparser
import json
import os
import pathlib
import shutil
from xml.etree import ElementTree

import cv2
import numpy as np
import tqdm.auto as tqdm


def patch_bounds(width, height, patch_size):
    x_step, y_step = patch_size
    for x_min in range(0, width, x_step):
        for y_min in range(0, height, y_step):
            yield x_min, x_min + x_step, y_min, y_min + y_step


def frames(capture):
    read, frame = capture.read()
    while read:
        yield frame
        read, frame = capture.read()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='dataset', help='path to directory containing raw_data')
    parser.add_argument('--patch-size', nargs=2, type=int, default=(640, 480), help='desired patch size')
    parser.add_argument('--target-videos', default=None, help='if set, only videos containing the word are processed')
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    source_dir = data_dir / 'raw_data'
    target_dir = data_dir / 'patched-raw-data'

    p_width, p_height = args.patch_size

    # noinspection PyTypeChecker
    directories = {split: os.listdir(source_dir / split) for split in os.listdir(source_dir)}

    for subset, videos in directories.items():

        for video in videos:

            if args.target_videos is not None:
                if args.target_videos not in video:
                    continue

            filepath = source_dir / subset / video / 'video.mp4'
            capture = cv2.VideoCapture(str(filepath))

            frame_rate = int(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            *bounds, = patch_bounds(width, height, args.patch_size)

            dir_paths = [target_dir / subset / f'{video}_{index+1:02d}' for index, bound in enumerate(bounds)]
            for dir_path in dir_paths:
                os.makedirs(dir_path, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writers = [cv2.VideoWriter(f'{dir_path}/video.mp4', fourcc, frame_rate, args.patch_size)
                       for bound, dir_path in zip(bounds, dir_paths)]

            for frame in tqdm.tqdm(frames(capture), total=num_frames, desc=video):

                for (x_min, x_max, y_min, y_max), writer in zip(bounds, writers):

                    patch = frame[y_min: y_max, x_min: x_max]
                    height, width, channels = patch.shape
                    h_pad = p_height - height
                    v_pad = p_width - width
                    padding = [(0, h_pad), (0, v_pad), (0, 0)]
                    if np.any(padding):
                        patch = np.pad(patch, padding)
                    writer.write(patch)

            for writer in writers:
                writer.release()

            for bound, dir_path in zip(bounds, dir_paths):
                shutil.copy(source_dir / subset / video / 'gt.xml', dir_path / 'gt.xml')
                with open(dir_path / 'info.json', 'w') as stream:
                    json.dump({'bound': bound}, stream)
