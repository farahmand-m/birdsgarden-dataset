import argparse
import configparser
import os
import pathlib

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir', type=pathlib.Path)
    parser.add_argument('--predictions', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    assert args.video_dir.exists(), '%s does not exist.' % args.video_dir

    config = configparser.ConfigParser()
    config.read(args.video_dir / 'info.ini')
    sequence = config['Sequence']

    images = [args.video_dir / 'img1' / filepath for filepath in sorted(os.listdir(args.video_dir / 'img1'))]
    columns = ('frame', 'id', 'left', 'top', 'width', 'height', 'confidence', 'X', 'Y', 'Z')
    filepath = os.path.join('pred', 'pred.txt') if args.predictions else os.path.join('gt', 'gt.txt')
    annotations = pd.read_csv(args.video_dir / filepath, names=columns, header=None)

    output_path = str(args.video_dir / 'annotated.mp4')
    encoding = cv2.VideoWriter_fourcc(*'mp4v')
    frame_rate = round(float(sequence['FrameRate']))
    frame_size = (int(sequence['ImWidth']), int(sequence['ImHeight']))
    video = cv2.VideoWriter(output_path, encoding, frame_rate, frame_size)

    for filepath in tqdm(images):
        filepath = str(filepath)
        image_dir, filename = os.path.split(filepath)
        image_name, extension = os.path.splitext(filename)
        frame_no = int(image_name)
        frame = cv2.imread(filepath)
        related = annotations.query(f'frame == {frame_no}')
        for index, row in related.iterrows():
            x, y, w, h = row.left, row.top, row.width, row.height
            color = np.random.default_rng(seed=row.id).random(3) * 255
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.putText(frame, f'#{row.id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f'{frame_no}', (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        if args.show:
            cv2.imshow('Current Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        video.write(frame)

    video.release()

    print('Saved to', output_path)
