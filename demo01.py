from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

config = 'C:/Users/User/Desktop/4th_yr_project/pysot-master/experiments/siamrpn_r50_l234_dwxcorr/config.yaml'
snapshot = 'C:/Users/User/Desktop/4th_yr_project/pysot-master/experiments/siamrpn_r50_l234_dwxcorr/model.pth'
video_name = 'C:/Users/User/Desktop/4th_yr_project/pysot-master/demo/car_drift.mp4'


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    print(device)

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(snapshot,
        map_location=lambda storage, loc: storage.cpu()), strict=False)
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if video_name:
        title = video_name.split('/')[-1].split('.')[0]
    else:
        title = 'webcam'
    cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(video_name):
        if first_frame:
            init_rect = cv2.selectROI(title, frame, False, False)
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            cv2.imshow(title, frame)
            cv2.waitKey(40)


if __name__ == '__main__':
    main()