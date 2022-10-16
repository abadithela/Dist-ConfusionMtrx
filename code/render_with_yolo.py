# Script to render scene with YoLo bounding boxes:
import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable

import cv2
from gen_video import GenYoloRender
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes

from yolo_bboxes import *
from itertools import chain, combinations
namesfile = 'data/coco.names'
class_names = load_class_names(namesfile)
import pdb
from utils import *
from yolo_bboxes import *
from shapely.geometry import MultiPoint, box
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box

def draw_rect(selected_corners, color, im, linewidth=2):
    prev = selected_corners[-1]
    for corner in selected_corners:
        cv2.line(im,
                 (int(prev[0]), int(prev[1])),
                 (int(corner[0]), int(corner[1])),
                 color, linewidth)
        prev = corner

def render_scene_yolo(scene_token: str,
                         channel: str = 'CAM_FRONT',
                         freq: float = 10,
                         imsize: Tuple[float, float] = (1280, 720),
                         out_path: str = None, plot_nusc:bool = True, plot_nusc_3d:bool = False) -> None:
        """
        Renders a full scene for a particular camera channel.
        :param scene_token: Unique identifier of scene to render.
        :param channel: Channel to render.
        :param freq: Display frequency (Hz).
        :param imsize: Size of image to render. The larger the slower this will run.
        :param out_path: Optional path to write a video file of the rendered frames.
        """
        yolo_box_gen = GenYoloRender()
        valid_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                          'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        assert imsize[0] / imsize[1] == 16 / 9, "Error: Aspect ratio should be 16/9."
        assert channel in valid_channels, 'Error: Input channel {} not valid.'.format(channel)

        if out_path is not None:
            assert osp.splitext(out_path)[-1] == '.avi'

        # Get records from DB.
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data'][channel])

        # Open CV init.
        name = '{}: {} (Space to pause, ESC to exit)'.format(scene_rec['name'], channel)
        cv2.namedWindow(name)
        cv2.moveWindow(name, 0, 0)

        if out_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(out_path, fourcc, freq, imsize)
        else:
            out = None

        has_more_frames = True
        while has_more_frames:

            # Get data from DB.
            boxes_gt_pixels, boxes_nusc, impath, camera_intrinsic = yolo_box_gen.get_gt_boxes(sample_rec, sensor=channel)
            # Get Yolo Boxes:
            boxes_yolo, boxes_yolo_pixels = yolo_box_gen.get_yolo_boxes(yolo_box_gen.yolo, impath)

            # Compare bounding boxes:
            matchings, matched_gt_boxes, matched_yolo_boxes,boxes_gt_dict, boxes_yolo_dict = yolo_box_gen.compare_boxes(boxes_gt_pixels, boxes_yolo_pixels)

            # Load and render.
            if not osp.exists(impath):
                raise Exception('Error: Missing image %s' % impath)
            im = cv2.imread(impath)

            if plot_nusc_3d:
                impath, boxes, camera_intrinsic = nusc.get_sample_data(sd_rec['token'],box_vis_level=BoxVisibility.ANY)
                for box in boxes:
                    c = nusc.explorer.get_color(box.name)
                    box.render_cv2(im, view=camera_intrinsic, normalize=True, colors=(c, c, c))
            # Render detected boxes:
            for ii in boxes_yolo_dict.keys():
                yolobox = boxes_yolo_dict[ii]
                yolo_xmin = yolobox[0] # x1
                yolo_ymin = yolobox[1] # y2
                yolo_wx = yolobox[2] # wx = x2-x1
                yolo_wy = yolobox[3] # wy = y1-y2
                yolo_xmax = yolo_wx + yolo_xmin
                yolo_ymax = -1*yolo_wy + yolo_ymin
                co = np.array([0.0, 0.0, 0.0]) # Blue detected/ Yellow ground truth
                colors = (co,co,co)
                corners = [[yolo_xmin, yolo_ymax], [yolo_xmin, yolo_ymin], [yolo_xmax, yolo_ymin], [yolo_xmax, yolo_ymax]]
                draw_rect(corners, colors[0][::-1], im, linewidth=4)

            # Render ground truth boxes:
            # ax.imshow(im)
            if plot_nusc:
                for ii in boxes_gt_dict.keys():
                    # pdb.set_trace()
                    box = boxes_gt_dict[ii]
                    xmin = box[0]
                    ymin = box[1]
                    wx = box[2] # wx = x2-x1
                    wy = box[3]
                    xmax = wx + xmin
                    ymax = -1*wy + ymin

                    co = np.array([225.0, 100.0, 100.0])  # Blue ground truth
                    colors = (co,co,co)
                    corners = [[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]]
                    draw_rect(corners, colors[0][::-1], im)

            # Render.
            im = cv2.resize(im, imsize)

            cv2.imshow(name, im)
            if out_path is not None:
                out.write(im)

            key = cv2.waitKey(10)  # Images stored at approx 10 Hz, so wait 10 ms.
            if key == 32:  # If space is pressed, pause.
                key = cv2.waitKey()

            if key == 27:  # If ESC is pressed, exit.
                cv2.destroyAllWindows()
                break

            if not sample_rec['next'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['next'])
                sample_rec = nusc.get('sample', sample_rec['next'])
            else:
                has_more_frames = False

        cv2.destroyAllWindows()
        if out_path is not None:
            out.release()
