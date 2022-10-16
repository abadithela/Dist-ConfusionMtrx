# Script to render video
import os
import pdb
from utils import *
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from shapely.geometry import MultiPoint, box
import pickle as pkl
from yolo_bboxes import *
from parse_matchings import ConfusionMatrix
import numpy as np
import pdb
from gen_video import GenYoloRender
from render_with_yolo import render_scene_yolo
from itertools import chain, combinations
namesfile = 'data/coco.names'
class_names = load_class_names(namesfile)

nusc = NuScenes(version='v1.0-mini', dataroot='/Users/apurvabadithela/Documents/software/nuscenes/data/sets/nuscenes', verbose=True)


for k in range(len(nusc.scene)):
    scene = nusc.scene[k]
    scene_token = scene['token']
    # Render normally
    nusc.render_scene_channel(scene_token, channel='CAM_FRONT', freq=10,imsize=(1280,720), out_path="scene_renders/scene_"+str(k)+".avi")
    # Render with yOlO
    # render_scene_yolo(scene_token=scene_token, channel='CAM_FRONT', freq=2, out_path="yolo_scene_renders/2Hz_scene_"+str(k)+".avi", plot_nusc=True)
    render_scene_yolo(scene_token=scene_token, channel='CAM_FRONT', freq=10, out_path="yolo_only_renders/yolo_only_scene_"+str(k)+".avi", plot_nusc=False)
    # render_scene_yolo(scene_token=scene_token, channel='CAM_FRONT', freq=2, out_path="yolo_scene_renders/2Hz_yolo_3d_nusc_scene_"+str(k)+".avi", plot_nusc=False, plot_nusc_3d=True)
