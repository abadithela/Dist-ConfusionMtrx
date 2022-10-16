import os
import pdb
from utils import *
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from yolo_bboxes import *
from shapely.geometry import MultiPoint, box
import pickle as pkl
from parse_matchings import ConfusionMatrix
import numpy as np
from itertools import chain, combinations
namesfile = 'data/coco.names'
class_names = load_class_names(namesfile)
PLOT = False

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def get_nclasses(Nclasses, cm_type):
    if cm_type == "prop_based":
        return 2**(Nclasses-1)
    else:
        return Nclasses

class GenYoloRender():
    def __init__(self, class_dict={0: 'pedestrian', 1:'obstacle', 2:'empty'}, sensors=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']):

        self.nusc = NuScenes(version='v1.0-mini', dataroot='/Users/apurvabadithela/Documents/software/nuscenes/data/sets/nuscenes', verbose=True)
        self.nms_thresh = 0.6
        self.iou_thresh = 0.4
        self.sensors = sensors # Sensor rleevant in this study
        self.class_dict = class_dict
        self.n = len(list(self.class_dict.keys()))
        self.cm_type = "prop_based"
        self.Nclasses = get_nclasses(self.n, self.cm_type)
        # if self.cm_type=="prop_based":
        #     self.cm_class_dict = self.get_prop_cm_classes()
        # else:
        #     self.cm_class_dict = self.get_class_cm_classes()
        # self.classes = [tuple(c) for c in list(self.cm_class_dict.values()) if c!={'empty'}]
        # self.rev_class_dict = {v: k for k, v in class_dict.items()}
        # self.rev_cm_class_dict = {tuple(v): k for k, v in self.cm_class_dict.items()}
        self.yolo = YoLo(self.nms_thresh, self.iou_thresh)

    def get_class_cm_classes(self):
        cm_class_dict = dict()
        cm_class_dict[self.Nclasses-1] = {'empty'} # Building an order
        key = 0
        empty_key = [k for k,v in self.class_dict.items() if v == 'empty'][0]
        for k,v in self.class_dict.items():
            cm_class_dict[k] = {v}
        return cm_class_dict

    def get_prop_cm_classes(self):
        cm_class_dict = dict()
        cm_class_dict[self.Nclasses-1] = {'empty'} # Building an order
        key = 0
        empty_key = [k for k,v in self.class_dict.items() if v == 'empty'][0] # There should only be one empty key
        class_keys = list(self.class_dict.keys())[:-1] # Fix; make better
        cm_classes_keys = list(powerset(class_keys))

        for k in range(len(cm_classes_keys)):
            class_combinations = cm_classes_keys[k]
            cm_class_dict[k] = {class_dict[class_combinations[0]]}
            for l in range(1, len(class_combinations)):
                cm_class_dict[k] |= {class_dict[class_combinations[l]]}
        return cm_class_dict

    # Function to get yOlO Boxes:
    def get_yolo_boxes(self, yolo, data_path):
        boxes, boxes_pixels_yolo = yolo.yolo(data_path)
        return boxes, boxes_pixels_yolo

    # function to get nuscenes boxes:
    def get_gt_boxes(self, sample, sensor): # Get ground truth boxes for each cam_front sensor
        # Get front camera data:
        cam_front_data_f = self.nusc.get('sample_data', sample['data'][sensor])
        data_path_f, boxes, camera_intrinsic = self.nusc.get_sample_data(cam_front_data_f['token'], box_vis_level=BoxVisibility.ANY)
        print("Accessed data for ground truth boxes")
        boxes_gt_pixels = box_nusc(boxes, camera_intrinsic)
        boxes_gt = []
        return boxes_gt_pixels, boxes, data_path_f, cam_front_data_f

    def get_bbox_pixels(self, sample, sensor):
        boxes_gt_pixels, boxes_nusc, data_path, cam_data = self.get_gt_boxes(self, sample, sensor=sensor)
        # Get Yolo Boxes:
        print("Getting YoLo predicted 2D boxes: ")
        boxes_yolo, boxes_yolo_pixels = self.get_yolo_boxes(self.yolo, data_path)

        # Compare bounding boxes:
        matchings, matched_gt_boxes, matched_yolo_boxes = self.compare_boxes(boxes_gt_pixels, boxes_yolo_pixels)

        # compute distance from ego to annotations:
        distance_to_annotations = self.compute_distance_to_ego(boxes_nusc, cam_data)
        return matched_gt_boxes, matched_yolo_boxes, data_path, cam_data

    def prepare_gt_box(self, nubox):
        xmin = nubox[0]
        ymin = nubox[1]
        xmax = nubox[2]
        ymax = nubox[3]
        new_nusc_box = [xmin, ymin, xmax-xmin, ymin-ymax]
        return new_nusc_box
    def compute_distance_to_ego(boxes_nusc, cam_data):
        sample_token = cam_data['token']
        sd_record = nusc.get('sample_data', sample_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        curr_sample_record = nusc.get('sample', sd_record['sample_token'])
        curr_ann_recs = [nusc.get('sample_annotation', token) for token in curr_sample_record['anns']]
        cam_front_boxes_nusc = list([box.token for box in boxes_nusc])
        cam_front_anns = [ann_rec for ann_rec in curr_ann_recs if curr_ann_recs in cam_front_boxes_nusc]# only look at front camera annotations:
        # Sanity check:
        assert cam_front_anns!=None

        # Move box to ego vehicle coordinate system from sensor coordinate system:
        distance_to_ego = dict()
        boxes_copy = boxes_nusc.copy()
        for box in boxes_copy:
            box.rotate(Quaternion(cs_record['rotation']))
            box.translate(np.array(cs_record['translation']))
            box_pos = box.center
            distance_to_ego[box.token] = np.linalg.norm(box_pos)
        return distance_to_ego

    def process_matchings(self, boxes_gt, boxes_yolo, matchings, distance_to_annotations):
        '''
        Process matchings for YoLo
        '''
        for j in range(len(boxes_gt)):
            nubox = boxes_gt[j] # Box object
            matchings[j]['nusc_token'] = nubox.token
            matchings[j]['distance_to_ego'] = distance_to_annotations[nubox.token] # Distance to ego
            matchings[j]['category'] = nubox.name

            if matchings[j]["yolo_match"]:
                yolo_id = matchings[j]["yolo_match"]["box_id"]
                yolo_box_id = boxes_yolo[yolo_id]
                if class_names[yolo_box_id[6]] in ["car", "truck", "bus"]:
                    if "vehicle" in list(self.class_dict.values()):
                        matchings[j]["yolo_match"]["pred_class"] = "vehicle"
                    else:
                        matchings[j]["yolo_match"]["pred_class"] = "obstacle"

                elif class_names[yolo_box_id[6]] == "person":
                    matchings[j]["yolo_match"]["pred_class"] = "pedestrian"
                else:
                    matchings[j]["yolo_match"]["pred_class"] = "obstacle"
        return matchings
    def compare_boxes(self, boxes_gt, boxes_yolo_pixels):
        matchings = {}
        matched_nusc_boxes = []
        matched_yolo_boxes = []
        boxes_gt_dict = {}
        boxes_yolo_dict = {}
        i = 0
        for box_gt in boxes_gt:
            boxes_gt_dict[i] = self.prepare_gt_box(box_gt)
            matchings[i] = dict()
            matchings[i]["yolo_match"] = dict()
            i += 1
        i= 0
        for box_yolo in boxes_yolo_pixels:
            boxes_yolo_dict[i] = prepare_yolo_box(box_yolo)
            i += 1

        for box_gt_i in boxes_gt_dict.keys():
            for box_yolo_i in boxes_yolo_dict.keys():
                box_gt = boxes_gt_dict[box_gt_i]
                box_yolo = boxes_yolo_dict[box_yolo_i]
                iou = compute_iou(box_gt, box_yolo)

                if iou >= 0.7:
                    matchings[box_gt_i]["yolo_match"]["box_id"] = box_yolo_i
                    matched_nusc_boxes.append(boxes_gt_dict[box_gt_i])
                    matched_yolo_boxes.append(boxes_yolo_dict[box_yolo_i])
                    print("Match found")
        return matchings, matched_nusc_boxes, matched_yolo_boxes, boxes_gt_dict, boxes_yolo_dict

    def fetch_scene(self, n):
        scene = self.nusc.scene[n-1]
        print("Iteration ........ ", str(n))
        sample_token = scene['first_sample_token']
        #pdb.set_trace()
        sample = self.nusc.get('sample', scene['first_sample_token'])
        return scene, sample_token, sample

    def get_next_sample(self, sample):
        sample_token = sample['next']
        sample = self.nusc.get('sample', sample_token)
        return sample, sample_token
