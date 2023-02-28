# Apurva Badithela
# June 12, 2022
# More organized version of the proposition labeled confusion matrix

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

class GenPropCM():
    def __init__(self, dataroot=None, traindir="trainval-all", ntrain_dirs =10, save_data_ext=False, chk_stored_scenes=False, horizon=100, discrete_horizon=10, class_dict={0: 'pedestrian', 1:'obstacle', 2:'empty'}, sensors=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'], cm_type="prop_based", cm_fname=None):
        self.cm_fname=cm_fname
        self.traindir = "trainval-all"
        self.dataroot = "/home/apurvabadithela/software/data/sets/nuscenes"
        self.save_data_ext = save_data_ext # parameter to save on external ssd
        self.chk_stored_scenes = chk_stored_scenes # parameter to not resolve for already solved scenes

        self.nusc = NuScenes(dataroot=self.dataroot)
        self.nms_thresh = 0.6
        self.iou_thresh = 0.4
        self.ntrain_dirs = ntrain_dirs
        self.cm_type = cm_type
        self.horizon = horizon
        self.discrete_horizon = discrete_horizon
        self.sensors = sensors # Sensor rleevant in this study
        self.class_dict = class_dict
        self.n = len(list(self.class_dict.keys()))
        self.Nclasses = get_nclasses(self.n, self.cm_type)
        if self.cm_type=="prop_based":
            self.cm_class_dict = self.get_prop_cm_classes()
        else:
            self.cm_class_dict = self.get_class_cm_classes()
        self.classes = [tuple(c) for c in list(self.cm_class_dict.values()) if c!={'empty'}]
        self.rev_class_dict = {v: k for k, v in class_dict.items()}
        self.rev_cm_class_dict = {tuple(v): k for k, v in self.cm_class_dict.items()}
        self.sup_categories = self.construct_class_clusters()

        self.dirname = self.get_dirname()
        self.distance_bins = int(self.horizon/self.discrete_horizon)
        self.C = ConfusionMatrix(self.classes, self.horizon, distance_bins=self.distance_bins)
        self.yolo = YoLo(self.nms_thresh, self.iou_thresh)

    def get_dirname(self):
        if self.save_data_ext:
            dirname = "/home/apurvabadithela/software/cm_processed/" + self.traindir + "/matchings"
        else:
            cwd = os.getcwd()
            dirname = cwd + "/matchings_new"
            # pdb.set_trace()
        if os.path.exists(dirname) is False:
            os.mkdir(dirname)
        return dirname

    def cluster_categories(self, categories):
        sup_categories = dict()
        for category in categories:
            if "human" in category and category not in sup_categories.keys():
                sup_categories[category] = "pedestrian"
            elif "vehicle" in category and category not in sup_categories.keys():
                if "vehicle" in list(self.class_dict.values()):
                    sup_categories[category] = "vehicle"
                else:
                    sup_categories[category] = "obstacle"
            else:
                if category not in sup_categories.keys():
                    sup_categories[category] = "obstacle"
        return sup_categories

    def construct_class_clusters(self):
        categories = []
        for category_dict in self.nusc.category:
            cat_name = category_dict['name']
            if cat_name not in categories:
                categories.append(cat_name)
        sup_categories = self.cluster_categories(categories)
        return sup_categories

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

    def get_yolo_object_type(self, box, class_names):
        # print('Objects Found and Confidence Level:\n')
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            return class_names[cls_id], cls_conf

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

    def prepare_gt_box(self, nubox):
        xmin = nubox[0]
        ymin = nubox[1]
        xmax = nubox[2]
        ymax = nubox[3]
        new_nusc_box = [xmin, ymin, xmax-xmin, ymin-ymax]
        return new_nusc_box

    # Compare YoLo and Nuscenes boxes:
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
        return matchings, matched_nusc_boxes, matched_yolo_boxes

    def process_anns_at_j(self, predictions, gt_val, scene_cmi):
        predicted_prop = set(predictions)
        true_prop = set(gt_val)
        assert true_prop != {'empty'}

        if 'empty' in true_prop:
            true_prop.remove('empty')
        if 'empty' in predicted_prop and predicted_prop != {'empty'}:
            predicted_prop.remove('empty')

        for k, v in self.rev_cm_class_dict.items():
            if predicted_prop == set(k):
                pred_idx = k
            if true_prop == set(k):
                true_idx = k
        row = self.rev_cm_class_dict[pred_idx]
        col = self.rev_cm_class_dict[true_idx]

        # pdb.set_trace()
        scene_cmi[row, col] += 1.0
        return scene_cmi
    # ===========================================================================#
    # Function to construct the confusion matrix from the categorized annotations :
    def construct_confusion_matrix(self, scene_obj):
        discretization = np.linspace(self.discrete_horizon, self.horizon, int(self.horizon/self.discrete_horizon))
        scene_cm = {i: np.zeros((self.Nclasses, self.Nclasses)) for i in range(len(discretization))} # distionary of confusion

        # class_dict = {0: 'pedestrian', 1:'vehicle', 2:'obstacle', 3:'empty'}
        obj_det=False

        # Discretization of the state space
        for i in range(len(discretization)):
            objs_at_dist_i = scene_obj[i]
            if objs_at_dist_i == []: # If no object at that distance return empty
                true_class = tuple({'empty'})
                pred_class = tuple({'empty'})
                row = self.rev_cm_class_dict[pred_class]
                col = self.rev_cm_class_dict[true_class]
                scene_cm[i][row, col] += 1
            else:
                obj_det = True
                predictions =  [obj[1] for obj in objs_at_dist_i]
                gt_val =  [obj[0] for obj in objs_at_dist_i]
                scene_cm[i] = self.process_anns_at_j(predictions, gt_val, scene_cm[i])
        return scene_cm
   #===========================================================================#
    # Scene confusion matrix:
    # scene_cm is an N-by-N matrix where scene_cm(i,j) has the ith location of
    def scene_confusion_matrix(self, distance_to_annotations, matchings, boxes_nusc):
        discretization = np.linspace(self.discrete_horizon, self.horizon, int(self.horizon/self.discrete_horizon))
        scene_obj = {i: [] for i in range(len(discretization))} # distionary of confusion matrix by distance
        # Categorize ground truth annotations based on their distance from ego:
        for j in range(len(boxes_nusc)):
            nubox = boxes_nusc[j] # Box object
            distance_to_ego = distance_to_annotations[nubox.token]
            if any(distance_to_ego < discretization):
                dindx = np.where(distance_to_ego < discretization)[0][0] # First index

                true_class = self.sup_categories[matchings[j]['category']]
                if matchings[j]["yolo_match"]:
                    pred_class = matchings[j]['yolo_match']['pred_class'] # Find the classified matching, otherwise empty
                else:
                    pred_class = 'empty'
                scene_obj[dindx].append((true_class, pred_class)) # Scenario of all objects at a given distance from the ego
        scene_cm = self.construct_confusion_matrix(scene_obj)
        return scene_cm

    # Compute distance to ego
    def compute_distance_to_ego(self, boxes_nusc, cam_data):
        sample_token = cam_data['token']
        sd_record = self.nusc.get('sample_data', sample_token)
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        curr_sample_record = self.nusc.get('sample', sd_record['sample_token'])
        curr_ann_recs = [self.nusc.get('sample_annotation', token) for token in curr_sample_record['anns']]
        cam_front_boxes_nusc = list([box.token for box in boxes_nusc])
        cam_front_anns = [ann_rec for ann_rec in curr_ann_recs if curr_ann_recs in cam_front_boxes_nusc] # only look at front camera annotations:
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

    def fetch_scene(self, n):
        scene = self.nusc.scene[n-1]
        print("Iteration ........ ", str(n))
        sample_token = scene['first_sample_token']
        #pdb.set_trace()
        sample = self.nusc.get('sample', scene['first_sample_token'])
        return scene, sample_token, sample

    def get_bbox_pixels(self, sample, sensor):
        boxes_gt_pixels, boxes_nusc, data_path, cam_data = self.get_gt_boxes(sample, sensor=sensor)
        # Get Yolo Boxes:
        print("Getting YoLo predicted 2D boxes: ")
        boxes_yolo, boxes_yolo_pixels = self.get_yolo_boxes(self.yolo, data_path)

        # Compare bounding boxes:
        matchings, matched_gt_boxes, matched_yolo_boxes = self.compare_boxes(boxes_gt_pixels, boxes_yolo_pixels)

        # compute distance from ego to annotations:
        distance_to_annotations = self.compute_distance_to_ego(boxes_nusc, cam_data)
        return boxes_gt_pixels, boxes_nusc, boxes_yolo_pixels, boxes_yolo, matchings, distance_to_annotations

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

    def get_next_sample(self, sample):
        sample_token = sample['next']
        sample = self.nusc.get('sample', sample_token)
        return sample, sample_token

    def print_cm(self, fname_suff):
        self.C.print_cm()
        if self.save_data_ext:
            dirname = "/home/apurvabadithela/software/cm_processed/" + self.traindir + "/"
        else:
            cwd = os.getcwd()
            dirname = cwd + "/"
        fname = dirname + "cm/" + fname_suff + str(self.horizon) + "_distbin_" + str(self.distance_bins) + ".p"
        pkl.dump(self.C.C, open(fname, "wb"))

    def construct_cm(self):
        Nscenes = len(self.nusc.scene)
        print("Number of scenes: ", self.ntrain_dirs*Nscenes/10)
        # pdb.set_trace()
        for n in range(1,int(self.ntrain_dirs*Nscenes/10)+1): # Uncomment this line
            fname = self.dirname + "/scene_"+str(n)+"_matchings.p"
            if self.chk_stored_scenes and os.path.exists(fname) is True:
                continue
            else:
                objects_detected = dict() # Matchings over objects detected
                scene, sample_token, sample = self.fetch_scene(n)
                sample_number = 1

                while sample_number < scene['nbr_samples']:
                    # Get ground truth 2D boxes for front camera:
                    # Store all matchings at the end
                    objects_detected[sample_number] = dict()
                    print("Getting Nuscenes ground truth 2D boxes: ")
                    for sensor in self.sensors:
                        boxes_gt_pixels, boxes_gt, boxes_yolo_pixels, boxes_yolo, matchings, distance_to_annotations = self.get_bbox_pixels(sample, sensor)

                        matchings = self.process_matchings(boxes_gt, boxes_yolo, matchings, distance_to_annotations)

                        # Construct confusion matrix for this scene:
                        scene_sensor_cm = self.scene_confusion_matrix(distance_to_annotations, matchings, boxes_gt)
                        self.C.add_conf_matrix(scene_sensor_cm)
                        objects_detected[sample_number][sensor] = matchings

                    # Update sample number:
                    sample, sample_token = self.get_next_sample(sample)
                    sample_number += 1
                # Save data:
                pkl.dump(objects_detected, open(fname, "wb"))
        self.print_cm(self.cm_fname) # Print confusion matrix

if __name__ == '__main__':
    example = "full1"
    class_dict = {0: 'pedestrian', 1:'vehicle', 2:'obstacle', 3:'empty'}
    class_dict = {0: 'pedestrian', 1:'obstacle', 2:'empty'}
    sensors = ['CAM_FRONT']

    if example =="mini":
        traindir = None
        cm_type = "prop_based"
        cm_fname = "mini_4prop_cm_ped_obs_cam_f_hz_"
        gen_cm_mini = GenPropCM(traindir=traindir, chk_stored_scenes=False, class_dict=class_dict, sensors=sensors, cm_type = cm_type, cm_fname = cm_fname)
        gen_cm_mini.construct_cm()

    if example =="full":
        cm_type = "class_based"
        cm_fname = "full_3prop_cm_ped_obs_cam_f_hz_"
        gen_cm_full = GenPropCM(save_data_ext=True, chk_stored_scenes=False, class_dict=class_dict, sensors=sensors, cm_type = cm_type, cm_fname = cm_fname)
        gen_cm_full.construct_cm()
    if example =="full1":
        cm_type = "class_based"
        cm_fname = "full1_3prop_cm_ped_obs_cam_f_hz_"
        gen_cm_full = GenPropCM(ntrain_dirs=1, save_data_ext=True, chk_stored_scenes=False, class_dict=class_dict, sensors=sensors, cm_type = cm_type, cm_fname = cm_fname)
        gen_cm_full.construct_cm()
