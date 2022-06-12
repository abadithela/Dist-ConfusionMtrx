# Proposition based distance parametrized confusion matrices
# Apurva Badithela
# 6/7/22

import os
import pdb
from utils import *
dataroot="/Users/apurvabadithela/Documents/software/nuscenes/data/sets/nuscenes"
dataroot_ext = "/Volumes/Extreme SSD/nuscenes"
traindir="trainval126"
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from yolo_bboxes import *
from shapely.geometry import MultiPoint, box
import pickle as pkl
from parse_matchings import ConfusionMatrix, cluster_categories
import numpy as np
from itertools import chain, combinations
namesfile = 'data/coco.names'
save_data_ext = True # parameter to save on external ssd
chk_stored_scenes = True # parameter to not resolve for already solved scenes
if save_data_ext:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot_ext)
else:
    nusc = NuScenes(dataroot=dataroot)
# ======================================================================== #
### Load the COCO object classes
class_names = load_class_names(namesfile)
cm_type = "prop_based"
nms_thresh = 0.6
iou_thresh = 0.4
horizon = 120
discrete_horizon = 5
sensor = 'CAM_FRONT' # Sensor rleevant in this study
PLOT = True

# Get number of classes in the confusion matrix:
# Nclasses: num of different objects
# Nprops: num of different confusion matrices
class_dict = {0: 'pedestrian', 1:'obstacle', 2:'empty'}
n = len(list(class_dict.keys()))
def get_nclasses(Nclasses):
    if cm_type == "prop_based":
        return 2**(Nclasses-1)
    else:
        return Nclasses

Nclasses = get_nclasses(n)

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

cm_class_dict = dict()
cm_class_dict[Nclasses-1] = {'empty'} # Building an order
key = 0
empty_key = [k for k,v in class_dict.items() if v == 'empty'][0] # There should only be one empty key
class_keys = list(class_dict.keys())[:-1] # Fix; make better
cm_classes_keys = list(powerset(class_keys))

for k in range(len(cm_classes_keys)):
    class_combinations = cm_classes_keys[k]
    cm_class_dict[k] = {class_dict[class_combinations[0]]}
    for l in range(1, len(class_combinations)):
        cm_class_dict[k] |= {class_dict[class_combinations[l]]}

print(cm_class_dict)
rev_class_dict = {v: k for k, v in class_dict.items()}
rev_cm_class_dict = {tuple(v): k for k, v in cm_class_dict.items()}

print(rev_cm_class_dict)


# Cluster categories from nuscenes dataset:
categories = []
for category_dict in nusc.category:
    cat_name = category_dict['name']
    if cat_name not in categories:
        categories.append(cat_name)
sup_categories = cluster_categories(categories)

# ==================================================================== #
# Collect all instances captured in the first annotation of the object:
def get_yolo_object_type(box, class_names):
    # print('Objects Found and Confidence Level:\n')
    if len(box) >= 7 and class_names:
        cls_conf = box[5]
        cls_id = box[6]
        return class_names[cls_id], cls_conf

# Function to get yOlO Boxes:
def get_yolo_boxes(yolo, data_path):
    boxes, boxes_pixels_yolo = yolo.yolo(data_path)
    return boxes, boxes_pixels_yolo

# function to get nuscenes boxes:
def get_gt_boxes(sample):
    # Get front camera data:
    cam_front_data_f = nusc.get('sample_data', sample['data'][sensor])
    data_path_f, boxes, camera_intrinsic = nusc.get_sample_data(cam_front_data_f['token'], box_vis_level=BoxVisibility.ANY)
    print("Accessed data for ground truth boxes")
    # pdb.set_trace()
    boxes_gt_pixels = box_nusc(boxes, camera_intrinsic)
    boxes_gt = []
    return boxes_gt_pixels, boxes, data_path_f, cam_front_data_f, camera_intrinsic

def prepare_gt_box(nubox):
    xmin = nubox[0]
    ymin = nubox[1]
    xmax = nubox[2]
    ymax = nubox[3]
    new_nusc_box = [xmin, ymin, xmax-xmin, ymin-ymax]
    return new_nusc_box

# ==================================================================== #
# Compare boxes:
# To-DO: Cleanup code
def compare_boxes(boxes_gt, boxes_yolo_pixels):
    matchings = {}
    matched_nusc_boxes = []
    matched_yolo_boxes = []
    boxes_gt_dict = {}
    boxes_yolo_dict = {}
    i = 0
    for box_gt in boxes_gt:
        boxes_gt_dict[i] = prepare_gt_box(box_gt)
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

# ==================================================================== #
# process list of observations seen at a distance j:
# ann_at_j is a list of predictions for the ground truth annotation, gt_ann, at the same distance in a single image

def process_anns_at_j(predictions, gt_val, scene_cmi):
    predicted_prop = set(predictions)
    true_prop = set(gt_val)
    assert true_prop != {'empty'}

    if 'empty' in true_prop:
        true_prop.remove('empty')
    if 'empty' in predicted_prop and predicted_prop != {'empty'}:
        predicted_prop.remove('empty')

    for k, v in rev_cm_class_dict.items():
        if predicted_prop == set(k):
            pred_idx = k
        if true_prop == set(k):
            true_idx = k

    row = rev_cm_class_dict[pred_idx]
    col = rev_cm_class_dict[true_idx]
    # pdb.set_trace()
    scene_cmi[row, col] += 1.0
    return scene_cmi
# ===========================================================================#
# Function to construct the confusion matrix from the categorized annotations :
def construct_confusion_matrix(scene_obj):
    discretization = np.linspace(discrete_horizon, horizon, int(horizon/discrete_horizon))
    scene_cm = {i: np.zeros((Nclasses, Nclasses)) for i in range(len(discretization))} # distionary of confusion

    # class_dict = {0: 'pedestrian', 1:'vehicle', 2:'obstacle', 3:'empty'}
    obj_det=False

    # Discretization of the state space
    for i in range(len(discretization)):
        objs_at_dist_i = scene_obj[i]
        if objs_at_dist_i == []: # If no object at that distance return empty
            true_class = tuple({'empty'})
            pred_class = tuple({'empty'})
            row = rev_cm_class_dict[pred_class]
            col = rev_cm_class_dict[true_class]
            scene_cm[i][row, col] += 1
        else:
            obj_det = True
            predictions =  [obj[1] for obj in objs_at_dist_i]
            gt_val =  [obj[0] for obj in objs_at_dist_i]
            scene_cm[i] = process_anns_at_j(predictions, gt_val, scene_cm[i])
    return scene_cm

# ===========================================================================#
# Scene confusion matrix:
# scene_cm is an N-by-N matrix where scene_cm(i,j) has the ith location of the
def scene_confusion_matrix(distance_to_annotations, matchings, boxes_nusc):
    discretization = np.linspace(discrete_horizon, horizon, int(horizon/discrete_horizon))
    scene_obj = {i: [] for i in range(len(discretization))} # distionary of confusion matrix by distance
    # Categorize ground truth annotations based on their distance from ego:
    for j in range(len(boxes_nusc)):
        nubox = boxes_nusc[j] # Box object
        distance_to_ego = distance_to_annotations[nubox.token]
        if any(distance_to_ego < discretization):
            dindx = np.where(distance_to_ego < discretization)[0][0] # First index
            true_class = sup_categories[matchings[j]['category']]
            if matchings[j]["yolo_match"]:
                pred_class = matchings[j]['yolo_match']['pred_class'] # Find the classified matching, otherwise empty
            else:
                pred_class = 'empty'
            scene_obj[dindx].append((true_class, pred_class)) # Scenario of all objects at a given distance from the ego
    scene_cm = construct_confusion_matrix(scene_obj)
    return scene_cm

# =========================================================================== #
# Compute distance to ego
def compute_distance_to_ego(boxes_nusc, cam_data):
    sample_token = cam_data['token']
    sd_record = nusc.get('sample_data', sample_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    curr_sample_record = nusc.get('sample', sd_record['sample_token'])
    curr_ann_recs = [nusc.get('sample_annotation', token) for token in curr_sample_record['anns']]
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

# =============================================================================#
# Iterating over scenes
# print(nusc.list_sample)
def get_dirname():
    if save_data_ext:
        dirname = "/Volumes/Extreme SSD/cm_processing/" + traindir + "/matchings"
    else:
        cwd = os.getcwd()
        dirname = cwd + "/matchings_new"
        # pdb.set_trace()
    if os.path.exists(dirname) is False:
        os.mkdir(dirname)
    return dirname

if __name__ == '__main__':
    classes = [tuple(c) for c in list(cm_class_dict.values()) if c!={'empty'}]
    print(classes)
    horizon = 100
    discrete_horizon = 10
    distance_bins = int(horizon/discrete_horizon)
    C = ConfusionMatrix(classes, horizon, distance_bins=distance_bins)
    Nscenes = len(nusc.scene)
    dirname=get_dirname()
    # instantiating YoLo obkect:
    yolo = YoLo( nms_thresh, iou_thresh)
    print(Nscenes)
    for n in range(1,Nscenes+1):
        fname = dirname + "/scene_"+str(n)+"_matchings.p"
        if chk_stored_scenes and os.path.exists(fname) is False:
        print("Iteration ........ ", str(n))
        objects_detected = dict() # Matchings over objects detected
        scene = nusc.scene[n-1]
        # iterating over timestamps / samples:
        sample_token = scene['first_sample_token']
        sample = nusc.get('sample', scene['first_sample_token'])
        sample_number = 1

        while sample_number < scene['nbr_samples']:
            # Get ground truth 2D boxes for front camera:
            print("Getting Nuscenes ground truth 2D boxes: ")
            boxes_gt_pixels, boxes_nusc, data_path, cam_data, camera_intrinsic = get_gt_boxes(sample)
            # Get Yolo Boxes:
            print("Getting YoLo predicted 2D boxes: ")
            boxes_yolo, boxes_yolo_pixels = get_yolo_boxes(yolo, data_path)

            # Compare bounding boxes:
            matchings, matched_gt_boxes, matched_yolo_boxes = compare_boxes(boxes_gt_pixels, boxes_yolo_pixels)
            # compute distance from ego to annotations:

            distance_to_annotations = compute_distance_to_ego(boxes_nusc, cam_data)

            # Add annotations:
            for j in range(len(boxes_nusc)):
                nubox = boxes_nusc[j] # Box object
                matchings[j]['nusc_token'] = nubox.token
                matchings[j]['distance_to_ego'] = distance_to_annotations[nubox.token] # Distance to ego
                matchings[j]['category'] = nubox.name

                if matchings[j]["yolo_match"]:
                    yolo_id = matchings[j]["yolo_match"]["box_id"]
                    yolo_box_id = boxes_yolo[yolo_id]
                    if class_names[yolo_box_id[6]] in ["car", "truck", "bus"]:
                        matchings[j]["yolo_match"]["pred_class"] = "obstacle"
                    elif class_names[yolo_box_id[6]] == "person":
                        matchings[j]["yolo_match"]["pred_class"] = "pedestrian"
                    else:
                        matchings[j]["yolo_match"]["pred_class"] = "obstacle"

            # Construct confusion matrix for this scene:
            scene_cm = scene_confusion_matrix(distance_to_annotations, matchings, boxes_nusc)
            C.add_conf_matrix(scene_cm)
            # Store all matchings at the end
            objects_detected[sample_number] = matchings
            # Update sample number:
            sample_token = sample['next']
            sample = nusc.get('sample', sample_token)
            sample_number += 1

        # Save data:
        pkl.dump(objects_detected, open(fname, "wb"))

    # Print confusion matrix:
    C.print_cm()
    if save_data_ext:
        dirname = "/Volumes/Extreme SSD/cm_processing/" + traindir + "/"
    else:
        dirname = cwd + "/"
    fname = dirname + "cm/prop_label_hrzn_cluster_cat" + str(horizon) + "_distbin_" + str(distance_bins) + ".p"
    pkl.dump(C.C, open(fname, "wb"))
    pdb.set_trace()
