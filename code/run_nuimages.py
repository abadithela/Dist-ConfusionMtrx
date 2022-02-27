## Apurva Badithela
# File to run nuimenes scenarios
import os
import pdb
from utils import *
dataroot="/Users/apurvabadithela/Documents/software/nuimenes/data/sets/nuimenes"
from nuimages.nuimages import NuImages
from yolo_bboxes_nuimages import *
from shapely.geometry import MultiPoint, box
import pickle as pkl
import numpy as np
nuim = NuImages(dataroot="/Users/apurvabadithela/Documents/software/nuimages/data/sets/nuimages/")
namesfile = 'data/coco.names'
### Load the COCO object classes
class_names = load_class_names(namesfile)
nms_thresh = 0.6
iou_thresh = 0.4

sensor = 'CAM_FRONT' # Sensor rleevant in this study
PLOT = True

# Helper function only to read front camera annotations:
# Get annotations only on camera front:
def is_cam_front_ann_token(anntoken: str,
                      margin: float = 10,
                      view: np.ndarray = np.eye(4),
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      out_path: str = None,
                      extra_info: bool = False) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nuim.get('sample_annotation', anntoken)
    sample_record = nuim.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    for cam in cams:
        _, boxes, _ = nuim.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                                selected_anntokens=[anntoken])
        if len(boxes) > 0:
            break  # We found an image that matches. Let's abort.
    assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
                           'Try using e.g. BoxVisibility.ANY.'
    assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    cam = sample_record['data'][cam]
    view = nuim.get('sample_data', cam)['channel']
    if view == 'CAM_FRONT':
        return True
    else:
        return False

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

# function to get nuimenes boxes:
def get_gt_boxes(sample):
    # Get front camera data:
    object_tokens, surface_tokens = nuim.list_anns(sample['token'])
    im_path = osp.join(self.dataroot, sample_data['filename'])
    boxes_gt_pixels = []
    for ann in object_tokens:
        category_token = ann['category_token']
        category_name = nuim.get('category', category_token)['name']
        color = nuim.color_map[category_name]
        bbox = ann['bbox']

    # pdb.set_trace()
    boxes_gt_pixels = box_nuim(boxes, camera_intrinsic)
    return boxes_gt_pixels, boxes, im_path,

def prepare_gt_box(nubox):
    xmin = nubox[0]
    ymin = nubox[1]
    xmax = nubox[2]
    ymax = nubox[3]
    new_nuim_box = [xmin, ymin, xmax-xmin, ymin-ymax]
    return new_nuim_box

# Compare boxes:
def compare_boxes(boxes_gt, boxes_yolo_pixels):
    matchings = {}
    matched_nuim_boxes = []
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
            # box_yolo = prepare_yolo_box(box_yolo)
            # box_gt = prepare_gt_box(box_gt)
            iou = compute_iou(box_gt, box_yolo)

            if iou >= 0.7:
                matchings[box_gt_i]["yolo_match"]["box_id"] = box_yolo_i
                matched_nuim_boxes.append(boxes_gt_dict[box_gt_i])
                matched_yolo_boxes.append(boxes_yolo_dict[box_yolo_i])
                #matchings[box_gt_i]["yolo_match"]["pred_category"] = get_yolo_object_type(boxes_yolo_pixels[box_yolo_i])
                print("Match found")
    return matchings, matched_nuim_boxes, matched_yolo_boxes

# Compute distance to ego
def compute_distance_to_ego(boxes_nuim, cam_data):
    sample_token = cam_data['token']
    sd_record = nuim.get('sample_data', sample_token)
    cs_record = nuim.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    curr_sample_record = nuim.get('sample', sd_record['sample_token'])
    curr_ann_recs = [nuim.get('sample_annotation', token) for token in curr_sample_record['anns']]
    cam_front_boxes_nuim = list([box.token for box in boxes_nuim])
    cam_front_anns = [ann_rec for ann_rec in curr_ann_recs if curr_ann_recs in cam_front_boxes_nuim]# only look at front camera annotations:
    # Sanity check:
    assert cam_front_anns!=None

    # Move box to ego vehicle coordinate system from sensor coordinate system:
    distance_to_ego = dict()
    boxes_copy = boxes_nuim.copy()
    for box in boxes_copy:
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        box_pos = box.center
        distance_to_ego[box.token] = np.linalg.norm(box_pos)
    return distance_to_ego

# Iterating over images
# print(nuim.list_sample)
Nimages = len(nuim.sample)
# instantiating YoLo obkect:
yolo = YoLo( nms_thresh, iou_thresh)
print(Nimages)
for n in range(1,Nimages+1):
    print("Iteration ........ ", str(n))
    objects_detected = dict() # Matchings over objects detected
    image = nuim.sample[n-1]
    # print(image)
    # iterating over timestamps / samples:
    sample_token = image['token']
    sample = nuim.get('sample', sample['token'])
    sample_number = 1

    key_camera_token = sample['key_camera_token']
    sample_data = nuim.get("sample_data", sample['key_camera_token'])
    print(sample_data)
    if sample_data['is_key_frame']:  # Only use keyframes (samples).
        sensor = nuim.shortcut('sample_data', 'sensor', sample_data['token'])
        if sensor['channel'] == 'CAM_FRONT':
            # Get ground truth 2D boxes for front camera:
            print("Getting NuImages ground truth 2D boxes: ")
            boxes_gt_pixels, boxes_nuim, data_path, cam_data, camera_intrinsic = get_gt_boxes(sample)
            # pdb.set_trace()
            # Get Yolo Boxes:
            print("Getting YoLo predicted 2D boxes: ")
            boxes_yolo, boxes_yolo_pixels = get_yolo_boxes(yolo, data_path)

            # Compare bounding boxes:
            matchings, matched_gt_boxes, matched_yolo_boxes = compare_boxes(boxes_gt_pixels, boxes_yolo_pixels)

            # compute distance from ego to annotations:

            # Plot boxes on single image:
            # plot only matched boxes
            if PLOT:
                # pdb.set_trace()
                # Plot matched boxes
                fname_configs = {'image_number': n, 'sample_number': sample_number}
                plot_boxes_both(matched_gt_boxes, matched_yolo_boxes, data_path, cam_data, fname_configs)

                # Plot all 2D nuimenes boxes:
                plot_nuim_bboxes_2D(data_path, camera_intrinsic, boxes_nuim, fname_configs)

            # Distance to ego:
            distance_to_annotations = compute_distance_to_ego(boxes_nuim, cam_data)

            # Add annotations:
            for j in range(len(boxes_nuim)):
                nubox = boxes_nuim[j] # Box object
                matchings[j]['nuim_token'] = nubox.token
                matchings[j]['distance_to_ego'] = distance_to_annotations[nubox.token] # Distance to ego
                matchings[j]['category'] = nubox.name

                if matchings[j]["yolo_match"]:
                    yolo_id = matchings[j]["yolo_match"]["box_id"]
                    yolo_box_id = boxes_yolo[yolo_id]
                    if class_names[yolo_box_id[6]] in ["car", "truck", "bus"]:
                        matchings[j]["yolo_match"]["pred_class"] = "vehicle"
                    elif class_names[yolo_box_id[6]] == "person":
                        matchings[j]["yolo_match"]["pred_class"] = "pedestrian"
                    else:
                        matchings[j]["yolo_match"]["pred_class"] = "obstacle"

            # Store all matchings at the end
            objects_detected[sample_number] = matchings

            # pdb.set_trace()
            # Save data:
            cwd = os.getcwd()
            dirname = cwd + "/nuim_matchings_new"
            # pdb.set_trace()
            if os.path.exists(dirname) is False:
                os.mkdir(dirname)
            fname = dirname + "/image_"+str(n)+"_matchings.p"
            pkl.dump(objects_detected, open(fname, "wb"))

# Draw 2D boxes:
# data_path, boxes, camera_intrinsic = nuim.get_sample_data(cam_front_data_f['token'], box_vis_level=BoxVisibility.ANY)
# ax = None
# if ax is None:
#     _, ax = plt.subplots(1, 1, figsize=(9, 16))
# im = plot_nuim_bboxes_2D(data_path, camera_intrinsic, boxes, imsize, ax=ax)
