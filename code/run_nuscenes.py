## Apurva Badithela
# File to run nuscenes scenarios
import os
import pdb
from utils import *
dataroot="/Users/apurvabadithela/Documents/software/nuscenes/data/sets/nuscenes"
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from yolo_bboxes import *
from shapely.geometry import MultiPoint, box

nusc = NuScenes(dataroot="/Users/apurvabadithela/Documents/software/nuscenes/data/sets/nuscenes/")
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
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                                selected_anntokens=[anntoken])
        if len(boxes) > 0:
            break  # We found an image that matches. Let's abort.
    assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
                           'Try using e.g. BoxVisibility.ANY.'
    assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    cam = sample_record['data'][cam]
    view = nusc.get('sample_data', cam)['channel']
    if view == 'CAM_FRONT':
        return True
    else:
        return False

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
    for pixel_box in boxes_gt_pixels:
        min_x = pixel_box[0]
        max_y = pixel_box[3]
        width_x = pixel_box[2] - pixel_box[0]
        width_y = pixel_box[3] - pixel_box[1]
        # pdb.set_trace()
        boxes_gt.append([min_x, max_y, width_x, width_y])
    return boxes_gt_pixels, boxes_gt, boxes, data_path_f, cam_front_data_f

# Compare boxes:
def compare_boxes(boxes_gt, boxes_yolo_pixels, boxes_gt_pixels):
    matchings = {}
    boxes_gt_dict = {}
    boxes_yolo_dict = {}
    i = 0
    for box_gt in boxes_gt:
        boxes_gt_dict[i] = box_gt
        matchings[i] = []
        i += 1
    i= 0
    for box_yolo in boxes_yolo_pixels:
        boxes_yolo_dict[i] = box_yolo
        i += 1

    for box_gt_i in boxes_gt_dict.keys():
        for box_yolo_i in boxes_yolo_dict.keys():
            box_gt = boxes_gt_dict[box_gt_i]
            if box_yolo_i == 0:
                print("Found yolo truck")
                if box_gt_i == 3:
                    print("Found correct ground truth prediction")
                    pdb.set_trace()
            box_yolo = boxes_yolo_dict[box_yolo_i]
            box_yolo = prepare_yolo_box(box_yolo)
            iou = compute_iou(box_gt, box_yolo)
            print("GT: ")
            print(box_gt)
            print("YoLo: ")
            print(box_yolo)
            if iou >= 0.6:
                matchings[box_gt_i].append(box_yolo_i)
                print("Match found")
    return matchings

# Iterating over scenes
print(nusc.list_sample)
Nscenes = len(nusc.scene)
# instantiating YoLo obkect:
yolo = YoLo( nms_thresh, iou_thresh)
for i in range(Nscenes):
    scene = nusc.scene[i]
    print("Evaluating scene {0}: ".format(i))
    print(scene)
    # iterating over timestamps / samples:
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', scene['first_sample_token'])
    sample_number = 1
    while sample_number <= scene['nbr_samples']:
        # Get ground truth 2D boxes for front camera:
        print("Getting Nuscenes ground truth 2D boxes: ")
        boxes_gt_pixels, boxes_gt, boxes_nusc, data_path, cam_data = get_gt_boxes(sample)
        # pdb.set_trace()
        # Get Yolo Boxes:
        print("Getting YoLo predicted 2D boxes: ")
        boxes_yolo, boxes_yolo_pixels = get_yolo_boxes(yolo, data_path)

        # Plot boxes on single image:

        if PLOT:
            fname_configs = {'scene_number': i, 'sample_number': sample_number}
            plot_boxes_both(boxes_gt_pixels, boxes_nusc, boxes_yolo, data_path, cam_data, fname_configs)

        # Compare bounding boxes:
        pdb.set_trace()
        matchings = compare_boxes(boxes_gt, boxes_yolo_pixels, boxes_gt_pixels)

        # Update sample number:
        next_sample_token = sample['next']
        sample = nusc.get('sample', next_sample_token)
        sample_number += 1

# Draw 2D boxes:
# data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam_front_data_f['token'], box_vis_level=BoxVisibility.ANY)
# ax = None
# if ax is None:
#     _, ax = plt.subplots(1, 1, figsize=(9, 16))
# im = plot_nusc_bboxes_2D(data_path, camera_intrinsic, boxes, imsize, ax=ax)
