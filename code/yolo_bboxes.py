# Apurva Badithela
# Script that matches YoLo bounding boxes to the ground truth nuscenes boxes

import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from darknet import Darknet
import pdb
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords, get_2d_boxes
import pdb
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os
import random
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box

## Nuscenes dataset:
dataroot="/Users/apurvabadithela/Documents/software/nuscenes/data/sets/nuscenes"
nusc = NuScenes(dataroot="/Users/apurvabadithela/Documents/software/nuscenes/data/sets/nuscenes/")
nms_thresh = 0.6
iou_thresh = 0.4
namesfile = 'data/coco.names'
### Load the COCO object classes
class_names = load_class_names(namesfile)
class YoLo():
    def __init__(self, nms_thresh, iou_thresh):
        self.nms_thresh = nms_thresh
        self.iou_thresh = iou_thresh
        self.m, self.class_names = self.load_yolo_nn()

    ## Loading pre-defined YoLo neural net:
    def load_yolo_nn(self):
        cfg_file = './cfg/yolov3.cfg'
        ### Set the location and name of the pre-trained weights file
        weight_file = 'weights/yolov3.weights'
        ### Set the location and name of the COCO object classes file
        namesfile = 'data/coco.names'
        ### Load the COCO object classes
        class_names = load_class_names(namesfile)
        ### Load the network architecture
        m = Darknet(cfg_file)
        ### Load the pre-trained weights
        m.load_weights(weight_file)
        return m, class_names

    ## Applying the YoLo algorithm:
    def yolo(self, data_path):
        plt.rcParams['figure.figsize'] = [24.0, 14.0]
        img = cv2.imread(data_path) # Successfully reading data
        # Convert the image to RGB
        # pdb.set_trace()
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # We resize the image to the input width and height of the first layer of the network.
        resized_image = cv2.resize(original_image, (self.m.width, self.m.height))
        boxes = detect_objects(self.m, resized_image, self.iou_thresh, self.nms_thresh)
        boxes_pixels_yolo = get_pixel_bounds(original_image, boxes, self.class_names)
        print(original_image.shape[1])
        print(original_image.shape[0])

        print("Modified pixels: ")
        print(self.m.width)
        print(self.m.height)

        # Print the objects found and the confidence level
        print_objects(boxes, self.class_names)
        #Plot the image with bounding boxes and corresponding object class labels
        # plot_boxes(original_image, boxes, self.class_names, plot_labels = True)
        return boxes, boxes_pixels_yolo


# Get a scene corresponding to scene number scene_no from dataset:
class Scenario:
    def __init__(self, scenario_no):
        self.scenario_no = scenario_no
        self.scenario = None
        self.prev_sample = ''
        self.curr_sample = None
        self.next_sample = None
        self.token = None
        self.nbr_samples = None
        self.get_scenario(scene_no)
        self.parse_scene()

    def get_scenario(self):
        N = len(nusc.list_scenes())
        print(".... Loading scene ....")
        if self.scenario_no <= N:
            scenario = nusc.scene[self.scenario_no]
            print(nusc.list_scenes()[self.scenario_no])
        else:
            rand_scenario_no = random.randrange(N)
            scenario = nusc.scene[rand_scenario_no]
            print(nusc.list_scenes()[rand_scenario_no])
        print(scenario)
        self.scenario = scenario

    # Function to parse scene:
    def parse_scene(self):
        self.token = self.scenario['token']
        self.nbr_samples = self.scenario['nbr_samples']
        self.curr_sample = nusc.get('sample',self.scenario['first_sample_token'])
        self.next_sample = nusc.get('sample',self.curr_sample['next'])

    # Function to get sensor data:
    def get_sensor_data(self, sensor):
        sensor_data = nusc.get('sample_data', self.curr_sample['data'][sensor])
        data_path, boxes_nuscenes, camera_intrinsic = nusc.get_sample_data(sensor_data['token'], box_vis_level=BoxVisibility.ANY)
        sd_record = nusc.get('sample_data', sensor_data['token'])
        imsize = (sd_record['width'], sd_record['height'])
        return sensor_data, data_path, boxes_nuscenes, camera_intrinsic, imsize

# Function for plotting 2D boxes in pixels:
def plot_pixel_boxes(box, img, a, plot_labels=True, color = None):
    # Define a tensor used to set the colors of the bounding boxes
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])

    # Define a function to set the colors of the bounding boxes
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))

        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    # Get the width and height of the image
    width = img.shape[1]
    height = img.shape[0]

    # Create a figure and plot the image
    a.imshow(img)

    # Set the default rgb value to red
    rgb = (1, 0, 0)
    x1 = box[0]
    y2 = box[1]

    # Use the same color to plot the bounding boxes of the same object class
    if len(box) >= 7 and class_names:
        cls_conf = box[5]
        cls_id = box[6]
        classes = len(class_names)
        offset = cls_id * 123457 % classes
        red   = get_color(2, offset, classes) / 255
        green = get_color(1, offset, classes) / 255
        blue  = get_color(0, offset, classes) / 255

        # If a color is given then set rgb to the given color instead
        if color is None:
            rgb = (red, green, blue)
        else:
            rgb = color

    # Calculate the width and height of the bounding box relative to the size of the image.
    width_x = box[2]
    width_y = box[3]

    rect = patches.Rectangle((x1, y2),
                                     width_x, width_y,
                                     linewidth = 2,
                                     edgecolor = rgb,
                                     facecolor = 'none')

    # Draw the bounding box on top of the image
    a.add_patch(rect)
    # plt.show()

# Plot 3D nuscenes bounding boxes and 2D YoLo bounding boxes:
def plot_nusc_bboxes_3D(data_path, camera_intrinsic, boxes, imsize, ax=None):
    im = cv2.imread(data_path)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 16))
    # Printing 3D Nuscenes bounding boxes:
    for box in boxes:
        c = np.array(list(nusc.colormap[box.name]))/255.0
        box.render_cv2(im, view=camera_intrinsic, normalize=True, colors=(c, c, c))
        print(box)
    im = cv2.resize(im, imsize)
    ax.imshow(im)


# Plotting 2D Yolo bounding boxes:
def plot_yolo_bboxes_2D(im, ax, boxes_pixels_yolo):
    print("YoLo bounding boxes")
    for ii in range(len(boxes_pixels_yolo)):
        plot_pixel_boxes(boxes_pixels_yolo[ii], im, ax)
    plt.imshow(im)

# Helper functions:
def process_coords(corner_coords,imsize=(1600, 900)):
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])
    # pdb.set_trace()
    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def draw_rect(selected_corners, color, im):
    linewidth = 2
    prev = selected_corners[-1]
    for corner in selected_corners:
        cv2.line(im,
                 (int(prev[0]), int(prev[1])),
                 (int(corner[0]), int(corner[1])),
                 color, linewidth)
        prev = corner

# Get 2D Nuscenes boxes:
def box_nusc(boxes, camera_intrinsic):
    nusc_boxes = []
    for box in boxes:
        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = process_coords(corner_coords)
        nusc_boxes.append(final_coords)
    return nusc_boxes

def parse_box(box):
    xmin = box[0]
    ymin = box[1]
    xmax = box[2] + xmin
    ymax = -box[3] + ymin
    return xmin, ymin, xmax, ymax

def find_intersection(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = parse_box(box1)
    xmin2, ymin2, xmax2, ymax2 = parse_box(box2)
    xmin_int = max(xmin1, xmin2)
    xmax_int = min(xmax1, xmax2)
    ymin_int = max(ymin1, ymin2)
    ymax_int = min(ymax1, ymax2)
    return xmin_int, ymin_int, xmax_int, ymax_int

# Compute iou of two boxes:
def compute_iou(box1, box2):
    '''
    box: [xmin, ymin, wx, wy] where (xmin,ymin) are coordinates of lower left corner and wx = xmax - xmin and wy = ymax-ymin
    Assumption: box1's lower left corner is closer to (0,0)
    '''
    # Get the Width and Height of each bounding box
    xmin1, ymin1, xmax1, ymax1 = parse_box(box1)
    xmin2, ymin2, xmax2, ymax2 = parse_box(box2)
    width_box1 = abs(box1[2])
    height_box1 = abs(box1[3])
    width_box2 = abs(box2[2])
    height_box2 = abs(box2[3])

    # Calculate the area of the each bounding box
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2

    # Find intersecting points
    xmin_int, ymin_int, xmax_int, ymax_int = find_intersection(box1, box2)
    intersection_width = xmax_int - xmin_int
    intersection_height = ymax_int - ymin_int

#     if intersection_width > 0 and intersection_height > 0:
#         pdb.set_trace()
    # If the the boxes don't overlap then their IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_width * intersection_height

    # Calculate the area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate the IOU
    iou = intersection_area/union_area

    return iou

# Preparing YoLo boxes:
def prepare_yolo_box(yolobox):
    yolo_xmin = yolobox[0] # x1
    yolo_ymax = yolobox[1] # y2
    yolo_wx = yolobox[2] # wx = x2-x1
    yolo_wy = yolobox[3]
    yolo_xmax = yolo_wx + yolo_xmin
    yolo_ymin = yolo_wy + yolo_ymax
    yolo_box_new = [yolo_xmin, yolo_ymin, yolo_xmax-yolo_xmin, yolo_ymin-yolo_ymax]
    return yolo_box_new

# Plotting 2D bounding boxes for Nuscenes:
def plot_nusc_bboxes_2D(data_path, camera_intrinsic, boxes, fname_configs, ax=None):
    im = cv2.imread(data_path)
    imsize = (1600,900)

    nusc_boxes = box_nusc(boxes, camera_intrinsic) # Getting all the nuscenes

    # Creating a new axis if none
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 16))

    # Plot bounding box:
    for ii in range(len(nusc_boxes)):
        box_3D = boxes[ii]
        box = nusc_boxes[ii]
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        co = np.array([255.0, 255.0, 0])
        colors = (co,co,co)
        corners = [[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]]
        draw_rect(corners, colors[0][::-1], im)
        # print(box)

    # Plot image on axis
    im = cv2.resize(im, imsize)
    ax.imshow(im)

    # Saving figures in imglib
    cwd = os.getcwd()
    dirname = cwd + "/imglib/scene_"+str(fname_configs['scene_number'])
    # pdb.set_trace()
    if os.path.exists(dirname) is False:
        os.mkdir(dirname)
    fname = dirname + "/gt_sample_no_{0}.pdf".format(fname_configs['sample_number'])
    plt.savefig(fname, dpi='figure', format='pdf')

# Plotting 2D Yolo bounding boxes:
def plot_yolo_bboxes(im, ax, boxes_pixels_yolo):
    print("YoLo bounding boxes")
    for ii in range(len(boxes_pixels_yolo)):
        box_ii = boxes_pixels_yolo[ii]
        xmin = box_ii[0]
        ymax = box_ii[1]
        wx = box_ii[2]
        wy = box_ii[3]
        ymin = ymax + wy
        xmax = xmin + wx
        c = np.array([0.3, 0.3, 0.3])
        colors = (c,c,c)
        corners = [[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]]
        draw_rect(corners, colors[0][::-1], im)
    plt.imshow(im)

# Plot Yolo and Nuscenes boxes together:
def plot_boxes_both(nusc_boxes, boxes_pixels_yolo, impath, data, fname_configs):
    im = cv2.imread(impath)
    sd_record = nusc.get('sample_data', data['token'])
    imsize = (sd_record['width'], sd_record['height'])
    ax = None
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 16))

    # Show image.
    print("YoLo bounding boxes")
    for ii in range(len(boxes_pixels_yolo)):
        yolobox = boxes_pixels_yolo[ii]
        yolo_xmin = yolobox[0] # x1
        yolo_ymin = yolobox[1] # y2
        yolo_wx = yolobox[2] # wx = x2-x1
        yolo_wy = yolobox[3] # wy = y1-y2
        yolo_xmax = yolo_wx + yolo_xmin
        yolo_ymax = -1*yolo_wy + yolo_ymin
        co = np.array([0.0, 0.0, 255.0]) # Red detected
        colors = (co,co,co)
        corners = [[yolo_xmin, yolo_ymax], [yolo_xmin, yolo_ymin], [yolo_xmax, yolo_ymin], [yolo_xmax, yolo_ymax]]
        draw_rect(corners, colors[0][::-1], im)

    # ax.imshow(im)
    for ii in range(len(nusc_boxes)):
        # pdb.set_trace()
        box = nusc_boxes[ii]
        xmin = box[0]
        ymin = box[1]
        wx = box[2] # wx = x2-x1
        wy = box[3]
        xmax = wx + xmin
        ymax = -1*wy + ymin

        co = np.array([255.0, 255.0, 0])  # Blue ground truth
        colors = (co,co,co)
        corners = [[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]]
        draw_rect(corners, colors[0][::-1], im)

    # Render.
    name = "Localization"
    im = cv2.resize(im, imsize)
    ax.imshow(im)

    # Saving figures in imglib
    cwd = os.getcwd()
    dirname = cwd + "/imglib/scene_"+str(fname_configs['scene_number'])
    # pdb.set_trace()
    if os.path.exists(dirname) is False:
        os.mkdir(dirname)
    fname = dirname + "/sample_no_{0}.pdf".format(fname_configs['sample_number'])
    plt.savefig(fname, dpi='figure', format='pdf')
    # plt.imshow(im)
    # print(imsize)
