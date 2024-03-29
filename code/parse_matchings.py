# Apurva Badithela
import pdb
import os
import pickle as pkl
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import numpy as np
import texttable
from tabulate import tabulate
import latextable

class ConfusionMatrix():
    def __init__(self, classes, horizon, distance_bins=5):
        self.distance_bins = distance_bins # Number of different splits within a total of 100 m
        self.horizon = horizon
        self.create_distance_markers() # Upper bound for each confusion matrix
        self.C = dict()
        for i in range(self.distance_bins+1):
            self.C[i] = np.zeros((len(classes)+1, len(classes)+1))

        self.classes = classes
        self.map = dict()
        for k in range(len(classes)):
            self.map[k] = classes[k]
        self.classes.append("empty")
        self.map[len(classes)-1] = ("empty",)
        self.reverse_map = {v: k for k, v in self.map.items()}

    def create_distance_markers(self):
        self.markers = []
        self.gap = self.horizon/self.distance_bins
        dist = 0.0
        while dist < self.horizon:
            dist += self.gap
            self.markers.append(dist)

    def add_conf_matrix(self, new_conf_matrix):
        for key in new_conf_matrix.keys():
            self.C[key] += new_conf_matrix[key]

    def add_prediction(self, distance_bin, true_class, predicted_class):
        predicted_label = self.reverse_map[predicted_class]
        true_label = self.reverse_map[true_class]
        self.C[distance_bin][predicted_label, true_label] += 1

    def find_distance_bin(self, distance_to_ego):
        bin_name = None
        marker = None
        for i in range(1, len(self.markers)):
            if distance_to_ego < self.markers[i] and distance_to_ego >= self.markers[i-1]:
                bin_name = i-1
                marker = self.markers[i-1]

        if distance_to_ego >= self.markers[len(self.markers)-1]:
            bin_name = len(self.markers) - 1
            marker = self.markers[bin_name]
        return bin_name, marker

    def process_detections(self, detection):
        prediction_class = detection[0]
        true_class = detection[1]
        distance_to_ego = detection[2]
        distance_bin, marker = self.find_distance_bin(distance_to_ego)
        # if distance_bin == 5:
        #     pdb.set_trace()
        self.add_prediction(distance_bin, true_class, prediction_class)


    def print_cm(self):
        # TO-DO: make it generic
        for i in range(self.distance_bins):
            print(" ")
            headers=['']
            predictions = []
            for k,v in self.map.items():
                headers.append(v)
                pred_class = v
                pred_row = list(self.C[i][k,:])
                predictions.append([pred_class, *pred_row])

            print("Printing confusion matrix from distance d <= {0}".format(self.markers[i]))

            table_C = tabulate(predictions, headers=headers, tablefmt='latex')
            print(table_C)

# Returns a map to categories of the confusion matrix
def cluster_categories(categories):
    sup_categories = dict()
    for category in categories:
        if "human" in category and category not in sup_categories.keys():
            sup_categories[category] = "pedestrian"
        elif "vehicle" in category and category not in sup_categories.keys():
            sup_categories[category] = "vehicle"
        else:
            if category not in sup_categories.keys():
                sup_categories[category] = "obstacle"
    return sup_categories

def process_objects_detected(C, objects_detected, category_map):
    for sample, sample_matchings in objects_detected.items():
        for gt_box_index in sample_matchings.keys():
            box = sample_matchings[gt_box_index]
            true_class = category_map[box['category']]
            distance_to_ego = box['distance_to_ego']
            # index, marker = C.find_distance_bin(distance_to_ego)
            if box["yolo_match"]:
                predicted_class = box["yolo_match"]["pred_class"]
            else:
                predicted_class = ("empty",)
            detection = [predicted_class, true_class, distance_to_ego]
            C.process_detections(detection) # Adding to confusion matrix


if __name__ == '__main__':
    dataroot="/Users/apurvabadithela/Documents/software/nuscenes/data/sets/nuscenes/"
    dataroot_ext = "/Volumes/Extreme SSD/nuscenes"
    traindir="trainval-all"
    save_data_ext = True
    if save_data_ext:
        nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot_ext)
        dirname = "/Volumes/Extreme SSD/cm_processing/" + traindir + "/matchings"
    else:
        nusc = NuScenes(dataroot=dataroot)
        cwd = os.getcwd()
        dirname = cwd + "/matchings_new"

    categories = []
    for category_dict in nusc.category:
        cat_name = category_dict['name']
        if cat_name not in categories:
            categories.append(cat_name)
    sup_categories = cluster_categories(categories)
    horizon = 100
    classes = ["pedestrian", "obstacle"]
    distance_bins = 10
    
    
    C = ConfusionMatrix(classes, horizon, distance_bins)
    Nscenes = len(nusc.scene)
    N = 85
    for n in range(1,N+1):
        scene = nusc.scene[n-1]
        fname = dirname + "/scene_"+str(n)+"_matchings.p"
        with (open(fname, "rb")) as openfile:
            try:
                objects_detected = pkl.load(openfile)
                process_objects_detected(C, objects_detected, sup_categories)
                print(C.C)
            except EOFError:
                print("Error opening file")
                break
    print("Finished computing confusion matrix")
    pdb.set_trace()
