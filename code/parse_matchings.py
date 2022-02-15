# Apurva Badithela
import pdb
import os
import pickle as pkl
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
nusc = NuScenes(dataroot="/Users/apurvabadithela/Documents/software/nuscenes/data/sets/nuscenes/")
import numpy as np
cwd = os.getcwd()
dirname = cwd + "/matchings_new"
import pdb
class ConfusionMatrix():
    def __init__(self, classes, distance_bins):
        self.distance_bins = distance_bins # Number of different splits within a total of 100 m
        self.horizon = 100.0
        self.create_distance_markers() # Upper bound for each confusion matrix
        self.C = dict()
        for i in range(self.distance_bins):
            self.C[i] = np.zeros(len(classes)+1)

        self.classes = classes
        self.map = dict()
        for k in range(len(classes)):
            self.map[k] = classes[k]
        self.classes.append("no detection")
        self.map[len(classes)] = "empty"
        self.reverse_map = {v: k for k, v in self.map.items()}

    def create_distance_markers(self):
        self.markers = []
        self.gap = self.horizon/self.distance_bins
        dist = 0.0
        while dist < self.horizon:
            dist += self.gap
            self.markers.append(dist)

    def add_prediction(self, distance_bin, true_class, predicted_class):
        predicted_label = self.reverse_map[predicted_class]
        true_label = self.reverse_map[true_class]
        self.C[distance_bin][predicted_label, true_label] += 1

    def find_distance_bin(self, distance_to_ego):
        for i in range(1, len(self.markers)):
            if distance_to_ego < self.markers[i] and distance_to_ego >= self.markers[i-1]
                return i-1, self.markers[i-1]

    def process_detections(self, matchings):
        for detection in matchings:
            prediction_class = detection[0]
            true_class = detection[1]
            distance_to_ego = detection[2]
            distance_bin = self.find_distance_bin(distance_to_ego)
            self.add_prediction(distance_bin, true_class, predicted_class)

    def compute_true_pos(self, class, conf_mat_indx):
        conf_matrix = self.C[conf_mat_indx].copy()

    def compute_true_neg(self, class, conf_mat_indx):
        conf_matrix = self.C[conf_mat_indx].copy()

    def compute_false_neg(self, class, conf_mat_indx):
        conf_matrix = self.C[conf_mat_indx].copy()

    def compute_false_pos(self, class, conf_mat_indx):
        conf_matrix = self.C[conf_mat_indx].copy()

    def print(self):
        for i in range(self.distance_bins):
            print("Printing confusion matrix from distance d <= {0}".format(self.markers[i]))
            print(self.C[i])

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

if __name__ == '__main__':
    categories = nusc.list_categories()
    sup_categories = cluster_categories(categories)

    for n in range(1,Nscenes+1):
        scene = nusc.scene[n-1]
        fname = dirname + "/scene_"+str(n)+"_matchings.p"
        with (open(fname, "rb")) as openfile:
            try:
                objects_detected = pkl.load(openfile)
                pdb.set_trace()
            except EOFError:
                print("Error opening file")
                break
