#!/usr/local/bin/python2.7
#-*- encoding:utf-8 -*-
import argparse as ap
import cv2
import imutils 
import numpy as np
import os
import csv
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

from Queue import Queue
from threading import Thread

class FeaturesWorker(Thread):
    def __init__(self, queue, output_list):
        Thread.__init__(self)
        self.queue = queue
        self.output_list = output_list
        
        # Create feature extraction and keypoint detector objects
        self.fea_det = cv2.FeatureDetector_create("SIFT")
        self.des_ext = cv2.DescriptorExtractor_create("SIFT")

    def run(self):
        while True:
            image_path = self.queue.get()
            im = cv2.imread(image_path)
            kpts = self.fea_det.detect(im)
            kpts, des = self.des_ext.compute(im, kpts)
            del im
            self.output_list.append((image_path, des))  
            
            self.queue.task_done()

flatten = lambda l : [y for x in l for y in x]

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
parser.add_argument("-c", "--csv")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]

if args['csv']:
    with open(args['csv'], 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        csvcontent = [[unicode(cell, 'utf-8') for cell in row] for row in r]
    image_paths, image_classes_full = zip(*[(os.path.join(train_path, f[0]), f[1]) for f in csvcontent])
    training_names = list(set(image_classes_full))
    image_classes = [training_names.index(i) for i in image_classes_full]
else:
    training_names = os.listdir(train_path)

    # Get all the path to the images and save them in a list
    # image_paths and the corresponding label in image_paths
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
        class_path = imutils.imlist(dir)
        image_paths+=class_path
        image_classes+=[class_id]*len(class_path)
        class_id+=1

# List of lists where all the descriptors are stored
des_lists = list()

# Parallelized feature extraction
queue = Queue()
for i in image_paths:
    queue.put(i)

for x in range(4):
    des_lists.append(list())
    worker = FeaturesWorker(queue, des_lists[x])
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    worker.daemon = True
    worker.start()

print "Workers created, processing…"

queue.join()
# Merging lists
des_list = flatten(des_lists)
del des_lists

print "Descriptor extraction done, creating dictionary"
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]

for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

# Perform k-means clustering
k = 500
voc, variance = kmeans(descriptors, k, 1) 

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Train the Linear SVM
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Save the SVM
joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)    
    
