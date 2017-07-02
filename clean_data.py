'''
USAGE: python clean_data.py path/to/folder

Clean up images in /images/ folder 
1. resize to square shape 
2. resize to (224,224,3)
3. 
and check size == (224,224,3)
'''

import numpy as np
import cv2 
import os, sys
from tqdm import *
import ConfigParser

config = ConfigParser.RawConfigParser()
config.read('local.cfg')

train_val = sys.argv[1]
if train_val 	== "TRAIN":
	INPUT_PATH 	 = config.get("data location", "TRAINING_DATA_FOLDER")
	OUTPUT_PATH  = config.get("data location", "TRAINING_DATA_FOLDER_CLEAN")
elif train_val 	== "VALID":
	INPUT_PATH 	 = config.get("data location", "VALIDATION_DATA_FOLDER")
	OUTPUT_PATH  = config.get("data location", "VALIDATION_DATA_FOLDER_CLEAN")
else:
	raise ValueError("Argument must be either VALID/TRAIN")


# globals
# INPUT_PATH = _path.strip()
# if _path[-1]=='/':
# 	OUTPUT_PATH = _path.replace('/', "_clean/")
# else:
# 	OUTPUT_PATH = INPUT_PATH + "_clean/"

# output and input folder name 
INPUT_FOLDER_NAME  = INPUT_PATH.rstrip("/") 
OUTPUT_FOLDER_NAME = OUTPUT_PATH.rstrip("/") 

print "INPUT_PATH: {} | OUTPUT_PATH: {}".format(INPUT_PATH, OUTPUT_PATH)
print "INPUT_FODLER_NAME: {} | OUTPUT_FODLER_NAME: {}".format(INPUT_FOLDER_NAME, OUTPUT_FOLDER_NAME)


if not os.path.exists(OUTPUT_PATH):
	os.mkdir(OUTPUT_PATH)

# files + path
list_files = []
for root, dirs, files in os.walk(INPUT_PATH, topdown=False):
    # print dirs,root
    for name in files:
        list_files.append(os.path.join(root, name))


# all types of classes + make dirs for them  
classes = os.listdir(INPUT_PATH)
print "Classes Found: ", classes
for c in classes:
	os.mkdir(os.path.join(OUTPUT_PATH,c))

# dump as resized .png files
print "Dumping resized (224x224) images to disk.."
for fname in tqdm(list_files):
	if not (fname.endswith(".jpg") or fname.endswith(".png")):
		continue
	img = cv2.imread(fname)
	img = cv2.resize(img, (img.shape[0], img.shape[0]))
	img = cv2.resize(img, (224,224))
	assert img.shape == (224,224,3), "--- Image is must be 224x224x3---"
	
	new_fname = fname.replace(INPUT_FOLDER_NAME,OUTPUT_FOLDER_NAME,1) 
	
	cv2.imwrite(new_fname, img)


print ".. done"




