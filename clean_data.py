''' 
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

# globals
INPUT_PATH = "./UIUC_PASCAL_DATA/"
OUTPUT_PATH = "./UIUC_PASCAL_DATA_clean/"

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
	if not fname.endswith(".jpg"):
		continue
	img = cv2.imread(fname)
	img = cv2.resize(img, (img.shape[0], img.shape[0]))
	img = cv2.resize(img, (224,224))
	assert img.shape == (224,224,3), "--- Image is must be 224x224x3---"

	
	new_fname = fname.replace("UIUC_PASCAL_DATA","UIUC_PASCAL_DATA_clean",1) 
	
	cv2.imwrite(new_fname, img)


print ".. done"




