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

# files + path
list_files = []
for root, dirs, files in os.walk("./images/", topdown=False):
    for name in files:
        list_files.append(os.path.join(root, name))

# dump as resized .png files
for fname in list_files:
	img = cv2.imread(fname)
	img = cv2.resize(img, (img.shape[0], img.shape[0]))
	img = cv2.resize(img, (224,224))

	assert img.shape == (224,224,3), "--- Image is must be 224x224x3---"

	new_fname = fname.replace(".jpg", ".png", 1) 
	
	cv2.imwrite(new_fname, img)

# delete old .jpg files
for fname in list_files:
	os.remove(fname)

print ".. done"




