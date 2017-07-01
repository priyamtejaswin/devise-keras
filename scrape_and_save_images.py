"""
Creates two folders: _data & _validation.
Validation contains every nth image from the main set.
"""

import os
import re
import sys
import h5py
import ipdb
from collections import defaultdict, Counter
import string
import pickle
import matplotlib.pyplot as plt
import numpy as np
import shutil

file_path = sys.argv[1]

image_re = re.compile('<td><img src="(.*)\/(.*)"><\/td>')

UIUC_ROOT = "UIUC_PASCAL_DATA"
UIUC_VAL = "UIUC_PASCAL_VAL"
UIUC_URL = "http://vision.cs.uiuc.edu/pascal-sentences"
WORD_DIM = 50

answer = raw_input("DO you want to continue with downloading UIUC_PASCAL stuff?<y/n>")
if answer == "n":
	sys.exit(0)

# UNK_ix = glove_index["<unk>"]

if os.path.exists(UIUC_ROOT):
	print "\nUIUC_PASCAL_DATA detected. The program has stopped here. Press ENTER to continue downloading all data. Press CTRL+C to exit program now.\n"
else:
	os.makedirs(UIUC_ROOT)
if os.path.exists(UIUC_VAL):
	print "\nUIUC_PASCAL_VAL detected. The program has stopped here. Press ENTER to continue downloading all data. Press CTRL+C to exit program now.\n"
else:
	os.makedirs(UIUC_VAL)

print "\nParsing and downloading html source...\n"

_c = 0
id_TO_class = {}
class_TO_images = defaultdict(list)

image_count = -1
uniq_class = set()

with open(file_path, 'r') as fp:
	for line in fp.readlines():
		clean = line.strip()

		match_image = image_re.search(clean)
		if match_image:
			image_count+=1

			class_name = match_image.group(1)
			image_name = match_image.group(2)

			uniq_class.add(class_name)

			dir_name = os.path.join(UIUC_ROOT, class_name)
			if not os.path.exists(dir_name):
				os.makedirs(dir_name)

			img_name = os.path.join(dir_name, image_name)
			img_url = os.path.join(UIUC_URL, class_name, image_name)
			system_string = "wget %s -O %s"%(img_url, img_name)

			if not os.path.exists(img_name):
				os.system(system_string)

			id_TO_class[len(uniq_class) -1] = class_name
			class_TO_images[len(uniq_class) -1].append(image_count)
			

		print image_count
		if ((image_count+1)%(3*50)==0) and (image_count>0):
			print "Downloaded and processed %d images"%(image_count+1)
			_response = raw_input("Download more?<y/n>:")
			if _response=="n":
				break

pickle.dump(id_TO_class, open("DICT_id_TO_class.pkl", "w"))
pickle.dump(class_TO_images, open("DICT_class_TO_images.pkl", "w"))

print "\nFinished downloading all images.\n"
