''' 
Clean and sort the VOC images into images/ folder. 
'''
import numpy as np 
import os, sys 
from tqdm import *
import shutil
import re
import time
import cv2 

TRAINVAL_FOLDER 	= "./VOCdevkit/VOC2012/ImageSets/Main/"
IMAGE_FOLDER		= "./VOCdevkit/VOC2012/JPEGImages/"
OUTPUT_FOLDER 		= "./voc_images/"

def get_classes_files():
	''' This function returns the classes and filenames associated with classes 
	from the folder VOCdevkit/VOC2012/ImageSets/Main/
	'''
	
	filenames = os.listdir(TRAINVAL_FOLDER)
	fnames_valid = [f for f in filenames if f.endswith("_val.txt")]
	fnames_train = [f for f in filenames if f.endswith("_train.txt")]

	validation = {}

	for txtfile in fnames_valid:
		f = open(os.path.join(TRAINVAL_FOLDER,txtfile),'r')
		data = f.readlines()
		data = [k.strip() for k in data]
		data = [k.split(" ")[0] for k in data if k.split(" ")[-1] == "1"]
		data = [k+".jpg" for k in data]
		data = [os.path.join(IMAGE_FOLDER,k) for k in data]
		f.close()
		classname = txtfile.strip().replace("_val.txt","")
		# print txtfile, classname
		validation[classname] = data

	training = {}
	for txtfile in fnames_train:
		f = open(os.path.join(TRAINVAL_FOLDER,txtfile),'r')
		data = f.readlines()
		data = [k.strip() for k in data]
		data = [k.split(" ")[0] for k in data if k.split(" ")[-1] == "1"]
		data = [k+".jpg" for k in data]
		data = [os.path.join(IMAGE_FOLDER,k) for k in data]
		f.close()
		classname = txtfile.strip().replace("_train.txt","")
		# print txtfile, classname
		training[classname] = data

	return training, validation


def main():

	# get sorted classes + filenames
	training_files, validation_files = get_classes_files()

	# Create directories in voc_images/
	classes = training_files.keys()
	for c in classes:
		os.mkdir(OUTPUT_FOLDER+c)

	# resize + copy training files to voc_images folder 
	for c in classes[0:2]:
		print "Copying images of class - ",c
		time.sleep(5)
		for file_location in tqdm(training_files[c]):
			img = cv2.imread(file_location)
			img = cv2.resize(img, (224,224))
			filename = file_location.split("/")[-1]
			cv2.imwrite(OUTPUT_FOLDER+c+"/"+filename, img)
			




if __name__ == '__main__':
	main()