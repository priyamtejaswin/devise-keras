import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input 
import keras.backend as K
import h5py
import argparse 
import os, sys

def define_model(path):

	input_shape = (3,224,224)

	# placeholder - input image tensor
	img_input = Input(shape=input_shape)

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)

	model = Model(inputs=img_input, outputs=x, name="vgg16")


	# load wts
	# model.load_weights(path, by_name=True)

	return model  

def create_indices(total_length, batch_size):
	return izip(xrange(0, total_length, batch_size), xrange(batch_size, total_length+batch_size, batch_size))


def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-weights_path", help="weights file path")
	parser.add_argument("-folder_path", help="folder where images are located")
	parser.add_argument("-dump_path", help="folder where features will be dumped")
	args = parser.parse_args()

	weights_path 	= args.weights_path
	folder_path 	= args.folder_path
	dump_path   	= args.dump_path

	assert os.path.isdir(folder_path), "---path is not a folder--"
	assert os.path.isdir(dump_path), "---path is not a folder--"
	
	model = define_model(weights_path)
	
	dir_fnames = []
	for dirpath, dirnames, filenames in os.walk(folder_path):
		if filenames != []:
			dir_fnames += [os.path.join(dirpath, fn) for fn in filenames]
	list_of_files = dir_fnames

	# h5py 
	hf = h5py.File(os.path.join(dump_path,"features.h5"),"w")
	data = hf.create_group("data")
	x_h5 = data.create_dataset("features",(0,4096), maxshape=(None,4096))
	fnames_h5 = data.create_dataset("fnames",(0,1),"S10", maxshape=(None,1))

	for i,j in create_indices(len(list_of_files), 2):
		loaded_images = []
		dump_name = "processed_files"

		for k in range(i,j,1):
			dump_name += "_"+dir_fnames[k]

			img = image.load_img(list_of_files[k], target_size=(224, 224))
			img = image.img_to_array(img)
			loaded_images.append(img)

		loaded_images = np.array(loaded_images)
		batch = preprocess_input(loaded_images)
		
		scores = model.predict(batch)

		np.save(os.path.join(dump_path, dump_name), scores)
		ipdb.set_trace()

if __name__=="__main__":
	main()

