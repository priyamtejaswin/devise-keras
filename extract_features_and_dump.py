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
import os, sys, ipdb
import cPickle as pickle
from tqdm import *

np.random.seed(123)

IMAGE_DIM = 4096
WORD_DIM = 50

def get_class_ranges(fnames):
	class_ranges = {} 

	for idx,fname in enumerate(fnames):
		class_name = fname.split("/")[-2]
		if class_name not in class_ranges.keys():
			class_ranges[class_name] = [idx,idx]
		else:
			class_ranges[class_name][1] = idx

	return class_ranges

def data_generator(path_to_h5py="processed_features/features.h5", batch_size=2):
	
	#print "\nloading data for training...\n"
	
	
	# load all image fnames
	with h5py.File(path_to_h5py, "r") as fp:
		image_fnames = map(lambda a:a[0], fp["data/fnames"][:]) #fnames is list of single lists!

	# load pickle which contains class ranges
	with open("image_class_ranges.pkl", "r") as fp:
		class_ranges = pickle.load(fp)

	F 			 = h5py.File(path_to_h5py, "r")
	vgg_feats 	 = F["data/features"]
	embeddings   = F["data/word_embeddings"]
	word_mapping = {l[0]:i for i,l in enumerate(F["data/word_names"])}
	DATASET_SIZE = len(image_fnames)
	#print "done\n"

	while 1:
		for i in range(DATASET_SIZE):
			X = np.zeros((1+batch_size, 4096))
			y = np.zeros((1+batch_size, 50))

			# correct one - first one 
			X[0] = vgg_feats[i]
			y[0] = embeddings[word_mapping[image_fnames[i].split("/")[-2]]][np.newaxis, :]

			# others - remaining
			class_of_first 	= image_fnames[i].split("/")[-2]
			start,end 		= class_ranges[class_of_first]
			range_of_nums 	= range(0,start) + range(end+1,DATASET_SIZE)
			selected_indices= np.random.choice(range_of_nums, size=batch_size, replace=False).tolist() # missed this! select without replacement! to avoid [1,1] error
			selected_indices= sorted(selected_indices) # unordered indexing is not supported?

			# print i, start, end, class_of_first, selected_indices
			X[1:]   		= vgg_feats[selected_indices]

			selected_words  = map(lambda a:image_fnames[a].split("/")[-2], selected_indices)
			selected_embeds = np.concatenate(
								map(lambda w:embeddings[word_mapping[w]][np.newaxis, :], selected_words), 
								0)
			y[1:]			= selected_embeds
			
			# print epoch, i
			yield X, y


def dump_to_h5(names, scores ,hf):
	''' Dump the list of names and the numpy array of scores 
		to given h5 file '''
	
	assert int(len(scores)) == len(names), "Number of output scores == number of file names to dump"
	
	x_h5 = hf["data/features"]
	fnames_h5 = hf["data/fnames"]

	cur_rows = int(x_h5.shape[0]) 
	new_rows = cur_rows + len(names) 

	x_h5.resize((new_rows,IMAGE_DIM))
	fnames_h5.resize((new_rows,1))

	for i in range(len(names)): 
		x_h5[cur_rows+i] = scores[i]
		fnames_h5[cur_rows+i] = names[i]

def dump_wv_to_h5(words, vectors, hf):
	assert int(len(vectors)) == len(words), "Number of words == number of vectors"

	v_h5 = hf["data/word_embeddings"]
	w_h5 = hf["data/word_names"]

	cur_rows = int(v_h5.shape[0]) 
	new_rows = cur_rows + len(words)

	v_h5.resize((new_rows, WORD_DIM))
	w_h5.resize((new_rows, 1))

	for i in range(len(words)):
		v_h5[cur_rows+i] = vectors[i]
		w_h5[cur_rows+i] = words[i]

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
	x = Dense(IMAGE_DIM, activation='relu', name='fc1')(x)
	x = Dense(IMAGE_DIM, activation='relu', name='fc2')(x)

	model = Model(inputs=img_input, outputs=x, name="vgg16")

	# load wts
	model.load_weights(path, by_name=True)

	return model  

def create_indices(total_length, batch_size):
	return izip(xrange(0, total_length, batch_size), xrange(batch_size, total_length+batch_size, batch_size))


def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-weights_path", help="weights file path")
	parser.add_argument("-images_path", help="folder where images are located")
	parser.add_argument("-embeddings_path", help="binary where word embeddings are saved")
	parser.add_argument("-dump_path", help="folder where features will be dumped")
	args = parser.parse_args()

	weights_path 	= args.weights_path
	images_path 	= args.images_path
	dump_path   	= args.dump_path
	embeddings_path = args.embeddings_path

	assert os.path.isdir(images_path), "---path is not a folder--"
	assert os.path.isdir(dump_path), "---path is not a folder--"
	
	print "defining model.."
	model = define_model(weights_path)
	
	dir_fnames = []
	for dirpath, dirnames, filenames in os.walk(images_path):
		if filenames != []:
			dir_fnames += [os.path.join(dirpath, fn) for fn in filenames]
	list_of_files = dir_fnames

	print "Total files:", len(list_of_files)
	
	print "creating h5py files features.h5.."
	# h5py 
	hf = h5py.File(os.path.join(dump_path,"features.h5"),"w")
	data = hf.create_group("data")
	x_h5 = data.create_dataset("features",(0,IMAGE_DIM), maxshape=(None,IMAGE_DIM))
	dt   = h5py.special_dtype(vlen=str)
	fnames_h5 = data.create_dataset("fnames",(0,1),dtype=dt, maxshape=(None,1))

	# extract and dump image features
	print "Dumping image features.."
	for i,j in tqdm(create_indices(len(list_of_files), batch_size=2)):
		
		j = min(j, len(list_of_files))

		loaded_images = []
		dump_names = []

		for k in range(i,j,1):
			
			dump_names.append(list_of_files[k])

			img = image.load_img(list_of_files[k], target_size=(224, 224))
			img = image.img_to_array(img)
			loaded_images.append(img)

		loaded_images = np.array(loaded_images)
		batch = preprocess_input(loaded_images)
		
		scores = model.predict(batch)
		#scores = np.random.randn(len(loaded_images), IMAGE_DIM)

		dump_to_h5(names=dump_names, scores=scores, hf=hf)

	# extract and dump class ranges
	class_ranges = get_class_ranges(map(lambda a:a[0], fnames_h5[:])) # fnames is list of single lists!
	with open("image_class_ranges.pkl","w") as f:
		pickle.dump(class_ranges, f)
		print "...saved to pickle image_class_ranges.pkl"

	# extract and dump word vectors
	print "Dumping word embeddings..."
	_ = data.create_dataset("word_embeddings", (0, WORD_DIM), maxshape=(None, WORD_DIM))
	_ = data.create_dataset("word_names", (0, 1), dtype=dt, maxshape=(None, 1))

	embeddings_index = {}
	f = open(embeddings_path)
	word_batch, vector_batch, _c = [], [], 1

	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	    word_batch.append(word)
	    vector_batch.append(coefs)
	    
	    if _c%5==0:
	    	dump_wv_to_h5(word_batch, vector_batch, hf)
	    	word_batch, vector_batch = [], []
	    _c+=1

	dump_wv_to_h5(word_batch, vector_batch, hf) # to catch the trailing vectors
	f.close()
	print 'Found %s word vectors.' % len(embeddings_index)

if __name__=="__main__":
	main()

