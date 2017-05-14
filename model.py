from keras.layers import * 
import keras.backend as K 
import h5py
import sys
from extract_features_and_dump import data_generator

def linear_transformation(x):
	''' Takes a 4096-dim vector and applies 
	    a linear transformation to get 500-dim vector '''
	x = Dense(500, name='transform')(x)
	return x

def myloss(image_vectors, word_vectors):
	"""write your loss function here, e.g mse"""

	slice_first = lambda x: x[0, :]
	slice_but_first = lambda x: x[1:, :]

	# separate correct/wrong images
	correct_image = Lambda(slice_first)(image_vectors)
	wrong_images = Lambda(slice_but_first)(image_vectors)

	# separate correct/wrong words
	correct_word = Lambda(slice_first)(word_vectors)
	wrong_words = Lambda(slice_but_first)(word_vectors)

	# l2 norm
    l2 = lambda x: K.sqrt(K.sum(K.square(x)))
    l2norm = lambda x: x/l2(x)

    # tiling to replicate correct_word and correct_image
    correct_words = K.tile(correct_word, K.shape(wrong_words)[0])
    correct_images = K.tile(correct_image, K.shape(wrong_images)[0])

    # converting to unit vectors
    correct_words = l2norm(correct_words)
    wrong_words = l2norm(wrong_words)
    correct_images = l2norm(correct_images)
    wrong_images = l2norm(wrong_images)

    # correct_image VS incorrect_words | Note the singular/plurals
    cost_images = K.maximum(
    	MARGIN - K.sum(correct_images * correct_words, 1) + K.sum(correct_images * wrong_words) , 
    	0.0)
    # correct_word VS incorrect_images | Note the singular/plurals
    cost_words = K.maximum(
    	MARGIN - K.sum(correct_words * correct_images, 1) + K.sum(correct_words * wrong_images) , 
    	0.0)

    return cost_images + cost_words
    

def build_model(image_features, word_features=None):
	image_vector = linear_transformation(ip)

	mymodel = Model(inputs=ip_image, output=image_vector, loss=myloss, optimizer='sgd')
	return mymodel
	# load word vectors from disk as numpy 
	# word_vectors_from_disk = load from numpy 

	# model.train(ip_image, word_vectors_from_disk)

def main():
	path_to_h5py 

	image_features = Input(shape=(4096,))
	model = build_model(image_features)

	# load all image fnames
	with h5py.File(path_to_h5py, "r") as fp:
		image_fnames = reduce(lambda x,y:x+y, fp["data/fnames"][:]) ## convert to list of strings - NOT list of lists.

	# load pickle which contains class ranges
	with open("image_class_ranges.pkl", "r") as fp:
		class_ranges = pickle.load(fp)

	for epoch in range(2):
		for raw_image_vectors, word_vectors in data_generator(image_fnames, class_ranges, batch_size):
			model.train_on_batch(raw_image_vectors, word_vectors)