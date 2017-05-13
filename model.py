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
	# write your loss function here, e.g mse  
	slice_first = lambda x: x[0, :]
	slice_but_first = lambda x: x[1:, :]

	correct_image = Lambda(slice_first)(image_vectors)
	wrong_images = Lambda(slice_but_first)(image_vectors)

	correct_word = Lambda(slice_first)(word_vectors)
	wrong_words = Lambda(slice_but_first)(word_vectors)

    l2 = lambda x: K.sqrt(K.sum(K.square(x)))
    l2norm = lambda x: x/l2(x)

    word = K.squeeze(correct_word)
    con_word = K.squeeze(K.concat(0, py_s[1:]))

    cap = tf.tile(cap, (num_con, 1))
    image = tf.tile(image, (num_con, 1))

    image = l2norm(image)
    con_image = l2norm(con_image)
    cap = l2norm(cap)
    con_cap = l2norm(con_cap)

    cost_im = margin - tf.reduce_sum((image * cap), 1) + tf.reduce_sum((image * con_cap), 1)
    cost_im = cost_im * tf.maximum(cost_im, 0.0)
    cost_im = tf.reduce_sum(cost_im, 0)

    cost_s  = margin - tf.reduce_sum((cap * image), 1) + tf.reduce_sum((cap * con_image), 1)
    cost_s  = cost_s  * tf.maximum(cost_s, 0.0)
    cost_s  = tf.reduce_sum(cost_s,  0)

    loss = cost_im + cost_s
    return loss

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
		image_fnames = fp["data/fnames"][:] ## convert to list of strings - NOT list of lists.

	# load pickle which contains class ranges
	with open("image_class_ranges.pkl", "r") as fp:
		class_ranges = pickle.load(fp)

	for epoch in range(2):
		for raw_image_vectors, word_vectors in data_generator(image_fnames, class_ranges, batch_size):
			model.train_on_batch(raw_image_vectors, word_vectors)