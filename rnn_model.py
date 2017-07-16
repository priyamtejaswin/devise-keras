from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Activation, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import RemoteMonitor
import keras
from time import time, sleep
import keras.backend as K 
import h5py
import sys, ipdb
import math, os, sys
from extract_features_and_dump import data_generator_coco
import numpy as np
from keras.callbacks import TensorBoard
# from validation_script import ValidCallBack
import cv2
import pickle
import numpy as np

PATH_h5 = "processed_features/features.h5"
MARGIN = 0.2
INCORRECT_BATCH = 32
BATCH = INCORRECT_BATCH + 1
IMAGE_DIM = 4096
WORD_DIM = 300
MAX_SEQUENCE_LENGTH = 20

class DelayCallback(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		pass

	def on_batch_end(self, batch, logs={}):
		sleep(0.01)

class EpochCheckpoint(keras.callbacks.Callback):
	def __init__(self, folder):
		super(EpochCheckpoint, self).__init__()
		assert folder is not None, "Err. Please specify a folder where models will be saved"
		self.folder = folder
		print "[LOG] EpochCheckpoint: folder to save models: "+self.folder

	def on_epoch_end(self, epoch, logs={}):
		print " Saving model..."
		self.model.save(os.path.join(self.folder,"epoch_"+str(epoch)+".hdf5"))

def get_num_train_images(from_pkl=False):
	'''
        if path_to_pkl is NOT False: get it from the pickle. else
	get the number of training images in processed_features/features.h5
	'''
        if from_pkl is not False:
            class_TO_images = pickle.load(open("DICT_class_TO_images.pkl"))
            return sum(len(l) for l in class_TO_images.itervalues())
	
	hf = h5py.File(PATH_h5, "r")
	x_h5 = hf["data/features"]
	num_train = x_h5.shape[0]
	hf.close()

	return num_train

def hinge_rank_loss(y_true, y_pred, TESTING=False):
	"""
	Custom hinge loss per (image, label) example - Page4.
	
	Keras mandates the function signature to follow (y_true, y_pred)
	In devise:master model.py, this function accepts:
	- y_true as word_vectors
	- y_pred as image_vectors

	For the rnn_model, the image_vectors and the caption_vectors are concatenated.
	This is due to checks that Keras has enforced on (input,target) sizes 
	and the inability to handle multiple outputs in a single loss function.

	These are the actual inputs to this function:
	- y_true is just a dummy placeholder of zeros (matching size check)
	- y_pred is concatenate([image_output, caption_output], axis=-1)
	The image, caption features are first separated and then used.
	"""
	## y_true will be zeros
	select_images = lambda x: x[:, :WORD_DIM]
	select_words = lambda x: x[:, WORD_DIM:]

	slice_first = lambda x: x[0:1 , :]
	slice_but_first = lambda x: x[1:, :]

	# separate the images from the captions==words
	image_vectors = Lambda(select_images, output_shape=(BATCH, WORD_DIM))(y_pred)
	word_vectors = Lambda(select_words, output_shape=(BATCH, WORD_DIM))(y_pred)

	# separate correct/wrong images
	correct_image = Lambda(slice_first, output_shape=(1, WORD_DIM))(image_vectors)
	wrong_images = Lambda(slice_but_first, output_shape=(INCORRECT_BATCH, WORD_DIM))(image_vectors)

	# separate correct/wrong words
	correct_word = Lambda(slice_first, output_shape=(1, WORD_DIM))(word_vectors)
	wrong_words = Lambda(slice_but_first, output_shape=(INCORRECT_BATCH, WORD_DIM))(word_vectors)

	# l2 norm
	l2 = lambda x: K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
	l2norm = lambda x: x/l2(x)

	# tiling to replicate correct_word and correct_image
	correct_words = K.tile(correct_word, (INCORRECT_BATCH,1))
	correct_images = K.tile(correct_image, (INCORRECT_BATCH,1))

	# converting to unit vectors
	correct_words = l2norm(correct_words)
	wrong_words = l2norm(wrong_words)
	correct_images = l2norm(correct_images)
	wrong_images = l2norm(wrong_images)

	# correct_image VS incorrect_words | Note the singular/plurals
	cost_images = MARGIN - K.sum(correct_images * correct_words, axis=1) + K.sum(correct_images * wrong_words, axis=1) 
	cost_images = K.maximum(cost_images, 0.0)
	
	# correct_word VS incorrect_images | Note the singular/plurals
	cost_words = MARGIN - K.sum(correct_words * correct_images, axis=1) + K.sum(correct_words * wrong_images, axis=1) 
	cost_words = K.maximum(cost_words, 0.0)

	# currently cost_words and cost_images are vectors - need to convert to scalar
	cost_images = K.sum(cost_images, axis=-1)
	cost_words  = K.sum(cost_words, axis=-1)

	if TESTING:
		# ipdb.set_trace()
		assert K.eval(wrong_words).shape[0] == INCORRECT_BATCH
		assert K.eval(correct_words).shape[0] == INCORRECT_BATCH
		assert K.eval(wrong_images).shape[0] == INCORRECT_BATCH
		assert K.eval(correct_images).shape[0] == INCORRECT_BATCH
		assert K.eval(correct_words).shape==K.eval(correct_images).shape
		assert K.eval(wrong_words).shape==K.eval(wrong_images).shape
		assert K.eval(correct_words).shape==K.eval(wrong_images).shape
	
	return (cost_words + cost_images) / INCORRECT_BATCH
	

def build_model(image_features, caption_features):
	
	# visual model 	
	# 	Hidden Layer - 1	
	image_dense1 = Dense(WORD_DIM, name="image_dense1")(image_features)
	image_dense1 = BatchNormalization()(image_dense1)
	image_dense1 = Activation("relu")(image_dense1)
	image_dense1 = Dropout(0.5)(image_dense1)	
	
	#   Hidden Layer - 2
	image_dense2 = Dense(WORD_DIM, name="image_dense2")(image_dense1)
	image_output = BatchNormalization()(image_dense2)

	# rnn model
	embedding_matrix = pickle.load(open("KERAS_embedding_layer.pkl"))

	cap_embed = Embedding(
		input_dim=embedding_matrix.shape[0],
		output_dim=WORD_DIM,
		weights=[embedding_matrix],
		input_length=MAX_SEQUENCE_LENGTH,
		trainable=False,
		name="caption_embedding"
		)(caption_features)

	lstm_out_1 = LSTM(300, return_sequences=True)(cap_embed)
	lstm_out_2 = LSTM(300)(lstm_out_1)
	caption_output = Dense(WORD_DIM, name="lstm_dense")(lstm_out_2)
	caption_output = BatchNormalization()(caption_output)

	concated = concatenate([image_output, caption_output], axis=-1)

	mymodel = Model(inputs=[image_features, caption_features], outputs=concated)
	mymodel.compile(optimizer="rmsprop", loss=hinge_rank_loss)
	return mymodel

def main():
	RUN_TIME = sys.argv[1]


	if RUN_TIME == "TRAIN":
		image_features = Input(shape=(4096,), name="image_feature_input")
		caption_features = Input(shape=(MAX_SEQUENCE_LENGTH,), name="caption_feature_input")

		model = build_model(image_features, caption_features)
		print model.summary()

		# number of training images 
		_num_train = get_num_train_images(from_pkl=False)
		# _num_train = 6

		# Callbacks 
		# remote_cb = RemoteMonitor(root='http://localhost:9000')
		tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
		epoch_cb    = EpochCheckpoint(folder="./snapshots/")
		# valid_cb    = ValidCallBack()

		# fit generator
		steps_per_epoch = math.ceil(_num_train*5) # /float(BATCH))
		print "Steps per epoch i.e number of iterations: ",steps_per_epoch
		
		train_datagen = data_generator_coco(incorrect_batch=INCORRECT_BATCH)
		history = model.fit_generator(
				train_datagen,
				steps_per_epoch=steps_per_epoch,
				epochs=100,
				callbacks=[tensorboard, epoch_cb]
			)
		print history.history.keys()


	elif RUN_TIME == "TEST":
		from keras.models import load_model 
		model = load_model("snapshots/epoch_69.hdf5", custom_objects={"hinge_rank_loss":hinge_rank_loss})

	hf = h5py.File("processed_features/features.h5","r")

	im_samples = hf["data/features"][:, :]
	word_index = pickle.load(open("DICT_word_index.pkl"))

	string = "man riding a bike"
	cap_sample = [word_index[x] for x in string.strip().split()]
	cap_sample = np.array([ cap_sample + [0 for i in range(MAX_SEQUENCE_LENGTH-len(cap_sample))] ])
	cap_sample = np.tile(cap_sample, (im_samples.shape[0], 1))

	## TESTING
	test_out = model.predict([im_samples, cap_sample], batch_size=5) ## Cannot do this because Keras expects a single output to be returned for a single input; while my ugly hack concats and returns two!
	im_outs = test_out[:, :WORD_DIM]
	cap_out = test_out[:, WORD_DIM:]

	print im_outs.shape
	print cap_out.shape

	im_outs = im_outs / np.linalg.norm(im_outs, axis=1, keepdims=True)
	cap_out = cap_out / np.linalg.norm(cap_out, axis=1, keepdims=True)

	diff = im_outs - cap_out
	diff = np.linalg.norm(diff, axis=1)
	print np.argsort(diff)[:25]
	print np.sort(diff)[:25]

	ipdb.set_trace()

	K.clear_session()

if __name__=="__main__":
	main()
