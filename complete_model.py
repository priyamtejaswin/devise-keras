import numpy as np
import pickle 
from extract_features_and_dump import define_model
# from rnn_model import hinge_rank_loss
import keras 
from keras.models import Model, load_model 
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Embedding, LSTM, concatenate
import keras.backend as K
from keras.layers import Lambda
import sys, os 

# globals 
MAX_SEQUENCE_LENGTH = 20
WORD_DIM=300 
INCORRECT_BATCH=0
BATCH = INCORRECT_BATCH + 1
MARGIN = 0.2
IMAGE_DIM = 4096

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

	# separate the images from the captions==words
	image_vectors = Lambda(select_images, output_shape=(BATCH, WORD_DIM))(y_pred)
	word_vectors = Lambda(select_words, output_shape=(BATCH, WORD_DIM))(y_pred)

	# separate correct/wrong images
	correct_image = Lambda(slice_first, output_shape=(1, WORD_DIM))(image_vectors)

	# separate correct/wrong words
	correct_word = Lambda(slice_first, output_shape=(1, WORD_DIM))(word_vectors)

	# l2 norm
	l2 = lambda x: K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
	l2norm = lambda x: x/l2(x)
	
	correct_image = l2norm(correct_image)
	correct_word = l2norm(correct_word)

	# correct_image VS incorrect_words | Note the singular/plurals
	cost_images = K.sum(correct_image * correct_word, axis=1)
	cost_images = K.sum(cost_images, axis=-1)

	if TESTING:
		import ipdb
		ipdb.set_trace()
		assert K.eval(wrong_words).shape[0] == INCORRECT_BATCH
		assert K.eval(correct_words).shape[0] == INCORRECT_BATCH
		assert K.eval(wrong_images).shape[0] == INCORRECT_BATCH
		assert K.eval(correct_images).shape[0] == INCORRECT_BATCH
		assert K.eval(correct_words).shape==K.eval(correct_images).shape
		assert K.eval(wrong_words).shape==K.eval(wrong_images).shape
		assert K.eval(correct_words).shape==K.eval(wrong_images).shape
	
	return cost_images ## WAS EARLIER DIVIDING BY INCORRECT_BATCH WHICH HAS BEEN SET TO 0, IDIOT.

def build_model(image_features, caption_features, embedding_matrix):
	
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
	# embedding_matrix = pickle.load(open("KERAS_embedding_layer.TRAIN.pkl"))

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

	return concated

def get_full_model(vgg_wts_path, rnn_wts_path):

	vgg              = define_model(vgg_wts_path) # get image->4096 model
	
	caption_features = Input(shape=(MAX_SEQUENCE_LENGTH,), name="caption_feature_input") # the caption input 
	embedding_matrix = pickle.load(open("./KERAS_embedding_layer.TRAIN.pkl"))
		
	full_model       = build_model(vgg.output, caption_features, embedding_matrix=embedding_matrix) # caption+4096Feats -> loss function 
	
	# import ipdb
	# ipdb.set_trace()

	completeModel    = Model(inputs=[vgg.input, caption_features], outputs=full_model)
	completeModel.load_weights(rnn_wts_path, by_name=True) # load up the rnn part weights
	completeModel.compile("rmsprop", hinge_rank_loss)
	return completeModel


def TEST():
	vgg_path = sys.argv[1] 
	rnn_path = sys.argv[2]
	full_model = get_full_model(vgg_wts_path=vgg_path, rnn_wts_path=rnn_path)
	rnn_model  = load_model(rnn_path, custom_objects={"hinge_rank_loss":hinge_rank_loss})
	import ipdb; ipdb.set_trace()

	# check if wts were copied sucessfully 
	full_layer_names = [layer.name for layer in full_model.layers]
	rnn_layer_names  = [layer.name for layer in rnn_model.layers]

	common_layers = set(full_layer_names).intersection(set(rnn_layer_names))
	print "\nCommon layers: ", common_layers

	for layer in common_layers:
		print "layer: ", layer

		assert len(full_model.get_layer(layer).get_weights()) == len(rnn_model.get_layer(layer).get_weights()), "different number of weights" 

		if len(full_model.get_layer(layer).get_weights()) > 0:
			rnn_params = rnn_model.get_layer(layer).get_weights()
			full_params = full_model.get_layer(layer).get_weights()
			for rp, fp in zip(rnn_params, full_params):
				assert np.allclose(rp, fp), "Values were not equal!"
			print " .. OK"
				


	K.clear_session()

if __name__ == '__main__':
	TEST()

	y_true = K.variable(np.zeros(1))
	y_pred = K.variable(np.random.rand(1, 600))
	print K.eval(hinge_rank_loss(y_true, y_pred, TESTING=False))
