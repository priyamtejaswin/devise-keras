import numpy as np
import pickle 
from extract_features_and_dump import define_model
from rnn_model import hinge_rank_loss
import keras 
from keras.models import Model, load_model 
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Embedding, LSTM, concatenate
import keras.backend as K
from keras.layers import Lambda
import sys, os 
from keras.applications import VGG16

class FullModel:
	
	def __init__(self, caption, rnn_model_loc):
		
		assert len(caption) == 20, "provided caption has incorrect length of {}".format(len(caption))
		assert os.path.isfile(rnn_model_loc), "Provided incorect location to rnn model"

		# vgg16 - fc2 output
		top_model = VGG16(weights="imagenet", include_top="True")
		self.top_model = Model(inputs=top_model.input, outputs=top_model.get_layer("fc2").output)

		# rnn part + recompile with new loss
		self.rnn_model = load_model(, custom_objects={"hinge_rank_loss":hinge_rank_loss})
		self.rnn_model.compile(optimizer="rmsprop", loss=FullModel.custom_distance_function)

		# fixed caption (to be used against perturbed images to calculate loss)
		self.caption = caption

		# some globals 
		self.WORD_DIM = 300
		self.BATCH = 1 

	@staticmethod
	def custom_distance_function(y_true, y_pred):

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

		return cost_images ## WAS EARLIER DIVIDING BY INCORRECT_BATCH WHICH HAS BEEN SET TO 0, IDIOT.

	def predict(x):

		# checks
		assert x.shape == (1,224,224,3), "Incorrect input dims of {}".format(x.shape) 
		assert self.caption.shape == (1,20), "Incorrect shape of captions having shape {}".format(self.caption.shape) 

		# pass through vgg 
		top_op = self.top_model.predict(x)

		# pass through rnn model (get loss) 
		loss = self.rnn_model.test_on_batch([x,self.caption], np.zeros(1))

		return loss

	
def TEST():
	mod = FullModel()

if __name__ == '__main__':
	TEST()

		