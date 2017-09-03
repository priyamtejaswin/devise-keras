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
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from extract_features_and_dump import define_model as TheVGGModel
from lime import lime_image
from scipy.spatial.distance import cdist

class FullModel:
	
	def __init__(self, caption, rnn_model_loc):
		# some globals 
		self.WORD_DIM = 300
		self.BATCH = 1 
		
		# fixed caption (to be used against perturbed images to calculate loss)
		self.caption = caption

		assert self.caption.shape == (1,20), "Incorrect shape of captions having shape {}".format(self.caption.shape) 
		assert os.path.isfile(rnn_model_loc), "Provided incorect location to rnn model"

		# vgg16 - fc2 output
		# top_model = VGG16(weights="imagenet", include_top="True")
		# self.top_model = Model(inputs=top_model.input, outputs=top_model.get_layer("fc2").output)
		self.top_model = TheVGGModel("/Users/tejaswin.p/projects/devise-keras/vgg16_weights_th_dim_ordering_th_kernels.h5")

		# rnn part + recompile with new loss
		self.rnn_model = load_model(rnn_model_loc, custom_objects={"hinge_rank_loss":hinge_rank_loss})
		self.rnn_model.compile(optimizer="rmsprop", loss=self.custom_distance_function)

	def custom_distance_function(self, y_true, y_pred, DEBUG=False):
		select_images = lambda x: x[:, :self.WORD_DIM]
		select_words = lambda x: x[:, self.WORD_DIM:]

		# separate the images from the captions==words
		image_vectors = Lambda(select_images, output_shape=(self.BATCH, self.WORD_DIM))(y_pred)
		word_vectors = Lambda(select_words, output_shape=(self.BATCH, self.WORD_DIM))(y_pred)

		# l2 norm
		l2 = lambda x: K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
		l2norm = lambda x: x/l2(x)
		
		image_vectors = l2norm(image_vectors)
		word_vectors = l2norm(word_vectors)

		# correct_image VS incorrect_words | Note the singular/plurals
		print "\n\n\t\tAM I PRINTING/COMPILING THE NEW CUSTOM LOSS??\n\n"
		cost = K.sum(image_vectors * word_vectors, axis=1)
		# cost_images = K.sum(cost_images, axis=-1) ## Commented because values expected for multiple inputs.

		if DEBUG:
			import ipdb
			ipdb.set_trace()

		return cost ## WAS EARLIER DIVIDING BY INCORRECT_self.BATCH WHICH HAS BEEN SET TO 0, IDIOT.

	def predict(self, x):

		# checks
		# assert x.shape == (1, 3, 224, 224), "Incorrect input dims of {}".format(x.shape) 
		## Change from (10, 224, 224, 3) to (10, 3, 224, 224)

		# x_rolled = np.moveaxis(np.moveaxis(x, 3, 1), 2, 3)

		assert len(x.shape)==4, "--WHOA, the shape is not 4?????--"

		if x.shape[-1]==3:
			print "CHANGING SHAPE"
			x_rolled = np.swapaxes(np.swapaxes(x, 3, 2), 2, 1)
		else:
			print "NOT CHANGING SHAPE"
			x_rolled = x

		print x_rolled.shape

		## pass through vgg 
		top_op = self.top_model.predict(x_rolled)
		import ipdb;ipdb.set_trace()

		## pass through rnn model (get loss) 
		rnn_output = self.rnn_model.predict( 
			[top_op, np.tile(self.caption, (x_rolled.shape[0], 1))]
			) ## returns a (n, 600) array

		print rnn_output.shape

		## calculate the distance b/w image perturbations and the caption
		word_vectors = rnn_output[:, :self.WORD_DIM]
		caption_vectors = rnn_output[:, self.WORD_DIM:]
		loss = 1 - cdist(word_vectors, caption_vectors, metric="cosine")

		return loss

	@staticmethod
	def preprocess_image(img_path):
		"""
		Pre-process image by ImageNet standards.
		Returns by expanding dims for "batch" dimension
		and type casting to np.float64: just scikit-image things...
		"""
		img_input = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img_input)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		x = x.astype(np.float64)
		return x	

	
def TEST_model():
	cap_input = np.array([[8, 214, 23, 1, 626, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
	model = FullModel(cap_input, "/Users/tejaswin.p/Downloads/epoch_9.hdf5")

	# import ipdb
	# ipdb.set_trace()

	x = model.preprocess_image("/Users/tejaswin.p/Downloads/eating_pizza_in_park.jpg")
	print x.shape

	# x = np.moveaxis(np.moveaxis(x, 1, 0), 2, 1)
	# x = np.expand_dims(x, axis=0)
	# print x.shape

	print "CORRECT caption similarity:", model.predict(x)

	model.caption = np.array([[8, 199, 23, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
	print "RANDOM caption similarity:", model.predict(x)

	K.clear_session()

def TEST_lime():
	cap_input = np.array([[8, 214, 23, 1, 626, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
	model = FullModel(cap_input, "/Users/tejaswin.p/Downloads/epoch_9.hdf5")
	
	x = model.preprocess_image("/Users/tejaswin.p/Downloads/eating_pizza_in_park.jpg")
	print x.shape

	print "\nMoving to 'channels last'."
	x = np.swapaxes(np.swapaxes(x, 1, 2), 2, 3)
	print x.shape

	print "\nSelecting the image...eliminating the 'batch' dimension."
	x = x[0] ## scikit-image works with batch last and only for single images.
	print x.shape

	import ipdb
	ipdb.set_trace()

	explainer = lime_image.LimeImageExplainer() ## LIME explainer.
	explanation = explainer.explain_instance(x, model.predict , top_labels=1, hide_color=0, num_samples=100)

	K.clear_session()

if __name__ == '__main__':
	TEST_model()
	# TEST_lime()
