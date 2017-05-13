from keras.layers import * 
import keras.backend as K 

def linear_transformation(x):
	''' Takes a 4096-dim vector and applies 
	    a linear transformation to get 500-dim vector '''
	x = Dense(500, name='transform')(x)
	return x

def myloss(y_true, y_pred):
	# write your loss function here, e.g mse  
	K.mean(K.square(y_pred - y_true), axis=-1)
	return loss

def build_model():

	image_features = Input(shape=(4096,))
	image_vector = linear_transformation(ip)

	mymodel = Model(inputs=ip_image, output=image_vector, loss=myloss, optimizer='sgd')
	
	# load word vectors from disk as numpy 
	word_vectors_from_disk = load from numpy 

	model.train(ip_image, word_vectors_from_disk)



