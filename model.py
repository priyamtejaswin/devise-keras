from keras.layers import Dense 

def linear_transformation(x):
	''' Takes a 4096-dim vector and applies 
	    a linear transformation to get 500-dim vector '''
	x = Dense(500, name='transform')(x)
	return x

