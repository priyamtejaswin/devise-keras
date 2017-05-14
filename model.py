from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
import keras.backend as K 
import h5py
import sys, ipdb
from extract_features_and_dump import data_generator
import numpy as np

PATH_h5 = "processed_features/features.h5"
MARGIN = 0.1
INCORRECT_BATCH = 2
BATCH = INCORRECT_BATCH + 1
IMAGE_DIM = 4096
WORD_DIM = 50

def linear_transformation(a):
    ''' Takes a 4096-dim vector and applies 
        a linear transformation to get 500-dim vector '''
    b = Dense(WORD_DIM, name='transform')(a)
    return b

def myloss(word_vectors, image_vectors, TESTING=False):
    """write your loss function here, e.g mse"""
    slice_first = lambda x: x[0:1 , :]
    slice_but_first = lambda x: x[1:, :]

    # separate correct/wrong images
    correct_image = Lambda(slice_first, output_shape=(1, WORD_DIM))(image_vectors)
    wrong_images = Lambda(slice_but_first, output_shape=(INCORRECT_BATCH, WORD_DIM))(image_vectors)

    # separate correct/wrong words
    correct_word = Lambda(slice_first, output_shape=(1, WORD_DIM))(word_vectors)
    wrong_words = Lambda(slice_but_first, output_shape=(INCORRECT_BATCH, WORD_DIM))(word_vectors)

    # l2 norm
    l2 = lambda x: K.sqrt(K.sum(K.square(x)))
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
    cost_images = MARGIN - K.sum(correct_images * correct_words, 1) + K.sum(correct_images * wrong_words, 1) 
    cost_images = K.maximum(cost_images, 0.0)
    
    # correct_word VS incorrect_images | Note the singular/plurals
    cost_words = MARGIN - K.sum(correct_words * correct_images, 1) + K.sum(correct_words * wrong_images, 1) 
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
    
    return cost_words + cost_images
    

def build_model(image_features, word_features=None):
    image_vector = linear_transformation(image_features)

    mymodel = Model(inputs=image_features, outputs=image_vector)
    mymodel.compile(optimizer="adam", loss=myloss)
    return mymodel
    
def main():

    image_features = Input(shape=(4096,))
    model = build_model(image_features)
    print model.summary()

    for epoch in range(5):
        for raw_image_vectors, word_vectors in data_generator(batch_size = INCORRECT_BATCH):
            loss = model.train_on_batch(raw_image_vectors, word_vectors)
            print "loss:",loss

if __name__=="__main__":
    main()