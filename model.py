from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
import keras.backend as K 
import h5py
import sys
from extract_features_and_dump import data_generator

PATH_h5 = "processed_features/features.h5"
MARGIN = 0.1
INCORRECT_BATCH = 3
BATCH = INCORRECT_BATCH + 1

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
    correct_image = Lambda(slice_first, output_shape=(1, None))(image_vectors)
    wrong_images = Lambda(slice_but_first, output_shape=(INCORRECT_BATCH, None))(image_vectors)

    # separate correct/wrong words
    correct_word = Lambda(slice_first, output_shape=(1, None))(word_vectors)
    wrong_words = Lambda(slice_but_first, output_shape=(INCORRECT_BATCH, None))(word_vectors)

    # l2 norm
    l2 = lambda x: K.sqrt(K.sum(K.square(x)))
    l2norm = lambda x: x/l2(x)

    # tiling to replicate correct_word and correct_image
    correct_words = K.tile(correct_word, INCORRECT_BATCH)
    correct_images = K.tile(correct_image, INCORRECT_BATCH)

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
    image_vector = linear_transformation(image_features)

    mymodel = Model(inputs=image_features, outputs=image_vector)
    mymodel.compile(optimizer="adam", loss=myloss)
    return mymodel
    # load word vectors from disk as numpy 
    # word_vectors_from_disk = load from numpy 

    # model.train(ip_image, word_vectors_from_disk)

def main():

    image_features = Input(shape=(4096,))
    model = build_model(image_features)

    for epoch in range(5):
        for raw_image_vectors, word_vectors in data_generator(batch_size = INCORRECT_BATCH):
            loss = model.train_on_batch(raw_image_vectors, word_vectors)
            print loss

if __name__=="__main__":
    main()