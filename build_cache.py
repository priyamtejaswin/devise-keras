''' This Script builds a cache of validation and training features in a .h5 file 
input -> location of features.h5 and validation_features.h5 
output -> A HDF5 file containing combined version of features+validation_features and their forward 
pass through rnn model (50-dim) 
'''
import h5py 
import numpy as np
from tqdm import *
from rnn_model import hinge_rank_loss

IMAGE_DIM = 4096
WORD_DIM  = 300
model_location = "/home/throwaway1akshaychawla/cache_ui/epoch_10.hdf5"
MAX_SEQUENCE_LENGTH = 20

def dump_to_h5(names, scores ,hf):
    ''' Dump the list of names and the numpy array of scores 
        to given h5 file '''
    
    assert int(len(scores)) == len(names), "Number of output scores == number of file names to dump"
    
    x_h5 = hf["data/features"]
    fnames_h5 = hf["data/fnames"]

    cur_rows = int(x_h5.shape[0]) 
    new_rows = cur_rows + len(names) 

    x_h5.resize((new_rows,IMAGE_DIM))
    fnames_h5.resize((new_rows,1))

    for i in range(len(names)): 
        x_h5[cur_rows+i] = scores[i]
        fnames_h5[cur_rows+i] = names[i]

def main():

    validation_features_loc = "processed_features/validation_features.h5"
    training_features_loc = "processed_features/features.h5"
    import ipdb; ipdb.set_trace()
    train_features_h5 = h5py.File(training_features_loc, "r")
    valid_features_h5 = h5py.File(validation_features_loc, "r")

    cache_h5 = h5py.File("cache.h5","w")
    cache_h5.create_group("data")

    data_h5 = cache_h5["data"].create_dataset("features", (0, IMAGE_DIM), maxshape=(None,IMAGE_DIM))
    dt = h5py.special_dtype(vlen=str)
    fnames_h5= cache_h5["data"].create_dataset("fnames", (0, 1), dtype=dt, maxshape=(None,1))

    # copy image feats+fnames from features.h5 to cache/data/features
    print "Copying features from features.h5 to cache.h5"
    batch_size = 500
    for lix in tqdm(xrange(0, len(train_features_h5["data/features"]), batch_size)):
        uix = min(len(train_features_h5["data/features"]), lix + batch_size)

        names = train_features_h5["data/fnames"][lix:uix]
        names = [n[0] for n in names]
        dump_to_h5( names, train_features_h5["data/features"][lix:uix], cache_h5 )

    # copy image feats+fnames from validation_features.h5 to cache/data/features
    # print "Copying validation features from features.h5 to cache.h5"
    # batch_size = 500
    # for lix in tqdm(xrange(0, len(valid_features_h5["data/features"]), batch_size)):
    #     uix = min(len(valid_features_h5["data/features"]), lix + batch_size)

    #     names = valid_features_h5["data/fnames"][lix:uix]
    #     names = [n[0] for n in names]
    #     dump_to_h5( names, valid_features_h5["data/features"][lix:uix], cache_h5 )

    # Load model 
    from keras.models import load_model
    print "..Loading model"
    model = load_model(model_location, custom_objects={"hinge_rank_loss":hinge_rank_loss})
    
    # Run image feats through model to get 300-dim embedding
    im_outs = cache_h5["data"].create_dataset("im_outs", (0, WORD_DIM), maxshape=(None, WORD_DIM))
    all_features = cache_h5["data/features"]
    print "Running model on all features of size", all_features.shape
    batch_size = 500
    for lix in tqdm(xrange(0, len(all_features), batch_size)):
        uix = min(len(all_features), lix + batch_size)
        output = model.predict([ all_features[lix:uix, :], np.zeros((uix-lix, MAX_SEQUENCE_LENGTH))])[:, :WORD_DIM]
        output = output / np.linalg.norm(output, axis=1, keepdims=True)

        # add ^ output to im_outs
        im_outs.resize((uix,WORD_DIM)) # expand size of im_outs (NOTE: we have uix and not uix+1 because uix=lix+batch)
        assert len(im_outs) == uix-lix, "lenth of im_outs MUST be == uix (which goes through entire dataset"
        im_outs[lix:uix] = output

    # CLOSE ALL H5
    train_features_h5.close()
    valid_features_h5.close()
    cache_h5.close()


if __name__ == '__main__':
    main()