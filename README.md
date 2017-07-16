# mscoco-search extends devise-rnn

## Changes to the devise-rnn branch to train an image search engine using mscoco dataset.

The project uses the following python packages over the conda python stack:
- tensorflow 1.1.0
- keras 2.0.4
- tensorboard 1.0.0a6
- opencv 3.2.0

### SETUP
````
# edit local.cfg to set LOCAL/PROD in ENV

bash SETUP.sh

python extract_word_embeddings.py glove.6B.300d.txt processed_features/embeddings.h5

python scrape_and_save.py SOURCE_PASCAL_SENTENCES_vision.cs.uiuc.edu.html LOCAL

## Ensure DS_Store files are not in the image folders.

python clean_data.py UIUC_PASCAL_DATA

python extract_features_and_dump.py -weights_path vgg16_weights_th_dim_ordering_th_kernels.h5 -images_path UIUC_PASCAL_DATA_clean/  -dump_path processed_features/features.h5

python shuffle_val_data.py

python clean_data.py UIUC_PASCAL_VAL

python extract_features_and_dump.py -weights_path vgg16_weights_th_dim_ordering_th_kernels.h5 -images_path UIUC_PASCAL_VAL_clean/  -dump_path processed_features/validation_features.h5

rm snapshots/* ## Optional.

python rnn_model.py TRAIN
````

### Changes required:
DO NOT look at branch:master for merging the missing pieces. Look at branch:priyam_dev for reference - it contains the master code WITHOUT the config.

- .gitignore: Done
- base validation script: akshay is working on it - Done
- Updates/correct loss function: Done
- validation split: Done - now deletes the images from DICT_class_TO_images.pkl
- tensorboard changes + logging: Done(added tensorboard_logging.py) Done(logging)
- embedding separation: Done
- caption preparation: Done(not changed for now)
- update extract_featurs: Done
- update to use CONFIG
- changes for server??
- add more parameters to model
