# devise-rnn: extends devise-keras

The project uses the following python packages over the conda python stack:
- tensorflow 1.1.0
- keras 2.0.4
- tensorboard 1.0.0a6
- opencv 3.2.0

### SETUP

bash SETUP.sh

python _scrape_and_preprocess_captions.py SOURCE_PASCAL_SENTENCES_vision.cs.uiuc.edu.html glove.6B.50d.txt

python clean_data.py

python extract_features_and_dump.py -weights_path vgg16_weights_th_dim_ordering_th_kernels.h5 -images_path UIUC_PASCAL_DATA_clean/ -embeddings_path glove.6B.50d.txt -dump_path processed_features/

python model.py TRAIN

### Changes required:
DO NOT look at branch:master for merging the missing pieces. Look at branch:priyam_dev for reference - it contains the master code WITHOUT the config.

- .gitignore: Done
- base validation script: akshay is working on it
- Updates/correct loss function: Done
- validation split: Done - now deletes the images from DICT_class_TO_images.pkl
- tensorboard changes + logging: Done(added tensorboard_logging.py)
- embedding separation: Done
- caption preparation: Done(not changed for now)
