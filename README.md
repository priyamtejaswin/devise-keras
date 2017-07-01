# devise-keras
The **master** branch contains code to setup the experiment, download/pre-process data for implementing the [Google-Devise paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41869.pdf). `model.py` contains the code for the model. Due to computational constraints, the experiments are run on the [UIUC PASCAL sentences dataset](http://vision.cs.uiuc.edu/pascal-sentences/) as opposed to [ImageNet](http://www.image-net.org/). This dataset contains 50 images per category for 20 categories along with 5 captions per image. These captions are used for the extension of the project in the **devise-rnn** branch.

The project uses the following python packages over the conda python stack:
- tensorflow 1.1.0
- keras 2.0.4
- tensorboard 1.0.0a6
- opencv 3.2.0

### SETUP
````
bash SETUP.sh

python scrape_and_save_images.py SOURCE_PASCAL_SENTENCES_vision.cs.uiuc.edu.html

python extract_word_embeddings.py glove.6B.50d.txt processed_features/embeddings.h5

python shuffle_val_data.py

## Ensure DS_Store files are not in the image folders.

python clean_data.py UIUC_PASCAL_DATA

python clean_data.py UIUC_PASCAL_VAL

python extract_features_and_dump.py -weights_path vgg16_weights_th_dim_ordering_th_kernels.h5 -images_path UIUC_PASCAL_DATA_clean/ -embeddings_path glove.6B.50d.txt -dump_path processed_features/features.h5 -image_class_ranges training_ranges.pkl

python extract_features_and_dump.py -weights_path vgg16_weights_th_dim_ordering_th_kernels.h5 -images_path UIUC_PASCAL_VAL_clean/ -embeddings_path glove.6B.50d.txt -dump_path processed_features/validation_features.h5 -image_class_ranges validation_ranges.pkl

rm snapshots/* ## Optional.

python model.py TRAIN
````
