# devise-keras
tensorflow 1.1.0

keras 2.0.4

tensorboard 1.0.0a6

# ------------

bash SETUP.sh

python extract_features_and_dump.py -weights_path vgg16_weights_th_dim_ordering_th_kernels.h5 -images_path images/ -embeddings_path glove.6B.50d.txt -dump_path processed_features/

python _scrape_and_preprocess_captions.py SOURCE_PASCAL_SENTENCES_vision.cs.uiuc.edu.html

python model.py
