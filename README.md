# Video2Text_Generator_MS_Thesis
This research designed a video-to-text generator using the largest MSR-VTT and MSVD datasets. The models were designed in four parts: feature extraction, model training, test caption generation, and model evaluation.

Video features, which were extracted using pretrained VGG16 CNN, were fed into the encoder using one LSTM layer. Then, the decoder in the form of another LSTM layer was implemented for video caption generation. In general, 12 models were trained for two datasets with various number of frames per video and vocabulary size. The best performing model, which was trained on MSVD dataset with 16 frames per video and 2000 vocabulary size, scored 12.8, 32.2, 32.9, and 44.0 on METEOR, BLEU1, ROUGE, and CIDEr evaluation metrics respectively.
