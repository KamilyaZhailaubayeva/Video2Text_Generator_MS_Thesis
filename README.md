<h1 align="center">
 Surveillance Video-to-Text Generator 
</h1>

Video-to-text generation is a relatively new field which is recently gaining a popularity as other generative models. It is particularly beneficial for surveillance to retrieve textual descriptions from CCTV cameras. Thus, this thesis aims to design an efficient video-to-text generator for surveillance. 

The research established that encoder-decoder neural networks are the most up-to-date technique. Specifically, pretrained CNN models and the LSTM based encoder and decoder models are the most prominent neural networks. Though the research was initially intended for surveillance videos, the largest MSR-VTT and MSVD datasets with general videos were used for training due to the absence of available surveillance dataset with captions. 

The models were designed in four parts: feature extraction, model training, test caption generation, and model evaluation. Video features, which were extracted using pretrained VGG16 CNN, were fed into the encoder using one LSTM layer. Then, the decoder in the form of another LSTM layer was implemented for video caption generation. 

In general, 12 models were trained for two datasets with various number of frames per video and vocabulary size. The best performing model, which was trained on MSVD dataset with 16 frames per video and 2000 vocabulary size, scored 12.8, 32.2, 32.9, and 44.0 on METEOR, BLEU1, ROUGE, and CIDEr evaluation metrics respectively. Therefore, MSVD dataset is the most suitable for the designed architecture. Furthermore, it was found that increasing number of frames per video was not legitimate in terms of computational resources for short videos between 10 to 30 seconds. Finally, 2000 vocabulary size is an excellent size for MSVD dataset. Though the proposed model generated captions, it performed worse than the past research in terms of evaluation metrics. This might be caused by the computational limitation, inaccurate caption datasets, and improper selection of the search algorithm.

## Methodology
### Dataset Collection
Dataset of the video-to-text generator includes both video dataset and its annotations. Thus, it is required to ensure that dataset has both videos and their descriptions. Most researchers investigating video-to-text generators have utilized [MSR-VTT](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf) and [MSVD](https://www.microsoft.com/en-us/research/publication/collecting-highly-parallel-data-for-paraphrase-evaluation/) datasets. MSR-VTT is the best and largest dataset dedicated to video-to-text generators collected from a commercial video search engine. 200,000 descriptions are available for 10,000 videos each with 10-30 seconds duration. In addition, Microsoft Research Video Description Corpus (MSVD) dataset is the first open dataset in multiple languages that was manually collected from YouTube. It is dedicated to both video-to-text and text-to-video generations. In this dataset, 1970 videos with 10-25 seconds duration have on average 41 descriptions. 1970 videos of MSVD dataset are mostly divided to 1,200 for training, 100 for validation, and 670 for testing throughout the literature. MSR-VTT dataset, which has 10,000 videos in total, is already split into 7,010 for training and validation, and 2,990 for testing. Some authors divided 7,010 videos to 6,513 for training and 497 for validation. No other datasets are utilized this extensively, thus MSR-VTT and MSVD datasets are the most suitable datasets for this research.

### Training parameters and Preprocessing
Video-to-text generating model is designed in Google Colaboratory with the runtime type: A100GPU and high-RAM of 80 GB. Video frames are obtained and resized using OpenCV’s “cv2” library. Moreover, pretrained CNN VGG16 model, LSTM layers for encoder and decoder along with “TextVectorization” for caption tokenization are leveraged by “tensorflow.keras” library. Video features are extracted in “.npy” format, while models are in “.h5” and test caption results are in “.json”. Due to computational limitations, the batch size is 128 for feature extraction and 256 for training, and training epochs are 150. Furthermore. ADAM optimizer is used for training. Learning rate is varied according to training results: 0.0001-0.0005. Using static learning rate is not sufficient for training, thus it is reduced by factor of 0.1 or 0.5 every time validation loss decreases for 5 consecutive epochs. Finally, a callback is used for saving models with best validation accuracy defined by an adjustable threshold.

Preprocessing of data is required to speed up the training process and to effectively work with the dataset. Firstly, video data would be preprocessed. The dimension of video frame would be scaled to 224x224x3, since it is demanded input size by VGG16 model. Throughout the literature different number of video frames are utilized. Though number of video frames and batch size should be as large as the computing machine can allow, Olivastri et al. state that 16 frames from each video is sufficient. Therefore, two types of number of frames, specifically 16 and 80 frames, are leveraged in this research to analyze how video frame size will affect overall results.

Apart from the video, annotations should be preprocessed as well. Some basic preprocessing like converting to lower case, removing punctuations, etc. is realized. The first token of each annotation is <BOS> (Beginning Of the Sentence). The last token of each sentence is <EOS> (End Of the Sentence). There is also a token <UNK> (Unknown) for words which appear in the whole textual data less than 5 times. Thus, <BOS> and <EOS> tokens are inserted at the beginning and ending of each caption respectively. After this, captions with length 6-10 and 6-13 words are collected for MSVD and MSRVTT datasets respectively to avoid possible errors occurred because of too lengthy or too short captions. Moreover, a tokenization of words should be carried out. Since model is trained by “tensorflow.keras” library, its “TextVectorization” tokenizer is implemented in this research. Also, each word in annotations should be fed into the model as one-hot vector, hence captions are padded to reach a common size, which is 10 in this case, by built-in TextVectorization’s padding. Finally, only training captions are used to create a vocabulary for TextVectorization to avoid a bias during designing process of this tokenizer. There are about 29,000 (MSRVTT) and 12,000 (MSVD) unique words in caption dataset, hence it is essential to analyze various vocabulary size. Therefore, three different vocabulary sizes are chosen: 1500, 2000, and 2500. Finally, a total number of trained models depending on the dataset, number of frames, and vocabulary size are 12.

### Evaluation metrics
The vast majority of studies on video-to-text generators have analyzed a quality of their method by four evaluation metrics, namely [METEOR](https://www.researchgate.net/publication/270878844_Meteor_Universal_Language_Specific_Translation_Evaluation_for_Any_Target_Language), [BLEU](https://www.researchgate.net/publication/2588204_BLEU_a_Method_for_Automatic_Evaluation_of_Machine_Translation), [CIDEr](https://www.researchgate.net/publication/268689555_CIDEr_Consensus-based_Image_Description_Evaluation), and [ROUGE](https://www.researchgate.net/publication/224890821_ROUGE_A_Package_for_Automatic_Evaluation_of_summaries). All of them are machine translation evaluation metrics. They indicate better results with a higher score. Finally, METEOR, BLEU, CIDEr, and ROUGE evaluation metrics can be utilized in the analysis of the video-to-text generator.

## Model Designing
### Feature Extraction
The first part of the model designing process is the feature extraction. The code for this part is [here](model_parts/part1_feat_extraction_MSRVTT.ipynb). Video features are the input of the model’s encoder, hence features must be extracted from the video before training. The code execution for this part requires high computational resources and takes the longest time (6-8 hours). VGG16 CNN, which was trained on the large ImageNet dataset, is leveraged to extract features from each video. For this, the last layer (“predictions”) of the model, which contains 23 layers overall, is eliminated. The last two layers of the VGG16 model is depicted on Figure 1. Therefore, the shape of the extracted features is (16, 4096) and (80, 4096) for 16 and 80 frames extractions respectively. Subsequently, the number of encoder tokens (num_enc_tokens variable) is 4096. Overall, there are 4 different sets of features in this research: features of MSVD video dataset with 16 and 80 frames and MSRVTT video dataset with 16 and 80 frames.

<p align="center">
  Figure 1. The last two layers of VGG16 model.
</p>

<p align="center">
  <img width="250" src=Images/Figure1.png>
</p>

### Model Training
The second part of the model designing process is the training. The code for this part is [here](model_parts/part2_train_MSRVTT16_2000.ipynb). Caption preprocessing takes place in this part along with saving trained models. The model consists of encoder and decoder parts where encoder is CNN-LSTM and decoder is LSTM. Thus, both models leverage one LSTM layer each. The block diagram of LSTM cell used in this study is shown in Figure 2. This type of LSTM is based on [this researh](https://arxiv.org/abs/1410.4615). There are three inputs: memory state c, hidden state h, and input x which is a hidden state from the previous layer. In order to obtain two outputs c and h, following four gates are required: forget gate f, input gate i, output gate o, and memory update c. The former three gates leverage a sigmoid nonlinear function σ, while the last one uses a hyperbolic tangent function. The bias is b and the weight is W. The outputs are obtained by element wise multiplication denoted by ⊙.

<p align="center">
  Figure 2. The block diagram of LSTM cell.
</p>

<p align="center">
  <img width="450" src=Images/Figure2.png>
</p>

The structure of the overall model is illustrated on Figure 3, while the summary with a total number of parameters is in Figure 4. The input of the encoder (“enc_inputs” layer in Figures 3 and 4) with the shape (16, 4096) is extracted features from the first part of the model designing process. Its output is the first input (“enc_lstm” layer) of the decoder LSTM layer (“dec_lstm” layer) which also takes preprocessed captions with the shape (10, 2000) as the second input (“dec_inputs” layer). Therefore, the number of decoder tokens (“num_dec_tokens” variable), which is a vocabulary size, is 2000 and the time steps of decoder (“time_steps_dec” variable), which is a number of generated words in one caption, is 10 in this structure. The total number of different structures are 6 depending on the number of frames and vocabulary size: “16:1500”, “16:2000” depicted in Figure 3, “16:2500”, “80:1500”, “80:2000”, and “80:2500”. Since there are two datasets, total number of trained models are 12. Both LSTM layers have a latent dimensionality equal to 512 and the hyperbolic tangent activation function (“tanh”). The output of the decoder LSTM layer is an input of the dense layer (“dec_relu” layer) with the activation function “ReLU” which stands for rectified linear unit. As a result of this layer, next possible words of captions are generated. During training, encoder and decoder are trained together as one model, however the trained model is split into encoder and decoder parts to save models after training. The structure and summary with parameters of the encoder model are shown on Figures 5 and 6, while the decoder model is on Figures 7 and 8 respectively. Finally, the vectorizer for each vocabulary size is also saved along with encoder and decoder models.

<p align="center">
  Figure 3. The overall bottom-up structure of the model.
</p>

<p align="center">
  <img width="450" src=Images/Figure3.png>
</p>


<p align="center">
  Figure 4. The overall summary of the model.
</p>

<p align="center">
  <img width="450" src=Images/Figure4.png>
</p>


<p align="center">
  Figure 5. The encoder structure of the model.
</p>

<p align="center">
  <img width="450" src=Images/Figure5.png>
</p>


<p align="center">
  Figure 6. The encoder summary of the model.
</p>

<p align="center">
  <img width="450" src=Images/Figure6.png>
</p>


<p align="center">
  Figure 7. The decoder structure of the model.
</p>

<p align="center">
  <img width="450" src=Images/Figure7.png>
</p>


<p align="center">
  Figure 8. The decoder summary of the model.
</p>

<p align="center">
  <img width="450" src=Images/Figure8.png>
</p>

### Test Caption Generation
The third part of the model designing process is a caption generation of the test dataset for all trained models from previous part. The code for this part is [here](model_parts/part3_MSVD16_1500.ipynb). The output of this part is a “json” file with the video ID and its caption for each test video. Every trained model’s encoder, decoder, and vectorizer from the previous part are loaded in this part. Also, video features of test dataset collected from the first part are obtained as well. Greedy Search Algorithm was implemented to find next word of the caption for each test video. For this, test video feature is encoded firstly. Simultaneously, the <BOS> token of the caption is defined. After that, the first iteration starts by obtaining next possible words from decoding the output of the encoder. The next word in caption is the word with the maximum probability. At the same step, the output of the decoder is saved, since it is an input of the next word generation’s decoder. This process is continued until <EOS> token is generated. Finally, resulting caption along with its video ID are recorded.

### Model Evaluation
The final part of the model designing process is a model evaluation by relevant machine translation metrics such as METEOR, BLEU, ROUGE, and CIDEr. Firstly, actual test captions from both datasets are separated in corresponding “json” files for convenience. The code is [here](model_parts/part4_1_dataset_format.ipynb). Then, these actual test captions and generated test captions from the third part are acquired to calculate evaluation metrics using “pycocoevalcap” library. [This library](https://github.com/tylin/coco-caption/tree/master) is initially designed to evaluate Microsoft COCO image captions. The code for evaluation is [here](model_parts/part4_MSVD16_1500.ipynb). There are multiple actual test captions for each test video, thus only actual captions, which have a similar size to corresponding generated captions, are selected. Finally, each metrics have two outputs: 1) metric score through all test captions; 2) metric scores of each caption separately.

## Results
There are four parts of video-to-text generation: feature extraction, model training, test captions generation, and evaluation of these captions by relevant metrics. At the first part, features for all MSVD (1,970) and MSRVTT (10,000) video datasets for 16 and 80 frames were extracted using pretrained VGG16 CNN model. Then, data was split into 1,200 train videos, 100 validation videos, and 670 test videos for MSVD dataset and 6,513 train videos, 497 validation videos, and 2,990 test videos for MSRVTT dataset. At the second part, the caption datasets were loaded and divided according to the data split above. There were 36,502 train captions and 3,237 validation captions for MSVD dataset and 99,777 train captions and 7,666 validation captions for MSRVTT dataset. The results of training each dataset with different number of frames and vocabulary sizes are summarized in Table 1, where epochs trained out of supposed epochs, selected model’s epoch, starting learning rate LR1, learning rate at the selected epoch LR2, its corresponding loss, accuracy, validation loss, and validation accuracy are given. Finally, the model training’s accuracy, loss, and learning rate vs. epoch plots are shown on Figure 9, Figure 10, and Figure 11 respectively.

<p align="center">
  Table 1. The training results of all models.
</p>

<p align="center">
  <img width="450" src=Images/Table1.png>
</p>


<p align="center">
  Figure 9. Model accuracy vs. epoch plots. a: MSVD16:2000; b: MSVD80:2500; c: MSRVTT16:1500; d: MSRVTT80:2000.
</p>

<p align="center">
  <img width="450" src=Images/Figure9.png>
</p>


<p align="center">
  Figure 10. Model loss vs. epoch plots. a: MSVD16:2000; b: MSVD80:2500; c: MSRVTT16:1500; d: MSRVTT80:2000.
</p>

<p align="center">
  <img width="450" src=Images/Figure10.png>
</p>


<p align="center">
  Figure 11. Model learning rate vs. epoch plots. a: MSVD16:2000; b: MSVD80:2500.
</p>

<p align="center">
  <img width="450" src=Images/Figure11.png>
</p>

As a result of the third part, test captions of all models were generated. These captions along with actual test captions of both datasets were used to calculate evaluation metrics in the fourth part. The results of METEOR (M), BLEU1 (B1), BLEU2 (B2), BLEU3 (B3), BLEU4 (B4), ROUGE (R) and CIDEr (C) metrics for all models are listed in Table 2. The numerical results are multiplied by 100 for comparison purposes. Successfully generated captions for the best performing model, which is the MSVD dataset with 16 frames per video and 2000 vocabulary size, are demonstrated in Table 3. For all evaluation metrics, these captions achieved a maximum which is 100.0 for METEOR, BLEU, and ROUGE metrics and 1000.0 for CIDEr metric. For the video “k-SWy-sU8cE_5_10.avi” from Table 3, a snippet is illustrated in Figure 12. Poorly generated captions for the best performing model are given in Table 4. Moderately generated captions as well as their evaluation results for this model are given in Table 5, where video ID1 = “kWLNZzuo3do_48_53”, ID2 = “lfGlDg47How_93_98”, ID3 = “nS6oQxX_Qi8_2_12”, ID4 = “qeKX-N1nKiM_37_43”, ID5 = “xxHx6s_DbUo_49_56”, and ID6 = “zFIn8DeV5PM_20_33”. 

<p align="center">
  Table 2. The results of evaluation the models.
</p>

<p align="center">
  <img width="450" src=Images/Table2.png>
</p>


<p align="center">
  Table 3. Successfully generated captions for the MSVD16:2000 model.
</p>

<p align="center">
  <img width="450" src=Images/Table3.png>
</p>


<p align="center">
  Table 4. Poorly generated captions for the MSVD16:2000 model.
</p>

<p align="center">
  <img width="450" src=Images/Table4.png>
</p>


<p align="center">
  Table 5. Moderately generated captions for the MSVD16:2000 model.
</p>

<p align="center">
  <img width="450" src=Images/Table5.png>
</p>


<p align="center">
  Figure 12. A snippet of testing video k-SWy-sU8cE_5_10.avi.
</p>

<p align="center">
  <img width="450" src=Images/Figure12.png>
</p>

Apart from the results of generated captions, the statistics of all generated words for MSVD16:2000 and MSRVTT16:2500 models are collected. They are summarized in Tables 6 and 7 respectively, where the second column represents the number of times given word appears in generated test captions with a maximum of 670 and 2990 respectively. The third column shows the number of times the word appears in any actual captions of test videos with the same maximum. The fourth column displays the number of times each word is used in training captions which has a total of 36,502 and 99,777 captions respectively. Finally, the last column shows the word’s order in the vocabulary of a designed tokenizer which has a maximum of 2000 for the first model and 2500 for the second model.

<p align="center">
  Table 6. The statistics of all generated words for the MSVD16:2000 model.
</p>

<p align="center">
  <img width="450" src=Images/Table6.png>
</p>


<p align="center">
  Table 7. The statistics of all generated words for the MSRVTT16:2500 model.
</p>

<p align="center">
  <img width="450" src=Images/Table7.png>
</p>

## Discussion
The training results of all 12 models given in Table 1 demonstrate that accuracy on both training and validation datasets is fluctuated around 0.5 on MSVD and 0.4 on MSRVTT datasets. Initially, all models failed to reach a desired number of epochs due to a high learning rate until it was largely improved by presenting a reducing learning rate. However, the majority of models still got to only half of epochs. Moreover, for models that get to the desired number of epochs as “MSRVTT16:1500” model, validation accuracy still did not increase as shown in Figure 9.a and the learning rate became insignificant by hitting a very insignificant value, 3.0*10^(-21). This pattern is observed in other models despite the fact that they did not reach required number of epochs. Thus, models were not forced to retrain for the appropriate management of computational resources. Resulting accuracy of models suggests that models could generate commonly only half or less portion of the caption. This can be also demonstrated by the results of the third part, where both models were leveraged to generate test captions. Resulting captions for some videos are listed in Tables 3, 4, and 5. Thus, models generally predict the subject of the event and action of the event, while the object of the event cannot be mostly generated. However, the correctness of these predicted captions should be evaluated by metrics as shown in Table 2 where it can be observed that the best performing model is “MSVD16:2000”. The statistics of each word generated on test videos in Tables 6 and 7 suggest that the majority of generated captions consists of “a”, “is”, “man” for MSVD16:2000 model and “a”, “is”, “man”, “talking”, “about” for MSRVTT16:2500 model. This can be explained by the fact that they are the most frequent words in train caption dataset. They are also the most relevant to the test videos.

## Conclusion
In the thesis, the video-to-text generator was modeled based on the state-of-art method termed encoder-decoder neural networks. The encoder is built using CNN-LSTM where CNN is a pretrained VGG16 model that was used for video feature extraction. Captions were generated by another LSTM layer as a decoder. Though, this research aimed to surveillance video-to-text generator, the largest MSR-VTT and MSVD datasets with general videos are used for training due to the absence of available surveillance dataset with annotations. Overall, there were 12 models trained for these datasets with different number of frames per video and vocabulary size. The best performing model, which was determined by METEOR, BLEU, CIDEr, and ROUGE evaluation metrics, is a model trained on MSVD dataset with 16 frames per video and 2000 vocabulary size. Thus, it was found that MSVD dataset is suitable for the designed model. Moreover, 16 frames per video is a sufficient for generating captions of short videos. It is also more efficient in terms of computational resources. In addition, it was discovered that 2000 is the most effective vocabulary size for MSVD dataset, which has a sixth times larger vocabulary. Though, the video-to-text generator produced captions, it failed to succeed the past research in terms of evaluation metrics. This is possibly caused by the computational limitation, improper caption datasets, and search algorithm. Nevertheless, the model can be modified by presenting the soft-attention mechanism in the decoder architecture in the future research.
