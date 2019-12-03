# Deep Learning Experiments

This repository will contain code's mainly related to my personal experiances while learning ML and Deep Learning. All will contain various examples and learning. 

This repo is mainly for myself to recollect and also share my journey and experiances with Deep Learning.

=====

### NLP / Deep Learning ###
Learning NLP through deep learning
- ##### Pytorch #####
  - ###### Basics ######
    - [Very Simple RNN](nlp/Name_Generator_Basic_RNN_Demo_Learning_First_RNN.ipynb) Creating a very Simple Character-Level RNN network 
    - [Character RNN Generate Indian Names](nlp/RNN_Indian_Name_Generator.ipynb) Generate random indian names getting very good results     - [Gender Specific Name Generation](nlp/RNN_Indian_Name_Generator_Gender_Specific.ipynb) This is a slight variation of the previous name generator, in this we can generate names for specific generic using a single NN
    - [Character Level RNN For Text Generation Paul Grahm Essays](nlp/LSTM_Character_Level_RNN_Model_Paul_Graham_Essay_Generator.ipynb)
  - [Simple Text Classification](autoencoder/pytorch_lstm_getting_started_classification.ipynb)
  - [Very Simple Bidirectional LSTM](nlp/pytorch_improved_lstm_bidirectional_and_2_layer.ipynb) Simple LSTM based network for text classification 
  - [LSTM with Golve Vectors](nlp/pytorch_improved_lstm_bidirectional_and_2_layer.ipynb) Better accuracy with glove vector and 2 layer lstm
- ##### Keras #####
  - [Golve Embedding with Keras CNN/LSTM](nlp/golve_vectors_with_keras_deep_nets.ipynb) Got good results using GloVe vectors with LSTM but it takes time to train the network. *Should come back to above later and test this with more relevent data*
  - [Spacy Embedding with Keras](nlp/spacy_vectors_with_keras_deep_nets.ipynb) Just trying out spacy embedding instead of GloVe. Altough spacy uses GloVe internally, it's just easier to use just a single library.  

### AutoEncoders ###

I was looking at different ways to be able cluster data using unsupervised learning. During this i encountered auto encoders
to reduce dimensionality of data (which could be used for clustering) and also came onto VAE which can be used to generate content.
In conclusion i found this not very effective but its good to get an understand of things and see some interesting applications.
We can use GAN's to achieve SOTA results with images and models like GPT2 for text generation. 
But auto-encoders are good concepts to understand and learn nonetheless. Would like to come back to this topic later on after explanding my knowledge futher with Deep Learning and Clustring. 

- ##### Image Based ######
  - [Keras AutoEncoder with DEC Clustering](autoencoder/auto_encoder_experiment_with_keras_with_clustering_DEC.ipynb) almost 25% improvement with clustering with auto encoders. experiment further with DEC - [ ] get back on this need to understand DEC better 
  - [Simple Autoencoder with pytorch](autoencoder/pytorch_basic_experiment_autoencoder_myself.ipynb) Nothing much done on this, just for learning pytorch 
  - [VAE Autoencoder CNN Pytorch](autoencoder/VAE_ecommerce_CNN_pytorch.ipynb)
  - [VAE CNN Autoencoder Ecommerce Data](autoencoder/VAE_ecommerce_pytorch_64x62_2_layer_ConVV_but_very_slow.ipynb) this is with images of size 64x64 and with 2 layers. this is quite slow to optmize and train the network. need a better rig to explore this. 
  - [VAE Autoencoder with Celeba Facedata](autoencoder/VAE_Experiment_with_Face_Data.ipynb) Got good results with this and also attached pre-trained model 
  - [VAE Conditional with MNIST](autoencoder/CVAE_Experiment_MNIST.ipynb)
  - [VAE DEC with Celeba Facedata](autoencoder/VAE_DFC_Experiment_with_Face_Data.ipynb)
  - This is just the start of autoencoders there is more like beta veta, factor vae, adversarial vae and which leads into GAN. I have done some basic experiments into beta-veta but need to pause and come back to this. there is a lot of learn in this and experiment but right now need to get back to project work related to nlp. But for sure need to back on this in between to understand this more 

- ##### Text Based #####
  - [LSTM Based Autoencoder](autoencoder/pytorch_autoencoder_for_nlp_with_lstm_+_cnn.ipynb) Experiementing with this but its unsolved, need to come back to this. -[] Incomplete 
  
- ###### Userful Links to Understand AutoEncoders ######
  Few good articles i found
  - https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
  - https://www.jeremyjordan.me/autoencoders/
  - https://www.jeremyjordan.me/variational-autoencoders/
  - https://wiseodd.github.io/techblog/2016/12/03/autoencoders/
  - https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
  - http://anotherdatum.com/vae.html
  - http://anotherdatum.com/vae2.html
  - http://ruishu.io/2017/01/14/one-bit/
  - http://krasserm.github.io/2018/07/27/dfc-vae/
  - https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html
  - https://ijdykeman.github.io/ml/2016/12/21/cvae.html
  - https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/

### Pytorch ###
- [Pytorch 101](pytorch/101/pytorch_getting_start_LR_101.ipynb) Simple linear regression 
- [Pytorch Basics Part2](pytorch/101/pytorch_getting_started_part2.ipynb) Doing simple linear regression 
- [Pytroch Simple NN on MNIST Dataset](pytorch/101/pytorch_basics_LR_on_MNIST_Dataset.ipynb)
- [Pytorch Simple CNN on MNIST Dataset](pytorch/101/Pytorch_Simple_CNN_on_MNIST.ipynb)
- [Pytroch Simple CNN on CIFAR10 Dataset](pytorch/101/pytorch_cnn_experiment_basics.ipynb)

### Keras ###
I found keras easy to start with when working with M/L and understand basics of neural networks. 
- [Keras 101](keras/Keras_First_Neural_Net_101.ipynb)
- [Keras Simple CNN and VGG16 Transfer Learning](keras/keras_very_simple_cnn_and_vgg16_transfer_learning.ipynb) Taken images from flipkart to classify images between tshirt/jeans
- [Keras NLP Text Classification With Simple FF Network Word2Vec + TF-IDF](keras/simple_FF_deep_learning_classifier_with_word2vec_embedding_+_TF_IDF.ipynb) Using a very simple NN to expriment text classification and using word2vec embeddings
- [LSTM/CNN Text Classification using word2vec](keras/deep_learning_word2vec_model_with_CNN,_LSTM.ipynb) Trying out LSTM/CNN modules to classify text 

### ML ###
Various expriments with Data Science and ML
- [PCA/t-Sine Data Visualization](ml/PCA,_t_sine_data_visualization.ipynb)
- [Logistic Regression for Text Classification](ml/logistic_regression_with_word2vec.ipynb) Using sklearn for LR and word2vec for embeddings together with Mean/TF-IDF
- [SkLean Classifiers Experiments SVC/SGD/Naive](ml/ski_lean_ml_classifier_naive_bayes_random_forest_svc_sgd.ipynb)

### NLP Basics ###
Understanding basics of NLP
- [NLP Basics](nlp/101/nlp_getting_started.ipynb) 
- [Bag of Words](nlp/101/count_vectorize_bag_of_words.ipynb)
- [TF-IDF](nlp/101/tf_idf_experments.ipynb)
- [Word2Vec](nlp/101/word2vec_experiments.ipynb)  Good results to understand word2vec
- [Word2Vec Visualize Data](nlp/101/word2vec_experiments_plotting_and_data_visualization.ipynb) Plotting word2vec data to understand

#### Bayesian vs Classical (Frequentist) Statistics ####
- https://planspace.org/2013/11/11/whats-the-difference-between-bayesian-and-non-bayesian-statistics/
- https://www.analyticsvidhya.com/blog/2016/06/bayesian-statistics-beginners-simple-english/

#### Bayesian Regression ####
- https://katbailey.github.io/post/from-both-sides-now-the-math-of-linear-regression/
- https://wiseodd.github.io/techblog/2017/01/05/bayesian-regression/
- Whats Regularization / Ridge Regression https://towardsdatascience.com/regularization-in-machine-learning-connecting-the-dots-c6e030bfaddd

#### Bayesian Inference ####
- https://blogs.oracle.com/datascience/introduction-to-bayesian-inference
- https://towardsdatascience.com/from-scratch-bayesian-inference-markov-chain-monte-carlo-and-metropolis-hastings-in-python-ef21a29e25a
- https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50



## Seq2Seq ##
Seq2Seq its a very important part of NLP and its very important to understand how it works. I will try to expirement with different Seq2Seq models going to upto Transformers so as to understand deeply its usage.

I will be closely following the posts here https://github.com/bentrevett/pytorch-seq2seq and mainly trying to reproduce them.

#### My Notes ####
This is the first Seq2Seq model i tried the most simple one https://colab.research.google.com/drive/1q88JjGC7xeRLuuoN8ZGpVu90iR2b2-E1
My take aways from this 

- First of language modelling is basically training a neural network to convert one sequence to another sequence. In general this can be any kind of sequence, but in this specific case this is for transalation i.e converting from one language to another. 

- The most simplest model have an Encoder model and Decoder. 
- Encoder is an LSTM which takes input as a full sentence. This is can be multiple layers or single layers as well. 
- Encoder is passed the entire source sequence and it passes through a RNN. Encoder is mainly used to get the hidden state and cell state. 
- These learned hidden/cell states are passed to the decoder. 
- Decoder uses these hidden/cell states to start with, i.e the decoder RNN is initialized. Intuitively, this means the main purpose of the encoder to learn about the source sentence and pass that learning to the decoder. Quite interesting approach, if you think about it !!
- Decoder works one word at time and not on the full sequence unlike the enoder. 
- Decoding process starts with <sos> i.e the first token given to decoder and it prediects what will be the next token. 
- The next token predicted is then used again as the input to decoder in loop till we loop through all the target tokens. 
- There is an exception here, based on the forced teaching ratio either the predirect token is used or the target token is used as the next input. 
- All the final outputs from the decoder RNN are collected and return backed. 
- Seq2Seq model is simply a combination on encoder/decoder
  
Part2 of this uses the model https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb

- Encoder remains the same
- Decoder changes, i.e now we are passing a new variable called "context" to the decoder rnn along with hidden states
- Context is nothing but the hidden state from encoder, we just don't update it after every decoder iteration
- Why this is done? In the previous model decoder at every step didn't the data of the source sentence. As after every iteration we update the hidden states. This means the hidden states of the decoder have learn or keep in "memory" the source information as well as need to learn how to decode. To remove this, if we always pass the source information or "context" to the decoder always, the decoder hidden states no long have to learn about the source, rather they can optimize only to learn the decoding process. 
- This is again quite interesting! We have logic that we want to implement which is decoder shouldn't learn about the source sentence, rather just the decoding process. To implement this we change our model parameters and force the model to learn this... We don't tell the model what to do explicitely but rather allow the model to learn based on what information we pass to the model..

Part3: Attention
...
