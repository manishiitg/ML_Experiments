# Deep Learning Experiments

This repository will contain code's mainly related to my personal experiances while learning ML and Deep Learning. All will contain various examples and learning. 

This repo is mainly for myself to recollect and also share my journey and experiances with Deep Learning.

=====

### NLP / Deep Learning ###
Learning NLP through deep learning
- ##### Pytorch #####
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
