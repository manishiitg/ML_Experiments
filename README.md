# Deep Learning Experiments

This repository will contain code's mainly related to my personal experiances while learning ML and Deep Learning. All will contain various examples and learning. 

This repo is mainly for myself to recollect and also share my journey and experiances with Deep Learning.

=====

## Recruit System Project ##

###### Word2Vec Embedding on Candidates Database ######
Trained a word2vec model on actual candidate data to learn about mainly about skills section of candidates resume. 
[Word2Vec Training](https://github.com/manishiitg/ML_Experiments/blob/master/recruit/word2vec_recruit.ipynb)
Overall i find the results very good and i should use it on production system, especially for skills. 
Also used magnitute and faiss both are good get effecient results with word vectors. 
But word2vec give good results in sense of that if search "seo" keyword, it will give other skills related to seo properly.
So if on the ui level i need to show different skills or find related skills i can do it well. But i don't this this is a good embedding model for down steam nlp task as this doesn't generalize properly.


###### FastText Embedding on Candidates Database ######
Trained fastText embedding on the same candidate database.
[FastText Training](https://github.com/manishiitg/ML_Experiments/blob/master/recruit/fasttext_recruit_training.ipynb)
Mainly in this learned what exactly is fastText, its an extension of word2vec but with n-gram model so that it can generalize better.
This can be used for nlp tasks further downsteam i guess, but need to look at newer embedding likes elmo, bert etc 
Not sure if this will be used anyware in the project yet.

###### Glove Embedding on Candidates Database ######
Trained glove embedding on the same candidate database
[Glove Training](https://github.com/manishiitg/ML_Experiments/blob/master/recruit/glove_recruit_training.ipynb)
Traing Glove on the same dataset, i think results are better than fastText prima facie. Glove uses global concurance matrix so predictoins are better.. 
again not sure if this will be on live tasks, but good to see the results

##### Magnitude/Faiss/Annoy #####

Used the 3 libraries in the above tasks, when playing around with word embeddings.
- Magnitude basically is useful toolkit which works on top of embeddings like word2vec, glove, fasttext. Main advantage is that its fast and also provides are unified interface when dealines with the above said vectors to find similarity etc. Overall i like it.
- Faiss is facebook library in C++ to manage vectors. This is good at searching, similarity, doing PCA, clustering as well and supports many different kinds of indexes. Its a bit complex library and should be used when need really effecient results as it can manage very large indexes like upto 1b vectors.
- Annoy is a library by spoitfy in C++ again to manage vectors, but this only has searching feature i.e similarity. This is a very simple and straight forward library but good at what it does. Magnitude uses this internally. If similary is all that is needed, go with annoy as its very simply and we can build indexes once and save to disk as well. 

##### FastText Text Classification #####
(FastText Classify)[https://github.com/manishiitg/ML_Experiments/blob/master/recruit/fasttext_text_classify_cv_recruit.ipynb]
Purpose of this to setup a baseline and see how fastText doesn't classification for labels. This was just a leanring experiment to set how is the data we have gathered till now.
Overall results were fine nothing great, but i think they are overfitting. Need a better generallized model and a bigger dataset...
This is was just a simple experiment, need to go with better models for documents. 

##### Facebook Starspace #####
(Starspace Experiments)[https://github.com/manishiitg/ML_Experiments/blob/master/nlp/facebook_starspace_experiments.ipynb]
Just playing around with this library, but i don't think its good enough. It say it can do a lot of things, but i couldn't do much with it and didn't understand it well as well. Didn't get any conclusive results, i think again its just to test out ur data and get a baseline for results. As training times a very fast.

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

Attention is very important concept in NLP and has lead to Transforms instead of RNN. https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
This provides are very good introduction at code level to attention 

- Encoder remains the same, except its using bi-direction GRU. 
- I find this interesting
hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
the last two hidden layers are concatinated and passed to a FF netework and  then passed to tanh
so this would basically activate only specific nodes on the source hidden layer. 
shouldn't this be just sum or average? hmmm...

- Next, what we have is hidden states are passed to the RNN and also context is passed same as before. 
- But now the decoder calculates attention from the source sequence. attention is basically means to which word from the source sequence should the neural network pay attention to while transalation. As in language, based on certain specific words the entire meaning of sentance would change. 
- In short attension can be a probablity distribution of the source sequence words and higher probablity of a word will result better translation. 
- So to calculate attention what is done is to 
 - hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)     this is basically to repeat the hidden state of decoder so that is the same length as source 
 - First, we calculate the energy between the previous decoder hidden state and the encoder hidden states. As our encoder hidden states are a sequence of $T$ tensors, and our previous decoder hidden state is a single tensor, the first thing we do is repeat the previous decoder hidden state $T$ times. We then calculate the energy, $E_t$, between them by concatenating them together and passing them through a linear layer (attn) and a $\tanh$ activation function.
    $$E_t = \tanh(\text{attn}(s_{t-1}, H))$$
   energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
  This can be thought of as calculating how well each encoder hidden state "matches" the previous decoder hidden state.

 - Next, we have another parameter called v 
       v = self.v.repeat(batch_size, 1).unsqueeze(1)
       attention = torch.bmm(v, energy).squeeze(1)
   
   We can think of this as calculating a weighted sum of the "match" over all dec_hid_dem elements for each encoder hidden state, where the weights are learned (as we learn the parameters of $v$).

 - Finally we do a softmax, Finally, we ensure the attention vector fits the constraints of having all elements between 0 and 1 and the vector summing to 1 by passing it through a $\text{softmax}$ layer.

- In the decoder everything is same, except that instead of using the context vector directory, a weight sum is used based on the attention. 
  weighted = torch.bmm(a, encoder_outputs)
  #weighted = [batch size, 1, enc hid dim * 2]
  weighted = weighted.permute(1, 0, 2)
  rnn_input = torch.cat((embedded, weighted), dim = 2)
  
  
- So this means the decoder now just doesn't have the context vector, but it also as attention of the source sequence so decoder can pay attention to specific words

- This model increases the accuracy, but also increases the training time a lot.

This blog post, shows the concept of attention very well https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/


Attention is All You Need

This is very complex and need to dig through this will take time. Will get back to this. but basics are clear for now 

https://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/
http://nlp.seas.harvard.edu/2018/04/03/attention.html
http://jalammar.github.io/illustrated-transformer/

https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb
https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb



