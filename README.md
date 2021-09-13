# Siamese-Network
<b>This project was completed by Patrick, Ian and Ariel.</b>

<h2>INTRODUCTION</h2>
A neural network can be trained to learn equivalence relations between objects. For this experiment, the Siamese network will be using two or more Convolutional Neural Network (CNN) as its subnetwork component. The network will be trained using the Omniglot dataset to predict whether the given inputs belong to the same equivalence class. 

<h2>EXPERIMENT STRUCTURE</h2>
This experiment has been conducted based on the Siamese network in which several CNNs are combined. The Experiment has been conducted in the Google Colab environment. The running environment for experiment includes Python version 3.6.9, tensorflow version 2.3.0 and keras version 

<h2>DATA PROCESS</h2>
The omniglot dataset is loaded and split into training and test datasets. Data used for training were only taken from the ‘train’ split.
The three datasets for testing are:
<ol>
  <li>Data from the ‘train’ split</li>
  <li>Data from both the ‘train’ and ‘test’ split</li>
  <li>Data from the ‘test’ split</li>
</ol>

<h2>TEST DESIGN</h2>
In order to test the performance of the Siamese network model generated, three experiments for two different losses, contrastive loss and triplet loss, are conducted. The three experiments used different pairs as input of the model:
<ul>
  <li>First experiment used the pairs from the set of glyphs from the ‘train’ split.</li>
  <li>Second experiment used the pairs from the set of glyphs from both ‘train’ and ‘test’ splits.</li>
  <li>Third experiment used the pairs from the set of glyphs from the ‘test’ split.</li>
</ul>
Each experiment followed the performance evaluation method to observe the calculated losses and accuracies of training and validation. Additionally, a callback function is applied to avoid overfitting of the model, which is set to stop the experiment if the validation loss does not change for twenty epochs.

<h2>MY CONTRIBUTION</h2>
<h3>create_triplets function</h3>
To test the network using triplet loss function, three inputs are required. An anchor sample, positive sample and negative sample. Two images of the same class were used for the anchor and positive sample. An image of a different class was used as the negative sample.

<h3>build_CNN_model function</h3>
For this experiment, four 2D convolutional layers are used. After each layer, batch normalisation is applied to maintain the output between 1 and 0. Max Pooling layers were used after the second and fourth convolutional layer for downsampling the input. Lastly, the network is connected by two fully connected layers. Euclidean distance is used to measure the distance between the two outputs from the CNN model with contrastive loss function. RMSprop is used as an optimizer for the model.

<h3>siamese_network_triplet_loss function</h3>
To build the Siamese network using triplet loss function, 3 CNN models are used for the anchor, positive and negative images. Returns a Siamese network model.
