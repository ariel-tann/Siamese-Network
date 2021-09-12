'''
------------------------------------------------------------------------------

        IFN680 Assignment2 Siamese Network

             Tan En Hui Ariel, n10497285

             Patrick Choi, n10240501

             Ian ChoiI, n10421106

------------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------------

        Version of tensorflow and keras for this program is
            tensorflow 2.3.0
            keras 2.4.3

        The Experiment was conducted in the Colab environment

------------------------------------------------------------------------------
'''

# Imports modules to complete the Assignment
import numpy as np
from tensorflow import keras
import random

# Imports required modules from the Keras Functional API
import keras
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Lambda, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

# import library to plot the result
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf

'''
------------------------------------------------------------------------------
List of functions in this .py document:

+ accuracy(y_true, y_pred)
+ euclidean_distance(vects)
+ eucl_dist_output_shape(shapes)
+ contrastive_loss_function(y_true, y_pred)
+ calc_triplet_loss(y_pred)
+ get_triplet_loss(y_true, y_pred)
+ omniglot_dataset(ds)
+ preprocess_omniglot_dataset(x_train, y_train, x_test, y_test)
+ create_pairs_set(input_data, label_indices, test_index)
+ create_triplets(input_data, label_indices, test_index)
+ build_CNN_model(input_shape)
+ siamese_network(input_shape, batch_size, training_pairs, training_target,
                    test_pairs, test_target, epochs,verbose)
+ siamese_network_triplet_loss(input_shape, batch_size, training_pairs, training_target,
                              test_pairs, test_target, epochs,verbose)
+ print_accuracy(model)

------------------- END OF LIST OF FUNCTIONS ---------------------------------
'''

### Hyper parameter
batch_size = 128
epochs_siamese = 50


def accuracy(y_true, y_pred):
    '''
    To calculate classification accuracy with a constant threshold on distances.
    Retrieved from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def euclidean_distance(vects):
    '''
    To calculate Euclidean distance
    Retrieved from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    '''
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)

    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    '''
    To return the Euclidean shape
    Retrieved from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    '''
    shape1, shape2 = shapes

    return (shape1[0], 1)


def contrastive_loss_function(y_true, y_pred):
    '''
    To calculate Contrastive loss
    Retrieved from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    '''
    # The margin m > 0 determines how far the embeddings of a negative pair should be pushed apart.
    m = 1  # margin. It can be changed and evaluated for the test of siamese network model.
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(m - y_pred, 0))

    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def calc_triplet_loss(y_pred):
    """
    Calculate Triple loss

    Arguments:
        y_pred -- list containing three objects:
            anchor   -- the encodings for the anchor data
            positive -- the encodings for the positive data (same class to anchor)
            negative -- the encodings for the negative data (different class from anchor)
    Returns:
        loss -- value of the loss
    """
    margin = 0.4  # margin. It can be changed and evaluated for the test of siamese network model.
    anchor = y_pred[0]
    positive = y_pred[1]
    negative = y_pred[2]

    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.maximum(basic_loss, 0.0)

    return loss


def get_triplet_loss(y_true, y_pred):
    '''
    Gets the mean of triple loss value
    '''

    return tf.reduce_mean(y_pred)


def omniglot_dataset(ds):
    '''
    To load the Omniglot data from the tensorflow dataset and to save it in variables.

    Arguments:
        ds -- tensorflow dataset (tensor type)
    Returns:
        x_train --  image from train dataset
        y_train --  label from train dataset
        x_test  --  image from test dataset
        y_test  --  label from test dataset
    '''

    # split into 2 dataset, train dataset and test dataset
    ds_train, ds_test = ds

    def convert_image(data):
        '''
        To convert and resize the image data
        '''

        # get image data
        image = data['image']

        # convert & resize image
        image = tf.image.rgb_to_grayscale(image)  # change RGB to grayscale
        image = tf.cast(image, tf.float32) / 255.
        image = tf.image.resize(image, (28, 28))
        return image

    # convert images in train dataset and test dataset
    ds_image_train = ds_train.map(convert_image)
    ds_image_test = ds_test.map(convert_image)

    # convert labels to numpy arrays
    y_train = np.array([glyph["label"].numpy() for glyph in ds_train])
    y_test = np.array([glyph["label"].numpy() for glyph in ds_test])

    # convert images to numpy arrays
    x_train = np.array([glyph.numpy() for glyph in ds_image_train])
    x_test = np.array([glyph.numpy() for glyph in ds_image_test])

    return x_train, y_train, x_test, y_test


def preprocess_omniglot_dataset(x_train, y_train, x_test, y_test):
    '''
    To make train and test sets for various experiments
    Return total 4 different datasets (one for training & 3 for testing)

    Arguments:
        x_train --  image from train dataset
        y_train --  label from train dataset
        x_test  --  image from test dataset
        y_test  --  label from test dataset
    Returns:
        (data_train, target_train)      --   Dataset for training the model
        (data_train, target_train)      --   Dataset 1 for testing : only training split
        (data_combine, target_combine)  --   Dataset 2 for testing : both splits
        (data_test, target_test)        --   Dataset 3 for testing : only test split
    '''

    data_train, target_train = x_train, y_train
    data_test, target_test = x_test, y_test

    # concatenate train and test data using numpy module
    data_combine = np.concatenate([x_train, x_test])
    target_combine = np.concatenate([y_train, y_test])

    return (data_train, target_train), (data_train, target_train), \
           (data_combine, target_combine), (data_test, target_test)


def create_pairs_set(input_data, label_indices, test_index):
    '''
    To create positive and negative pairs for siam network

    Creates an array of positive and negative pairs combined with their label (1 or 0) -
    depending on if the two images used as input is considered to be from the same equivalence class
    then they are considered a positive pair. If they are not, they are considered a negative pair.
    Adapted from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py

    Arguments:
        input_data     --  Dataset for making pairs for each testing type
        digit_indices  --  An array of label indices
        test_index     --  Index of 1 to 3 depending on the testing type
    Returns:
        Numpy array containing the pairs of images
        Numpy array containing the labels if they are positive (1) or negative (0)
    '''

    # create arrays for pairs and labels
    pairs = []
    labels = []

    # define the range of label index that are in the current dataset from where the pairs are to be created
    if (test_index == 1):
        label_list = list(range(964))
    if (test_index == 2):
        label_list = list(range(1623))
    if (test_index == 3):
        label_list = list(range(964, 1623))

    # every classes in Omniglot dataset have same number of examples, which is 20
    num_samples = 20

    # iterate through the range of label list
    for d in range(len(label_list)):
        for i in range(num_samples - 1):
            # assign values z1 and z2
            z1, z2 = label_indices[d][i], label_indices[d][i + 1]
            # add the z1 and z2 coordinates to the pairs array
            pairs += [[input_data[z1], input_data[z2]]]

            # get a random number between 1 and the num of classes of dataset
            # then find the modulus of d + the new random number
            # divided by the length of the label array and assigns it to the variable dn
            # it is a way to select the data from different classes
            rand = random.randrange(1, len(label_list))
            dn = (d + rand) % len(label_list)

            # assign the values of z1 and z2 and adds them to the pairs array, using the dn variable
            z1, z2 = label_indices[d][i], label_indices[dn][i]
            pairs += [[input_data[z1], input_data[z2]]]

            # add the coordinates 1,0 to the labels array
            labels += [1.0, 0.0]

    # return 2 arrays
    return np.array(pairs), np.array(labels)


def create_triplets(input_data, label_indices, test_index):
    '''
    To create anchor, positive and negative triplet sets for using triplet loss

    Arguments:
        input_data     --  Dataset for making pairs for each testing type
        digit_indices  --  An array of label indices
        test_index     --  Index of 1 to 3 depending on the testing type
    Returns:
        Numpy array containing the triplet sets of images
        Numpy array containing the labels if they are positive with anchor (1) or negative (0) e.g [1,1,0]
    '''

    # create arrays for pairs and labels
    triplets = []
    labels = []

    # define the range of label index that are in the current dataset from where the pairs are to be created
    if (test_index == 1):
        label_list = list(range(964))
    if (test_index == 2):
        label_list = list(range(1623))
    if (test_index == 3):
        label_list = list(range(964, 1623))

    # every classes in Omniglot dataset have same number of examples, which is 20
    num_samples = 20

    # iterate through the range of label list
    for d in range(len(label_list)):
        for i in range(num_samples - 1):
            # it is a way to select the data from different classes
            rand = random.randrange(1, len(label_list))
            dn = (d + rand) % len(label_list)

            # get the label indices of Anchor, Positive and Negative
            t1, t2, t3 = label_indices[d][i], label_indices[d][i + 1], label_indices[dn][i]

            # add the images of Anchor, Positive and Negative
            triplets += [[input_data[t1], input_data[t2], input_data[t3]]]

            # add the coordinates 1,0 to the labels array
            labels += [[1.0, 1.0, 0.0]]

    # returns 2 arrays
    return np.array(triplets), np.array(labels)


def build_CNN_model(input_shape):
    '''
    To build a CNN model to be used as a shared network in the siamese network model.
    CNN practical of week 7 is used as a reference

    Argument:
        input_shape -- The dimenstions of the dataset to be used
    Return:
        model -- A keras Sequential model
    '''

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # fully connected layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1623))
    model.add(Activation('softmax'))

    # return the specified sequential model
    return model


def siamese_network(input_shape,
                    batch_size,
                    training_pairs,
                    training_target,
                    test_pairs,
                    test_target,
                    epochs,
                    verbose
                    ):
    '''
    To build Siamese network combined 2 CNN models

    Argument:
        input_shape      --     dimension of the dataset to be used
        batch_size       --     the size of batch for training
        training_pairs   --     image pair sets for training
        training_target  --     labels for training
        test_pairs       --     image pair sets for testing
        test_target      --     labels for testing
        epochs           --     The num of epochs for siamese network
        verbose          --     To print the process

    Return:
        model -- Siamese network model
    '''

    # use a CNN model as a shared network
    cnn_network_model = build_CNN_model(input_shape)

    # initiate inputs with the same amount of slots to keep the image arrays sequences to be used as input data
    #   when processing the inputs
    image_vector_shape_1 = Input(shape=input_shape)
    image_vector_shape_2 = Input(shape=input_shape)

    # the CNN network model will be shared
    output_cnn_1 = cnn_network_model(image_vector_shape_1)
    output_cnn_2 = cnn_network_model(image_vector_shape_2)

    # concatenates the two output vectors into one using lambda layer
    distance = keras.layers.Lambda(euclidean_distance,
                                   output_shape=eucl_dist_output_shape)([output_cnn_1, output_cnn_2])

    # define a trainable model linking the two different image inputs to the distance
    #   between the processed input by the each cnn model.
    model = Model([image_vector_shape_1, image_vector_shape_2], distance)

    # optimizer for the network model
    rms = keras.optimizers.RMSprop()

    model.summary()

    # compile the model with the contrastive loss function.
    model.compile(loss=contrastive_loss_function,
                  optimizer=rms,
                  metrics=[accuracy])

    # define the callback for early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    # the num of epochs and bath size is defined in the beginning of the document as hyper parameter
    # validate and print the result using the test data
    print()
    hist = model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_target,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=verbose,
                     callbacks=[early_stopping],
                     validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_target)
                     )

    return model


def siamese_network_triplet_loss(input_shape,
                                 batch_size,
                                 training_pairs,
                                 training_target,
                                 test_pairs,
                                 test_target,
                                 epochs,
                                 verbose
                                 ):
    '''
    To build Siamese network combined 3 CNN models for triplets

    Argument:
        input_shape      --     dimension of the dataset to be used
        batch_size       --     the size of batch for training
        training_pairs   --     image pair sets for training
        training_target  --     labels for training
        test_pairs       --     image pair sets for testing
        test_target      --     labels for testing
        epochs           --     The num of epochs for siamese network
        verbose          --     To print the process

    Return:
        model -- Siamese network model
    '''

    # use a CNN model as a shared network
    cnn_network_model = build_CNN_model(input_shape)

    # initiate inputs with the same amount of slots to keep the image arrays sequences to be used as input data
    #   when processing the inputs
    image_vector_shape_1 = Input(shape=input_shape)
    image_vector_shape_2 = Input(shape=input_shape)
    image_vector_shape_3 = Input(shape=input_shape)

    # the CNN network model will be shared for anchor, positive and negarive images
    A = cnn_network_model(image_vector_shape_1)
    P = cnn_network_model(image_vector_shape_2)
    N = cnn_network_model(image_vector_shape_3)

    # Calculates the triple loss value for the anchor, positve and negative input
    tri_loss = Lambda(calc_triplet_loss)([A, P, N])

    # Creates a model that takes three iputs and triple loss value as output
    triplet_model = Model(inputs=[image_vector_shape_1, image_vector_shape_2, image_vector_shape_3],
                          outputs=tri_loss)

    # optimizer for the network model
    rms = keras.optimizers.RMSprop()

    # model.summary()
    triplet_model.summary()

    # compiles the model to use triple loss as its loss function
    triplet_model.compile(loss=get_triplet_loss,
                          optimizer=rms,
                          metrics=[accuracy])

    # the num of epochs and bath size is defined in the beginning of the document as hyper parameter
    # validate and print the result using the test data
    print()
    hist = triplet_model.fit([training_pairs[:, 0], training_pairs[:, 1], training_pairs[:, 2]], training_target,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=verbose,
                             validation_data=([test_pairs[:, 0], test_pairs[:, 1], test_pairs[:, 2]], test_target)
                             )

    return triplet_model


def print_accuracy(model):
    '''
    To print the result of experiment

    '''

    # Plot the the result for first experiment using triplet loss
    train_acc = model.history.history['accuracy']
    val_acc = model.history.history['val_accuracy']
    train_loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']
    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Val acc')
    plt.title('Training and Val accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Val loss')
    plt.title('Training and Val loss')
    plt.legend()

    plt.show()


def contrastive_test1():
    # load the omniglot data as a tensor dataset type
    ds, ds_info = tfds.load(name='omniglot', split=['train', 'test'], with_info=True)

    # load image and label data from dataset
    x_train, y_train, x_test, y_test = omniglot_dataset(ds)

    # the num of classes
    num_classes = ds_info.features['label'].num_classes
    num_classes_trainset = len(set(y_train))
    num_classes_testset = len(set(y_test))

    # Prepocess the data for the training and variour test into 4 different datasets
    (input_trainset, target_trainset), (input_testset1, target_testset1), \
    (input_testset2, target_testset2), (input_testset3, target_testset3) \
        = preprocess_omniglot_dataset(x_train, y_train, x_test, y_test)

    # print information about the initial datasets before converting into pairs
    # in order to demonstrate the amount of images that are used for training and testing respectively
    print("input_trainset: ", input_trainset.shape)
    print("target_trainset: ", target_trainset.shape)
    print("input_testset1: ", input_testset1.shape)
    print("target_testset1: ", target_testset1.shape)
    print()

    # list for labels contained in each dataset
    # in the case of Omniglot data,
    #   label (0~963)    is for train set
    #   label (964~1622) is for test set
    set1_labels = list(range(num_classes_trainset))

    # the specific index used for the different dataset sets of image
    # create image pairs for training the model
    label_indices = [np.where(target_trainset == i)[0] for i in set1_labels]
    training_pairs, training_target = create_pairs_set(input_trainset, label_indices, 1)

    # create image pairs for test1
    label_indices = [np.where(target_testset1 == i)[0] for i in set1_labels]
    test_pairs_set1, test_target_set1 = create_pairs_set(input_testset1, label_indices, 1)

    # print the shape of pair sets
    print("training_pairs shape: ", training_pairs.shape)
    print("training_target shape: ", training_target.shape)
    print("test_pairs_set1 shape: ", test_pairs_set1.shape)
    print("test_target_set1 shape: ", test_target_set1.shape)
    print()

    # the shape of image data
    input_shape = input_trainset.shape[1:]

    # test 1 using siamese network with contrastive loss
    print('------- Test1 with Contrastive loss -------')
    model1 = siamese_network(input_shape=input_shape,
                             batch_size=batch_size,
                             training_pairs=training_pairs,
                             training_target=training_target,
                             test_pairs=test_pairs_set1,
                             test_target=test_target_set1,
                             epochs=epochs_siamese,
                             verbose=1
                             )
    print()
    print("------- Test1 using Contrastive loss -------")
    print_accuracy(model1)
    print()


def contrastive_test2():
    # load the omniglot data as a tensor dataset type
    ds, ds_info = tfds.load(name='omniglot', split=['train', 'test'], with_info=True)

    # load image and label data from dataset
    x_train, y_train, x_test, y_test = omniglot_dataset(ds)

    # the num of classes
    num_classes = ds_info.features['label'].num_classes
    num_classes_trainset = len(set(y_train))
    num_classes_testset = len(set(y_test))

    # Prepocess the data for the training and variour test into 4 different datasets
    (input_trainset, target_trainset), (input_testset1, target_testset1), \
    (input_testset2, target_testset2), (input_testset3, target_testset3) \
        = preprocess_omniglot_dataset(x_train, y_train, x_test, y_test)

    # print information about the initial datasets before converting into pairs
    # in order to demonstrate the amount of images that are used for training and testing respectively
    print("input_trainset: ", input_trainset.shape)
    print("target_trainset: ", target_trainset.shape)
    print("input_testset2: ", input_testset2.shape)
    print("target_testset2: ", target_testset2.shape)
    print()

    # list for labels contained in each dataset
    # in the case of Omniglot data,
    #   label (0~963)    is for train set
    #   label (964~1622) is for test set
    set1_labels = list(range(num_classes_trainset))
    set2_labels = list(range(num_classes))

    # the specific index used for the different dataset sets of image
    # create image pairs for training the model
    label_indices = [np.where(target_trainset == i)[0] for i in set1_labels]
    training_pairs, training_target = create_pairs_set(input_trainset, label_indices, 1)

    # create image pairs for test2
    label_indices = [np.where(target_testset2 == i)[0] for i in set2_labels]
    test_pairs_set2, test_target_set2 = create_pairs_set(input_testset2, label_indices, 2)

    # print the shape of pair sets
    print("training_pairs shape: ", training_pairs.shape)
    print("training_target shape: ", training_target.shape)
    print("test_pairs_set2 shape: ", test_pairs_set2.shape)
    print("test_target_set2 shape: ", test_target_set2.shape)
    print()

    # the shape of image data
    input_shape = input_trainset.shape[1:]

    # test 2 using siamese network with contrastive loss
    print('------- Test2 with Contrastive loss -------')
    model2 = siamese_network(input_shape=input_shape,
                             batch_size=batch_size,
                             training_pairs=training_pairs,
                             training_target=training_target,
                             test_pairs=test_pairs_set2,
                             test_target=test_target_set2,
                             epochs=epochs_siamese,
                             verbose=1
                             )
    print()
    print("------- Test2 using Contrastive loss -------")
    print_accuracy(model2)
    print()


def contrastive_test3():
    # load the omniglot data as a tensor dataset type
    ds, ds_info = tfds.load(name='omniglot', split=['train', 'test'], with_info=True)

    # load image and label data from dataset
    x_train, y_train, x_test, y_test = omniglot_dataset(ds)

    # the num of classes
    num_classes = ds_info.features['label'].num_classes
    num_classes_trainset = len(set(y_train))
    num_classes_testset = len(set(y_test))

    # Prepocess the data for the training and variour test into 4 different datasets
    (input_trainset, target_trainset), (input_testset1, target_testset1), \
    (input_testset2, target_testset2), (input_testset3, target_testset3) \
        = preprocess_omniglot_dataset(x_train, y_train, x_test, y_test)

    # print information about the initial datasets before converting into pairs
    # in order to demonstrate the amount of images that are used for training and testing respectively
    print("input_trainset: ", input_trainset.shape)
    print("target_trainset: ", target_trainset.shape)
    print("input_testset3: ", input_testset3.shape)
    print("target_testset3: ", target_testset3.shape)
    print()

    # list for labels contained in each dataset
    # in the case of Omniglot data,
    #   label (0~963)    is for train set
    #   label (964~1622) is for test set
    set1_labels = list(range(num_classes_trainset))
    set3_labels = list(range(num_classes_trainset, num_classes))

    # the specific index used for the different dataset sets of image
    # create image pairs for training the model
    label_indices = [np.where(target_trainset == i)[0] for i in set1_labels]
    training_pairs, training_target = create_pairs_set(input_trainset, label_indices, 1)

    # create image pairs for test3
    label_indices = [np.where(target_testset3 == i)[0] for i in set3_labels]
    test_pairs_set3, test_target_set3 = create_pairs_set(input_testset3, label_indices, 3)

    # print the shape of pair sets
    print("training_pairs shape: ", training_pairs.shape)
    print("training_target shape: ", training_target.shape)
    print("test_pairs_set3 shape: ", test_pairs_set3.shape)
    print("test_target_set3 shape: ", test_target_set3.shape)
    print()

    # the shape of image data
    input_shape = input_trainset.shape[1:]

    # test 3 using siamese network with contrastive loss
    print('------- Test3 with Contrastive loss -------')
    model3 = siamese_network(input_shape=input_shape,
                             batch_size=batch_size,
                             training_pairs=training_pairs,
                             training_target=training_target,
                             test_pairs=test_pairs_set3,
                             test_target=test_target_set3,
                             epochs=epochs_siamese,
                             verbose=1
                             )
    print()
    print("------- Test3 using Contrastive loss -------")
    print_accuracy(model3)
    print()


def triplet_test1():
    # load the omniglot data as a tensor dataset type
    ds, ds_info = tfds.load(name='omniglot', split=['train', 'test'], with_info=True)

    # load image and label data from dataset
    x_train, y_train, x_test, y_test = omniglot_dataset(ds)

    # the num of classes
    num_classes = ds_info.features['label'].num_classes
    num_classes_trainset = len(set(y_train))
    num_classes_testset = len(set(y_test))

    # Prepocess the data for the training and variour test into 4 different datasets
    (input_trainset, target_trainset), (input_testset1, target_testset1), \
    (input_testset2, target_testset2), (input_testset3, target_testset3) \
        = preprocess_omniglot_dataset(x_train, y_train, x_test, y_test)

    # print information about the initial datasets before converting into pairs
    # in order to demonstrate the amount of images that are used for training and testing respectively
    print("input_trainset: ", input_trainset.shape)
    print("target_trainset: ", target_trainset.shape)
    print("input_testset1: ", input_testset1.shape)
    print("target_testset1: ", target_testset1.shape)

    # list for labels contained in each dataset
    # in the case of Omniglot data,
    #   label (0~963)    is for train set
    #   label (964~1622) is for test set
    set1_labels = list(range(num_classes_trainset))

    # create image triplet sets for training and various test
    label_indices = [np.where(target_trainset == i)[0] for i in set1_labels]
    training_triplets, training_triplets_target = create_triplets(input_trainset, label_indices, 1)

    label_indices = [np.where(target_testset1 == i)[0] for i in set1_labels]
    test_triplet_set1, test_triplet_target_set1 = create_triplets(input_testset1, label_indices, 1)

    # print the shape of triplet sets
    print("training_triplets shape: ", training_triplets.shape)
    print("test_triplets shape: ", training_triplets_target.shape)
    print("test_triplet_set1 shape: ", test_triplet_set1.shape)
    print("test_triplet_target_set1: ", test_triplet_target_set1.shape)
    print()

    # the shape of image data
    input_shape = input_trainset.shape[1:]

    # test 1 using siamese network with triplet loss
    print('------- Test1 with Triplet loss -------')
    tri_model1 = siamese_network_triplet_loss(input_shape=input_shape,
                                              batch_size=batch_size,
                                              training_pairs=training_triplets,
                                              training_target=training_triplets_target[:, 1],
                                              test_pairs=test_triplet_set1,
                                              test_target=test_triplet_target_set1[:, 1],
                                              epochs=epochs_siamese,
                                              verbose=1
                                              )
    print()
    print("------- Test1 using Triplet loss -------")
    print_accuracy(tri_model1)
    print()


def triplet_test2():
    # load the omniglot data as a tensor dataset type
    ds, ds_info = tfds.load(name='omniglot', split=['train', 'test'], with_info=True)

    # load image and label data from dataset
    x_train, y_train, x_test, y_test = omniglot_dataset(ds)

    # the num of classes
    num_classes = ds_info.features['label'].num_classes
    num_classes_trainset = len(set(y_train))
    num_classes_testset = len(set(y_test))

    # Prepocess the data for the training and variour test into 4 different datasets
    (input_trainset, target_trainset), (input_testset1, target_testset1), \
    (input_testset2, target_testset2), (input_testset3, target_testset3) \
        = preprocess_omniglot_dataset(x_train, y_train, x_test, y_test)

    # print information about the initial datasets before converting into pairs
    # in order to demonstrate the amount of images that are used for training and testing respectively
    print("input_trainset: ", input_trainset.shape)
    print("target_trainset: ", target_trainset.shape)
    print("input_testset2: ", input_testset2.shape)
    print("target_testset2: ", target_testset2.shape)
    print()

    # list for labels contained in each dataset
    # in the case of Omniglot data,
    #   label (0~963)    is for train set
    #   label (964~1622) is for test set
    set1_labels = list(range(num_classes_trainset))
    set2_labels = list(range(num_classes))

    # create image triplet sets for training and various test
    label_indices = [np.where(target_trainset == i)[0] for i in set1_labels]
    training_triplets, training_triplets_target = create_triplets(input_trainset, label_indices, 1)

    label_indices = [np.where(target_testset2 == i)[0] for i in set2_labels]
    test_triplet_set2, test_triplet_target_set2 = create_triplets(input_testset2, label_indices, 2)

    # print the shape of triplet sets
    print("training_triplets shape: ", training_triplets.shape)
    print("test_triplets shape: ", training_triplets_target.shape)
    print("test_triplet_set2 shape: ", test_triplet_set2.shape)
    print("test_triplet_target_set2: ", test_triplet_target_set2.shape)
    print()

    # the shape of image data
    input_shape = input_trainset.shape[1:]

    # test 2 using siamese network with triplet loss
    print('------- Test2 with Triplet loss -------')
    tri_model2 = siamese_network_triplet_loss(input_shape=input_shape,
                                              batch_size=batch_size,
                                              training_pairs=training_triplets,
                                              training_target=training_triplets_target[:, 1],
                                              test_pairs=test_triplet_set2,
                                              test_target=test_triplet_target_set2[:, 1],
                                              epochs=epochs_siamese,
                                              verbose=1
                                              )
    print()
    print_accuracy(tri_model2)
    print()


def triplet_test3():
    # load the omniglot data as a tensor dataset type
    ds, ds_info = tfds.load(name='omniglot', split=['train', 'test'], with_info=True)

    # load image and label data from dataset
    x_train, y_train, x_test, y_test = omniglot_dataset(ds)

    # the num of classes
    num_classes = ds_info.features['label'].num_classes
    num_classes_trainset = len(set(y_train))
    num_classes_testset = len(set(y_test))

    # Prepocess the data for the training and variour test into 4 different datasets
    (input_trainset, target_trainset), (input_testset1, target_testset1), \
    (input_testset2, target_testset2), (input_testset3, target_testset3) \
        = preprocess_omniglot_dataset(x_train, y_train, x_test, y_test)

    # print information about the initial datasets before converting into pairs
    # in order to demonstrate the amount of images that are used for training and testing respectively
    print("input_trainset: ", input_trainset.shape)
    print("target_trainset: ", target_trainset.shape)
    print("input_testset3: ", input_testset3.shape)
    print("target_testset3: ", target_testset3.shape)
    print()

    # list for labels contained in each dataset
    # in the case of Omniglot data,
    #   label (0~963)    is for train set
    #   label (964~1622) is for test set
    set1_labels = list(range(num_classes_trainset))
    set3_labels = list(range(num_classes_trainset, num_classes))

    # create image triplet sets for training and various test
    label_indices = [np.where(target_trainset == i)[0] for i in set1_labels]
    training_triplets, training_triplets_target = create_triplets(input_trainset, label_indices, 1)

    label_indices = [np.where(target_testset3 == i)[0] for i in set3_labels]
    test_triplet_set3, test_triplet_target_set3 = create_triplets(input_testset3, label_indices, 3)

    # print the shape of triplet sets
    print("training_triplets shape: ", training_triplets.shape)
    print("test_triplets shape: ", training_triplets_target.shape)
    print("test_triplet_set3 shape: ", test_triplet_set3.shape)
    print("test_triplet_target_set3: ", test_triplet_target_set3.shape)
    print()

    # the shape of image data
    input_shape = input_trainset.shape[1:]

    # test 3 using siamese network with triplet loss
    print('------- Test3 with Triplet loss -------')
    tri_model3 = siamese_network_triplet_loss(input_shape=input_shape,
                                              batch_size=batch_size,
                                              training_pairs=training_triplets,
                                              training_target=training_triplets_target[:, 1],
                                              test_pairs=test_triplet_set3,
                                              test_target=test_triplet_target_set3[:, 1],
                                              epochs=epochs_siamese,
                                              verbose=1
                                              )
    print()
    print("------- Test3 using Triplet loss -------")
    print_accuracy(tri_model3)
    print()


if __name__ == '__main__':
    contrastive_test1()
    contrastive_test2()
    contrastive_test3()
    triplet_test1()
    triplet_test2()
    triplet_test3()