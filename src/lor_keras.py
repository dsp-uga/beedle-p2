import cv2
import numpy as np
import glob
import tables
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark import SparkContext
import re
from skimage import io
import tables
import numpy as np
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt
from keras.utils import np_utils
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint



def small_data_prediction():
    hdf5_path = 'dataset.hdf5'
    sc= SparkSession.builder.getOrCreate()

    #training files
 
    train_df = sc.read.csv("gs://uga-dsp/project2/files/X_small_train.csv", header=True, inferSchema=True)
    train_df = train_df.withColumnRenamed('Sex (subj)','label')
    train_img_files_list = list(train_df.select("Image File").toPandas()['Image File'])
    train_img_labels_list = list(train_df.select("label").toPandas()['label'])
    for i in range(len(train_img_files_list)):
        train_img_files_list[i]='https://storage.googleapis.com/uga-dsp/project2/images/' + train_img_files_list[i]


    #testing files
    test_df1 = sc.read.csv("gs://uga-dsp/project2/files/X_small_test.csv", header=True, inferSchema=True)
    test_df1 = test_df1.withColumnRenamed('Sex (subj)','label')
    # test_df2 = sc.read.csv("gs://uga-dsp/project2/files/Xb_test.csv", header=True, inferSchema=True)
#   test_df3 = sc.read.csv("gs://uga-dsp/project2/files/Xc_test.csv", header=True, inferSchema=True)
    test_img_files_list1 = list(test_df1.select("Image File").toPandas()['Image File'])
    test_img_labels_list1 = list(test_df1.select("label").toPandas()['label'])
    for i in range(len(test_img_files_list1)):
        test_img_files_list1[i]= 'https://storage.googleapis.com/uga-dsp/project2/images/' + test_img_files_list1[i] 
#       print(test_img_files_list1[i])
	
	val_addrs = test_img_files_list1[0:int(0.5*len(test_img_files_list1))]
	val_labels = test_img_labels_list1[0:int(0.5*len(test_img_labels_list1))]

	test_addrs = test_img_files_list1[int(0.5*len(test_img_files_list1)):]
	test_labels = test_img_labels_list1[int(0.5*len(test_img_labels_list1)):]

    #HDF5 dataset creation

    data_order = 'tf'
    img_dtype = tables.UInt8Atom()
    if data_order == 'tf':
        data_shape = (0, 128, 128, 3)
    #hdf5_file.close()
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    try:
        train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
        test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)
        val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)
        mean_storage = hdf5_file.create_earray(hdf5_file.root, 'train_mean', img_dtype, shape=data_shape)

    # create the label arrays and copy the labels data in them
        hdf5_file.create_array(hdf5_file.root, 'train_labels', train_img_labels_list)
        hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)
        hdf5_file.create_array(hdf5_file.root, 'test_labels',test_img_labels_list1 )

    # a numpy array to save the mean of the images
        mean = np.zeros(data_shape[1:], np.float32)

    # loop over train addresses
        for i in range(len(train_img_files_list)):
        # print how many images are saved every 100 images
#         if i % 10 == 0 and i > 1:
#             print('Train data: {}/{}'.format(i, len(train_img_files_list)))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
            addr = train_img_files_list[i]
#         print(addr)
            img = io.imread(addr)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # add any image pre-processing here

        # if the data order is Theano, axis orders should change
#         if data_order == 'th':
#             img = np.rollaxis(img, 2)
            train_storage.append(img[None])
            mean += img / float(len(train_img_labels_list))

        for i in range(len(val_addrs)):
        # print how many images are saved every 1000 images
        # if i % 10 == 0 and i > 1:
        #     print ('Validation data: {}/{}'.format(i, len(val_addrs)))
    
        # read an image and resize to (128, 128)
        # cv2 load images as BGR, convert it to RGB
        	addr = val_addrs[i]
        	img = io.imread(addr)
        	img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # add any image pre-processing here
    
        # if the data order is Theano, axis orders should change
        # if data_order == 'th':
        #     img = np.rollaxis(img, 2)
    
        # save the image
        	val_storage.append(img[None])
    # loop over test addresses
        for i in range(len(test_addrs)):
        # print how many images are saved every 1000 images
#         if i % 10 == 0 and i > 1:
#             print ('Test data: {}/{}'.format(i, len(test_img_files_list)))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
            addr = test_addrs[i]
        #print(addr)
            img = io.imread(addr)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # add any image pre-processing here

        # if the data order is Theano, axis orders should change
            if data_order == 'th':
                img = np.rollaxis(img, 2)

        # save the image
            test_storage.append(img[None])

    # save the mean and close the hdf5 file
        mean_storage.append(mean[None])
        print('HDF5 Done')
        hdf5_file.close()
    finally:
        print('In Finally')
    train_cnn(hdf5_path)

def train_cnn(hdf5_path):
	subtract_mean = True
	batch_size = 50
	nb_class = 2

	hdf5_file = tables.open_file(hdf5_path, mode='r')

	# subtract the training mean
	if subtract_mean:
    	mm = hdf5_file.root.train_mean[0]
    	mm = mm[np.newaxis, ...]

    train_data = np.array(hdf5_file.root.train_img)
	train_label = np.array(hdf5_file.root.train_labels)

	test_data = np.array(hdf5_file.root.test_img)
	test_label = np.array(hdf5_file.root.test_labels)

	val_data = np.array(hdf5_file.root.val_img)
	val_label = np.array(hdf5_file.root.val_labels)

	print('train data:',train_data.shape,' train_label',train_label.shape)
	print('test_data:',test_data.shape,' test_label:',test_label.shape)

	# one-hot encode the labels
	num_classes = len(np.unique(train_label))
	train_label = np_utils.to_categorical(train_label, num_classes)
	test_label = np_utils.to_categorical(test_label, num_classes)

# print shape of training set
	print('num_classes:', num_classes)

# print number of training, and test images
	print(train_label.shape, 'train samples')
	print(test_label.shape, 'test samples')

	model1 = build_model()

	# train the model
	
	checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
	hist = model.fit(train_data, train_label, batch_size=None, epochs=20, validation_data=(val_data, val_label), callbacks=[checkpointer], verbose=1, shuffle=True)
	model1.load_weights('model.weights.best.hdf5')
	score = model1.evaluate(test_data, test_label, verbose=0)
	print('\n', 'Test accuracy:', score[1])

def build_model():

	model = Sequential()
	model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(128, 128, 3)))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='tanh'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='tanh'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(2, activation='softmax'))

	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model
	
	



def main():
    small_data_prediction()

if __name__ == "__main__":
    main()
