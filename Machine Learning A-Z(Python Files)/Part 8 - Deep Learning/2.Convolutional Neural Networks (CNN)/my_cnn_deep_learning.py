
#imports
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# INITIAL
classifier = Sequential()

#s-1--convoluation
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))
#s-2--maxpooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#second time
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#s-3--flatten
classifier.add(Flatten())

#S-4--nn
#create hidden layer
classifier.add(Dense(output_dim=128, activation='relu'))
#output layer
classifier.add(Dense(output_dim=1, activation='sigmoid'))

#compile
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
                                    'dataset/training_set',
                                    target_size=(64, 64),
                                    batch_size=32,
                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(train_set,
                            samples_per_epoch=8000,
                            nb_epoch=25,
                            validation_data=test_set,
                            nb_val_samples=2000)

#make pridection for new image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/test_set/cats/cat.4420.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'