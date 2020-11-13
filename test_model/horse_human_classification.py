import tensorflow as tf
import os
import numpy as np
import zipfile, glob
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

local_zip = '/Users/daisy/Downloads/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/Users/daisy/Downloads/horse-or-human')
zip_ref.close()


model = tf.keras.Sequential([

	tf.keras.layers.Conv2D(32, (3,3),  input_shape=(300,300,3), activation='relu'),
	tf.keras.layers.MaxPool2D(3,3),
	tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
	tf.keras.layers.MaxPool2D(3,3),
	tf.keras.layers.Conv2D(256, (3,3), activation = 'relu'),
	tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
	tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
	tf.keras.layers.MaxPool2D(3,3),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(256, activation='relu'),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

class Callback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, batch, logs={}):
		if logs.get('accuracy') > .99:
			print('accuracy performance good enough!')
			self.model.stop_training = True

# compile model
model.compile(
              loss='binary_crossentropy',
              metrics=['accuracy'],
			  optimizer=tf.keras.optimizers.RMSprop(lr=.002))

# preprocessing of images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(rescale=1./255)
train_generator = train_data_gen.flow_from_directory(
	'/Users/daisy/Downloads/horse-or-human',
	target_size = (300,300),
	batch_size = 28,
	class_mode = 'binary'
)

callback = Callback()
history = model.fit(
	train_generator,
	epochs=56,
	steps_per_epoch=8,
	callbacks = [callback]
)


validation = image.load_img('/Users/daisy/Desktop/validation/hm.jpeg', target_size=(300,300) )
img = image.img_to_array(validation)
img /= 255
img = np.resize(img, (1,300,300,3))
print(model.predict(img))

print(train_generator.class_indices)
plt.imshow(img[0,:])


act_layers = model.layers
layer_output = [i.output for i in act_layers]
activation = tf.keras.models.Model(inputs= model.input, outputs=layer_output)

fig, ax = plt.subplots(8,10, figsize = (26,26))
demo = activation.predict(img)
for i in range(len(act_layers)):
	single_layer = demo[i]
	for j in range(10):
		ax[i,j].imshow(single_layer[0][:,:,j])
		ax[i,0].set_title('layer ' + str(i))
