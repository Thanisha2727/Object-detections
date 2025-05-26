import tensorflow as tf
from tensorflow.keras import datasets,layers,models
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print(train_images.shape)
train_images,test_images=train_images/255.0,test_images/255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),input_shape=(32,32,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation="softmax"))
model.summary()
model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(train_images,train_labels,validation_data=(test_images,test_labels),epochs=10)
print(test_labels)
