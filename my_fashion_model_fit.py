
# %%
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# %% 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# %% 
train_images

# %%
train_images = train_images / 255.0
test_images = test_images / 255.0


# %%
from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)




#########################################################
# model simple
#########################################################


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential(
    [
      Flatten(input_shape=(28, 28)),
      Dense(128, activation='relu'),
      Dense(64, activation='relu'),
      Dense(10, activation='softmax')
    ]
)
# %%
from tensorflow.keras.optimizers import SGD

model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
  
# %%
model.fit(train_images, train_labels, epochs=10, batch_size=32)
# %%
test_loss, test_acc = model.evaluate(train_images, train_labels)

# %%
import matplotlib.pyplot as plt

IMAGE_INDEX = 20

plt.imshow(train_images[IMAGE_INDEX], cmap='gray')
input_image = np.expand_dims(train_images[IMAGE_INDEX], axis=0)
predicted = model.predict(input_image)

print(
    np.argmax(predicted)
)
np.argmax(predicted)
# %%
train_labels[IMAGE_INDEX]
np.argmax(train_labels[IMAGE_INDEX])

# %%
predicted_all = model.predict(train_images)
predicted_max_all = np.argmax(predicted_all, axis=1)
label_max_all = np.argmax(train_labels, axis=1)

correct = np.sum(predicted_max_all == label_max_all)
not_correct = len(train_labels) - correct
all = correct + not_correct

print([correct/all, not_correct/all])
# %%
predicted_all = model.predict(test_images)
predicted_max_all = np.argmax(predicted_all, axis=1)
label_max_all = np.argmax(test_labels, axis=1)

correct = np.sum(predicted_max_all == label_max_all)
not_correct = len(test_labels) - correct
all = correct + not_correct

print([correct/all, not_correct/all])


#########################################################
# model complex
#########################################################
# %%
from tensorflow.keras import layers, models
model_complex = models.Sequential()

model_complex.add(
    layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1))
)
model_complex.add(
    layers.MaxPooling2D((2, 2))
)

model_complex.add(
    layers.Conv2D(64, (3, 3), activation='relu')
)

model_complex.add(
    layers.MaxPooling2D((2,2))
)

model_complex.add(
    layers.Conv2D(64, (3, 3), activation='relu')
)

model_complex.add(
    layers.Flatten()
)
model_complex.add(
    layers.Dense(64, activation='relu')
)
model_complex.add(
    layers.Dense(10, activation='softmax')
)
# %%
model_complex.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        #loss="sparse_categorical_crossentropy",
        loss="categorical_crossentropy",
        metrics=["accuracy"])
# %%
model_complex.fit(train_images, train_labels, epochs=10, batch_size=32)
# %%
test_loss, test_acc = model_complex.evaluate(train_images, train_labels)

# %%
import matplotlib.pyplot as plt

IMAGE_INDEX = 20

plt.imshow(train_images[IMAGE_INDEX], cmap='gray')
input_image = np.expand_dims(train_images[IMAGE_INDEX], axis=0)
predicted = model_complex.predict(input_image)

print(
    np.argmax(predicted)
)
np.argmax(predicted)
# %%
train_labels[IMAGE_INDEX]
np.argmax(train_labels[IMAGE_INDEX])

# %%
predicted_all = model_complex.predict(train_images)
predicted_max_all = np.argmax(predicted_all, axis=1)
label_max_all = np.argmax(train_labels, axis=1)

correct = np.sum(predicted_max_all == label_max_all)
not_correct = len(train_labels) - correct
all = correct + not_correct

print([correct/all, not_correct/all])
# %%
predicted_all = model_complex.predict(test_images)
predicted_max_all = np.argmax(predicted_all, axis=1)
label_max_all = np.argmax(test_labels, axis=1)

correct = np.sum(predicted_max_all == label_max_all)
not_correct = len(test_labels) - correct
all = correct + not_correct

print([correct/all, not_correct/all])

# %%
model.save('model/simple_model.keras')
model_complex.save('model/complex_model.keras')
# %%
