# %%
import pandas as pd
import numpy as np
import math
from keras.utils import to_categorical

data = pd.read_csv("./mnist/train.csv")
data.head()

labels = []
images = []
for i, row in data.iterrows():
    labels.append(row[0])
    images.append(np.array(row[1:]).reshape(1, 28, 28))

y = np.array(labels).astype("int64")
x = np.array(images).astype("float32")
np.divide(x, np.max(x))

y = to_categorical(y, num_classes=10)

print(y.shape)


# %%
import keras
from keras.models import Sequential
from keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Masking,
    Embedding,
    Flatten,
    Conv1D,
    Conv2D,
    MaxPooling2D,
)

print(x.shape)

model = Sequential()
model.add(
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(1, 28, 28),
        data_format="channels_first",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.18))
model.add(Dense(10, activation="softmax"))
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

print(model.summary())

model.fit(x=x, y=y, validation_split=0.1, verbose=1, epochs=15)

# %%
test = pd.read_csv("./mnist/test.csv")
test.head()

images = []
for i, row in test.iterrows():
    images.append(row[:].reshape(1, 28, 28))

test_x = np.array(images)
np.divide(test_x, np.max(test_x))
predictions = model.predict_classes(test_x)

with open("./mnist/prediction.csv", "w") as out:
    out.write("ImageId,Label\n")
    for i, pred in enumerate(predictions):
        out.write(f"{i+1},{pred}\n")
