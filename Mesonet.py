from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_mesonet():

    inp = Input(
        shape=(256,256,3)
    )

    x = Conv2D(
        8,
        (3,3),
        activation="relu"
    )(inp)

    x = MaxPool2D()(x)

    x = Conv2D(
        8,
        (5,5),
        activation="relu"
    )(x)

    x = MaxPool2D()(x)

    x = Conv2D(
        16,
        (5,5),
        activation="relu"
    )(x)

    x = MaxPool2D()(x)

    x = Flatten()(x)

    x = Dense(
        16,
        activation="relu"
    )(x)

    output = Dense(
        1,
        activation="sigmoid"
    )(x)

    return Model(inp, output)