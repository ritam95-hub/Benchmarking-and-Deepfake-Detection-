from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_efficientnet():

    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(299,299,3)
    )

    x = GlobalAveragePooling2D()(
        base.output
    )

    output = Dense(
        1,
        activation="sigmoid"
    )(x)

    return Model(
        base.input,
        output
    )