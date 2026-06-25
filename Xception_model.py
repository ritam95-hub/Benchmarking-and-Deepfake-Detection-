from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_xception():

    base = Xception(
        weights="imagenet",
        include_top=False,
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
        inputs=base.input,
        outputs=output
    )