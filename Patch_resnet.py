from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_patch_resnet():

    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(299,299,3)
    )

    x = base.layers[80].output

    x = GlobalAveragePooling2D()(x)

    output = Dense(
        1,
        activation="sigmoid"
    )(x)

    return Model(
        inputs=base.input,
        outputs=output
    )