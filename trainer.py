from tensorflow.keras.optimizers import Adam

class Trainer:

    def __init__(self,
                 model):

        self.model = model

    def compile(self):

        self.model.compile(
            optimizer=Adam(
                learning_rate=2e-4
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def fit(self,
            X_train,
            y_train):

        return self.model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            batch_size=16,
            epochs=2
        )