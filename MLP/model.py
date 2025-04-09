from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam


def build_model(input_shape, learning_rate):
    model = Sequential([
            Input(shape=input_shape),
            Dense(256, activation='relu', kernel_initializer=HeNormal()),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_initializer=HeNormal()),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_initializer=HeNormal()),
            Dense(32, activation='relu', kernel_initializer=HeNormal()),
            Dense(16, activation='relu', kernel_initializer=HeNormal()),
            Dense(1, activation='sigmoid')
            ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

        