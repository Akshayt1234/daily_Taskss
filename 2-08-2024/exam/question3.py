import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import numpy as np

# Define the custom loss function
class MSEReg(tf.keras.losses.Loss):
    def __init__(self, regularization_factor=0.01, **kwargs):
        super().__init__(**kwargs)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return mse_loss

# Create a simple neural network model with built-in regularization
model = Sequential([
    Input(shape=(10,)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(1)
])

# Compile the model with the custom loss function
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss=MSEReg(regularization_factor=0.01),
              metrics=['mse'])

# Generate some dummy data
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=16)

# Print training history
print(f"Final training loss: {history.history['loss'][-1]}")
