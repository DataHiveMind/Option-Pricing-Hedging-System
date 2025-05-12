import tensorflow as tf

def create_option_pricing_model(input_dim):
    """
    Creates a neural network model for option pricing using TensorFlow.

    Args:
        input_dim (int): The number of input features (e.g., time to expiration,
            strike price, stock price).

    Returns:
        tf.keras.Model: A TensorFlow Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for option price
    ])
    return model

def train_option_pricing_model(model, train_data, val_data=None, num_epochs=100, batch_size=32):
    """
    Trains the neural network model for option pricing.

    Args:
        model (tf.keras.Model): The TensorFlow Keras model to train.
        train_data (tuple): A tuple containing (X_train, y_train), where X_train
            is the training input data and y_train is the training option prices.
        val_data (tuple, optional): A tuple containing (X_val, y_val) for
            validation data. Defaults to None.
        num_epochs (int): The number of training epochs.
        batch_size (int): The batch size.

    Returns:
        tf.keras.Model: The trained TensorFlow Keras model.
        tf.keras.callbacks.History: The training history object.
    """
    model.compile(optimizer='adam', loss='mse')  # Use mean squared error loss
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
    if val_data:
        history = model.fit(
            train_data[0], train_data[1],
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(val_data[0], val_data[1]),
            callbacks=callbacks
        )
    else:
        history = model.fit(
            train_data[0], train_data[1],
            epochs=num_epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
    return model, history

def predict_option_price(model, test_data):
    """
    Predicts option prices using the trained neural network model.

    Args:
        model (tf.keras.Model): The trained TensorFlow Keras model.
        test_data (np.ndarray): The input data for which to make predictions
            (e.g., time to expiration, strike price, stock price).

    Returns:
        np.ndarray: The predicted option prices.
    """
    predictions = model.predict(test_data)
    return predictions