import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

def cho_to_label(cho):
    """
    Convert CHO grams → category 0–5.
    Label 0: no CHO
    Label 1: CHO < 10 g
    Label 2: 10 ≤ CHO < 40 g
    Label 3: 40 ≤ CHO < 70 g
    Label 4: 70 ≤ CHO < 100 g
    Label 5: CHO ≥ 100 g
    """
    if cho <= 0:
        return 0
    elif cho < 10:
        return 1
    elif cho < 40:
        return 2
    elif cho < 70:
        return 3
    else:
        return 4


# Helper: convert array of CHO values to labels using cho_to_label
def convert_regression_to_labels(y):
    """
    Convert an array of CHO gram values (true or predicted) into
    discrete meal-size categories 0–5 using cho_to_label.
    """
    y = np.asarray(y).reshape(-1)
    labels = [cho_to_label(float(max(0.0, v))) for v in y]
    return np.asarray(labels, dtype=int)

def evaluate_regression(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mse = np.mean((y_true - y_pred)**2)

    # ---- NEW: Convert to meal-size categories ----
    y_true_cat = convert_regression_to_labels(y_true)
    y_pred_cat = convert_regression_to_labels(y_pred)

    cls_report = classification_report(y_true_cat, y_pred_cat, output_dict=False)
    cm = confusion_matrix(y_true_cat, y_pred_cat)

    print("\n📊 Regression → Classification Evaluation")
    print(cls_report)
    print("\nConfusion matrix:\n", cm)

    return {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "cls_report": cls_report,
        "confusion_matrix": cm
    }

def build_regression_lstm(
    timesteps,
    features,
    n_lstm_layers=2,
    lr=1e-3,
    batchnorm=True,
    dense1_units=64,
    dense2_units=32,
    lstm_dropout=0.2,
    lstm_recurrent_dropout=0.0,
    dense_dropout=0.3,
):
    """
    Improved LSTM regression model:
    - Uses Huber loss (better for outliers)
    - Clips predictions to [0, 200] g CHO
    - Predicts continuous CHO grams
    """

    inputs = layers.Input(shape=(timesteps, features))

    # ---- LSTM layers ----
    x = layers.LSTM(
        128,
        return_sequences=(n_lstm_layers > 1),
        dropout=lstm_dropout,
        recurrent_dropout=lstm_recurrent_dropout
    )(inputs)

    if n_lstm_layers >= 2:
        x = layers.LSTM(
            64,
            return_sequences=False,
            dropout=lstm_dropout + 0.1,
            recurrent_dropout=lstm_recurrent_dropout
        )(x)

    # ---- Dense head ----
    x = layers.Dense(dense1_units)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dense_dropout)(x)

    if dense2_units > 0:
        x = layers.Dense(dense2_units, activation="relu")(x)

    # ---- Regression output ----
    raw_output = layers.Dense(1, activation="relu")(x)

    # ---- Clip predictions to [0, 200] ----
    output = layers.Lambda(lambda z: tf.clip_by_value(z, 0.0, 200.0))(raw_output)

    model = models.Model(inputs, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(delta=10.0),
        metrics=["mae"]
    )

    return model