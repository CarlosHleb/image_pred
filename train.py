
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import sys

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import preprocess_data

def build_and_train(dataset, filters, batch_size, model_path=os.environ["MODEL_PATH"]):
    dataset = dataset.batch(batch_size).prefetch(1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.ConvLSTM2D(
            filters=filters,
            kernel_size=(5, 5),
            padding="same",
            return_sequences=True,
            activation="relu",
            dropout=myDO,
            recurrent_dropout=myDO
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ConvLSTM2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
            dropout=myDO,
            recurrent_dropout=myDO
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ConvLSTM2D(
            filters=filters,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
            dropout=myDO,
            recurrent_dropout=myDO
        ),
        tf.keras.layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
        ),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')

    history = model.fit(dataset, epochs=int(os.environ["EPOCHS"]))
    model.save(model_path)

    # Plot learning curves
    history_dict = history.history
    loss_values = history_dict["loss"]
    epochs = range(1, len(loss_values) + 1)
    loss_plot_filename = model_path.split("/")[-1] + "_loss_plot.png"

    plt.plot(epochs, loss_values, "b", label="Training loss") 
    plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.environ["WORKING_DIR"] + "loss_plots/" + os.environ["MOD_VERSION"] + "/" + loss_plot_filename)
    plt.clf()

# Build model
myDO = 0.3
dataset, _, _, _, _ = preprocess_data.pre_process()

if len(os.environ["FILTERS_COUNT"]) > 4:
    filters = [int(f) for f in os.environ["FILTERS_COUNT"].split(",")]
    batch_sizes = [int(f) for f in os.environ["BATCH_SIZE"].split(",")]

    for idx, filter in enumerate(filters):
        build_and_train(dataset, filter, batch_sizes[idx], 
            model_path=os.environ["MODEL_PATH"][:-6] + "_filters" + str(filter) + 
            os.environ["MODEL_PATH"][-6:])
else:
    build_and_train(dataset, int(os.environ["FILTERS_COUNT"]), int(os.environ["BATCH_SIZE"]))
