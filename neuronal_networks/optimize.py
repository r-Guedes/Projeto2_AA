import time
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense


def optimize(x_train, y_train, x_test, y_test):

    dense_layers = [0, 1, 2]
    layer_sizes = [32, 64, 128]
    classifications = 3

    results = []

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
                NAME = "{}-nodes-{}-dense-{}".format(layer_size, dense_layer, int(time.time()))
                print(NAME)

                model = Sequential()

                model.add(Dense(12, input_dim=11, activation="relu"))	

                for _ in range(dense_layer):
                    model.add(Dense(layer_size, activation="relu"))

                model.add(Dense(8, activation='relu'))
                model.add(Dense(classifications, activation="softmax"))

                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME), write_graph=True)

                adam = optimizers.adam(lr=0.01, clipnorm=1.)
                model.compile(loss='sparse_categorical_crossentropy',
                              optimizer=adam,
                              metrics=['accuracy'],
                              )

                history = model.fit(x_train, y_train, epochs=50, batch_size=50, verbose=2, validation_data=(x_test, y_test),
                          callbacks=[tensorboard])

                results.append({
                    "dense_layer": dense_layer,
                    "layer_size": layer_size,
                    "val_acc": sum(history.history["val_acc"]) / float(len(history.history["val_acc"]))
                })

    best_val_acc = None
    best_parameters = None

    for result in results:
        if best_val_acc is None or best_val_acc < result["val_acc"]:
            best_val_acc = result["val_acc"]
            best_parameters = result

    return best_parameters
