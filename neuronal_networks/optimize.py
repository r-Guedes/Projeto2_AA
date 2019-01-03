import time
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense


def optimize(x_train, y_train, x_test, y_test):

    dense_layers = [2, 4, 8]
    layer_sizes = [32, 64, 128]
    #learn_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    learn_rates = [0.1]
    classifications = 3

    results = []

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for learn_rate in learn_rates:
                    NAME = "{}-nodes-{}-dense-{}-Lr-{}".format(dense_layer, layer_size,learn_rate, int(time.time()))
                    print(NAME)
    
                    model = Sequential()
    
                    model.add(Dense(9, input_dim=8, activation="relu"))	
    
                    for _ in range(dense_layer):
                        model.add(Dense(layer_size, activation="relu"))
    
                    model.add(Dense(8, activation='relu'))
                    model.add(Dense(classifications, activation="softmax"))
    
                    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME), write_graph=True)
    
                    adam = optimizers.adam(lr=learn_rate, clipnorm=1.)
                    model.compile(loss='sparse_categorical_crossentropy',
                                  optimizer=adam,
                                  metrics=['accuracy'],
                                  )
    
                    history = model.fit(x_train, y_train, epochs=5, batch_size=50, verbose=2, validation_data=(x_test, y_test),
                              callbacks=[tensorboard])
    
                    results.append({
                        "dense_layer": dense_layer,
                        "layer_size": layer_size,
                        "learn_rate": learn_rate,
                        "val_acc": sum(history.history["val_acc"]) / float(len(history.history["val_acc"]))
                    })

    best_val_acc = None
    best_parameters = None

    for result in results:
        if best_val_acc is None or best_val_acc < result["val_acc"]:
            best_val_acc = result["val_acc"]
            best_parameters = result

    return best_parameters
