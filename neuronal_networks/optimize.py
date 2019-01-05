import time
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense


def optimize(x_train, y_train, x_test, y_test):

    dense_layers = [2, 4, 8]
    #dense_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    layer_sizes = [32, 64, 128]
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
    #acc = []

    for result in results:
        #acc.append(result["val_acc"])
        if best_val_acc is None or best_val_acc < result["val_acc"]:
            best_val_acc = result["val_acc"]
            best_parameters = result
    """        
    plt.plot(dense_layers, acc, color="#111111")
    plt.xlabel("Dense layer"), plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
    """

    return best_parameters
