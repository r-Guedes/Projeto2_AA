from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


def nn_train(x_train, y_train, x_test, y_test, best_parameters):

    classifications = 3

    model = Sequential()

    model.add(Dense(12, input_dim=11, activation='relu'))

    for _ in range(best_parameters["dense_layer"]):
        model.add(Dense(best_parameters["layer_size"], activation="relu"))

    model.add(Dense(8, activation='relu'))
    model.add(Dense(classifications, activation="softmax"))

    adam = optimizers.adam(lr=0.01, clipnorm=1.)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #model = KerasClassifier(build_fn=model, epochs=150, batch_size=10, verbose=0)
    #cross_val_score(estimator=model, X=x_test, y=y_test)

    model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=2, validation_data=(x_test, y_test))

    return model