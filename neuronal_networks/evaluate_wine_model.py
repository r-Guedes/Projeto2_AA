
def evaluate(model, x_train, y_train):
    score = model.evaluate(x_train, y_train, batch_size=32)
    print(score)
#    print("Score - training set:", model.score(x_train, y_train))
#    print("Mean squared error - training set:", mean_squared_error(y_train, predicted_data))
#    print("Coefficient of determination - training set:", r2_score(y_train, predicted_data))
#    print("Accuracy - training set:", accuracy_score(y_train, predicted_data))
