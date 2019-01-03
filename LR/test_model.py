from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


def test(model, x_test, y_test):

    predicted_data = model.predict(x_test)

    print("Mean squared error - test set:", mean_squared_error(y_test, predicted_data))
    print("Coefficient of determination - test set:", r2_score(y_test, predicted_data))
    print("Accuracy - test set:", accuracy_score(y_test, predicted_data))

