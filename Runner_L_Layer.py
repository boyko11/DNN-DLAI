from l_layer_nn import L_Layer_NN
import data_service, plotting_service, dnn_service


train_x, train_y, test_x, test_y, classes = data_service.load_and_preprocess_data()

n_x = train_x.shape[0]     # num_px * num_px * 3
n_h1 = 20
n_h2 = 7
n_h3 = 5
n_y = train_y.shape[0]
layers_dims = [n_x, n_h1, n_h2, n_h3, n_y] #  4-layer model

learning_rate = 0.0075
num_iterations = 2501

l_layer_NN = L_Layer_NN()
parameters, costs = l_layer_NN.fit(train_x, train_y, layers_dims, learning_rate=learning_rate,
                                   num_iterations=num_iterations, print_cost=True)

plotting_service.plot_learning_curve(costs, learning_rate)

train_predictions = l_layer_NN.predict(parameters, train_x)
test_predictions = l_layer_NN.predict(parameters, test_x)

train_accuracy = dnn_service.accuracy(train_predictions, train_y)
test_accuracy = dnn_service.accuracy(test_predictions, test_y)

print("Train/Test accuracy: ", train_accuracy, test_accuracy)


