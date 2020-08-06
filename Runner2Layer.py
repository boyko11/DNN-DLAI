from two_layer_nn import TwoLayerNNModel
import data_service, plotting_service, dnn_service


train_x, train_y, test_x, test_y, classes = data_service.load_and_preprocess_data()

n_x = train_x.shape[0]     # num_px * num_px * 3
n_h = 7
n_y = train_y.shape[0]
layers_dims = (n_x, n_h, n_y)

learning_rate = 0.0075
num_iterations = 3000

two_layer_NN = TwoLayerNNModel()
parameters, costs = two_layer_NN.fit(train_x, train_y, layers_dims, learning_rate=learning_rate,
                                     num_iterations=num_iterations, print_cost=True)

plotting_service.plot_learning_curve(costs, learning_rate)

pred_train = dnn_service.predict(train_x, train_y, parameters)
print('Training accuracy: ', pred_train)

