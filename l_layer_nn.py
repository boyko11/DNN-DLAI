import numpy as np
import dnn_service

class L_Layer_NN:

    def __init__(self):
        pass

    def fit(self, X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []                         # keep track of cost

        # Parameters initialization. (≈ 1 line of code)
        parameters = dnn_service.initialize_parameters_deep(layers_dims)

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = dnn_service.L_model_forward(X, parameters)

            # Compute cost.
            cost = dnn_service.compute_cost(AL, Y)

            # Backward propagation.
            grads = dnn_service.L_model_backward(AL, Y, caches)

            # Update parameters.
            parameters = dnn_service.update_parameters(parameters, grads, learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)


        return parameters, costs

    def predict(self, parameters, X):

        y_hat, _ = dnn_service.L_model_forward(X, parameters)
        predictions = np.rint(y_hat)

        return predictions