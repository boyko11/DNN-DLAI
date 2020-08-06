import matplotlib.pyplot as plt
import numpy as np

def set_up():

    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

# credit - Andrew Ng's deeplearning.ai, last assignment Course 1
def plot_something(model, X, y, title=''):

    set_up()


    plt.title(title)
    plt.show()

def display_image(train_x, train_y, classes, index):

    set_up()

    plt.imshow(train_x[index])
    plt.show()
    print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")


def plot_learning_curve(costs, learning_rate):
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
