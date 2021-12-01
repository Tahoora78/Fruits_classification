import numpy as np
import matplotlib.pyplot as plt
from Loading_Datasets import *
import random
import time

train, test = loading_dataset()

def sigmoid(x):
    return 1 / (1 + (np.exp(-1 * x)))


def sigmoid_deriv(x):
    x = 1 / (1 + np.exp(-1 * x))
    result = x * (1 - x)
    return result


def calculating_cost(a, y):
    cost = np.sum((a-y)**2)
    return cost


def plotting_costs(mean_costs):
    plt.plot(np.arange(len(mean_costs)), np.array(mean_costs), '-o')
    plt.ylabel("mean_costs")
    plt.xlabel("epochs")
    plt.show()


class NeuralNetwork:
    def __init__(self, learning_rate, epochs, batch_size):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        # Initializing w and b for each layer
        w1 = np.random.randn(150, 102)
        w2 = np.random.randn(60, 150)
        w3 = np.random.randn(4, 60)
        self.w = [w1, w2, w3]

        b0 = np.zeros((150, 1))
        b1 = np.zeros((60, 1))
        b2 = np.zeros((4, 1))
        self.b = [b0, b1, b2]


    def feedforward(self, a0):
        z1 = (self.w[0] @ a0) + self.b[0]
        a1 = sigmoid(z1)
        z2 = (self.w[1] @ a1) + self.b[1]
        a2 = sigmoid(z2)
        z3 = (self.w[2] @ a2) + self.b[2]
        a3 = sigmoid(z3)
        return [a1, a2, a3], [z1, z2, z3]


    def training_with_feedforward(self, train):
        true_predicted = 0
        for i in range(len(train)):
            a, z = self.feedforward(train[i][0])
            true_predicted += self.check_true_prediction(a[2], train[i][1])
        print("accuracy with only feedforward is: ", true_predicted/len(train))

    def vectorized_backpropagation(self, grad_w, grad_b, grad_a, a, z, y):
        grad_w[2] += (sigmoid_deriv(z[2]) * (2 * a[2] - 2 * y[1])) @ (np.transpose(a[1]))
        grad_b[2] += (sigmoid_deriv(z[2]) * (2 * a[2] - 2 * y[1]))

        grad_a[1] += np.transpose(self.w[2]) @ (sigmoid_deriv(z[2]) * (2 * a[2] - 2 * y[1]))
        grad_w[1] += (sigmoid_deriv(z[1]) * grad_a[1]) @ (np.transpose(a[0]))
        grad_b[1] += (sigmoid_deriv(z[1]) * grad_a[1])

        grad_a[0] += np.transpose(self.w[1]) @ (sigmoid_deriv(z[1]) * grad_a[1])
        grad_w[0] += (sigmoid_deriv(z[0]) * grad_a[0]) @ np.transpose(y[0])
        grad_b[0] += (sigmoid_deriv(z[0]) * grad_a[0])
        return grad_w, grad_b, grad_a

    def backpropagation(self, grad_w, grad_b, grad_a, a, z, y):
        # Third layer
        # Grad_w3
        for j in range(4):
            for k in range(60):
                grad_w[2][j, k] += a[1][k, 0] * sigmoid_deriv(z[2][j, 0]) * (2 * a[2][j, 0] - 2 * y[1][j, 0])

        # Grad_b3
        for j in range(4):
            grad_b[2][j, 0] += sigmoid_deriv(z[2][j, 0]) * (2 * a[2][j, 0] - 2 * y[1][j, 0])

        # Grad_a2
        for k in range(60):
            for j in range(4):
                grad_a[1][k, 0] += self.w[2][j, k] * sigmoid_deriv(z[2][j, 0]) * (2 * a[2][j, 0] - 2 * y[1][j, 0])

        # Second layer
        # Grad_w2
        for j in range(60):
            for k in range(150):
                grad_w[1][j, k] += grad_a[1][j, 0] * sigmoid_deriv(z[1][j, 0]) * a[0][k, 0]
        # Grad_b2
        for j in range(60):
            grad_b[1][j, 0] += sigmoid_deriv(z[1][j, 0]) * grad_a[1][j, 0]
        # Grad_a1
        for k in range(150):
            for j in range(60):
                grad_a[0][k, 0] += self.w[1][j, k] * sigmoid_deriv(z[1][j, 0]) * grad_a[1][j, 0]

        # First layer
        # Grad_w1
        for j in range(150):
            for k in range(102):
                grad_w[0][j, k] += grad_a[0][j, 0] * sigmoid_deriv(z[0][j, 0]) * y[0][k]
        # Grad_b1
        for j in range(150):
            grad_b[0][j, 0] += sigmoid_deriv(z[0][j, 0]) * grad_a[0][j, 0]
        return grad_w, grad_b, grad_a

    def check_true_prediction(self, a, y):
        if np.argmax(a) == np.argmax(y):
            return 1
        else:
            return 0

    def training_nonvectorized(self, train):
        mean_costs = []
        true_predicted = 0

        start_time = time.time()

        for i in range(self.epochs):
            random.shuffle(train)
            #a0, y = shuffling_train_set(train[0:200])
            cost_per_epoch = 0
            for batch in range(int(len(train) / self.batch_size)):
                # Allocating grad_w and grad_b and grad_w
                grad_w3 = np.zeros((4, 60))
                grad_w2 = np.zeros((60, 150))
                grad_w1 = np.zeros((150, 102))
                grad_w = [grad_w1, grad_w2, grad_w3]

                grad_b3 = np.zeros((4, 1))
                grad_b2 = np.zeros((60, 1))
                grad_b1 = np.zeros((150, 1))
                grad_b = [grad_b1, grad_b2, grad_b3]

                grad_a3 = np.zeros((4, 1))
                grad_a2 = np.zeros((60, 1))
                grad_a1 = np.zeros((150, 1))
                grad_a = [grad_a1, grad_a2, grad_a3]

                for b_ in range(batch * self.batch_size, (batch + 1) * self.batch_size):

                    # Compute the output for the image
                    a, z = self.feedforward(train[b_][0])
                    grad_w, grad_b, grad_a = self.backpropagation(grad_w, grad_b, grad_a, a, z, train[b_])

                    true_predicted += self.check_true_prediction(a[2], train[b_][1])

                    cost_per_epoch += calculating_cost(a[2], train[b_][1])

                    print(b_, cost_per_epoch)

                for k in range(3):
                    self.w[k] -= (grad_w[k] / self.batch_size) * self.learning_rate
                    self.b[k] -= (grad_b[k] / self.batch_size) * self.learning_rate

            mean_costs.append(cost_per_epoch / len(train))
        print("Total time: ", time.time() - start_time)
        print("accuracy is: ", self.calculate_accuracy(train))
        plotting_costs(mean_costs)

    def training_vectorized(self, train):
        predicted_output = []
        mean_costs = []
        true_predicted = 0

        start_time = time.time()

        for i in range(self.epochs):
            random.shuffle(train)
            #a0, y = shuffling_train_set(train[0:200])
            cost_per_epoch = 0
            for batch in range(int(len(train) / self.batch_size)):
                # Allocating grad_w and grad_b and grad_w
                grad_w3 = np.zeros((4, 60))
                grad_w2 = np.zeros((60, 150))
                grad_w1 = np.zeros((150, 102))
                grad_w = [grad_w1, grad_w2, grad_w3]

                grad_b3 = np.zeros((4, 1))
                grad_b2 = np.zeros((60, 1))
                grad_b1 = np.zeros((150, 1))
                grad_b = [grad_b1, grad_b2, grad_b3]

                grad_a3 = np.zeros((4, 1))
                grad_a2 = np.zeros((60, 1))
                grad_a1 = np.zeros((150, 1))
                grad_a = [grad_a1, grad_a2, grad_a3]

                for b_ in range(batch * self.batch_size, (batch + 1) * self.batch_size):

                    # Compute the output for the image
                    a, z = self.feedforward(train[b_][0])
                    grad_w, grad_b, grad_a = self.vectorized_backpropagation(grad_w, grad_b, grad_a, a, z, train[b_])

                    true_predicted += self.check_true_prediction(a[2], train[b_][1])
                    cost_per_epoch += calculating_cost(a[2], train[b_][1])
                    print(b_, cost_per_epoch)

                for k in range(3):
                    self.w[k] -= (grad_w[k] / self.batch_size) * self.learning_rate
                    self.b[k] -= (grad_b[k] / self.batch_size) * self.learning_rate

            mean_costs.append(cost_per_epoch / len(train))
        print("Total time: ", time.time() - start_time)
        print("accuracy is: ", self.calculate_accuracy(train))
        plotting_costs(mean_costs)

    def calculate_accuracy(self, data):
        true_predict = 0
        for i in range(len(data)):
            if np.argmax(self.feedforward(data[i][0])[0][2]) == np.argmax(data[i][1]):
                true_predict = true_predict + 1
        return (true_predict / len(data))

    def predicting_test_data(self, test):
        mean_costs = []
        true_predicted = 0

        start_time = time.time()

        for batch in range(int(len(test) / 66.2)):
            cost_per_epoch = 0
            for b_ in range(batch * self.batch_size, (batch + 1) * self.batch_size):
                # Compute the output for the image
                a, z = self.feedforward(test[b_][0])
                true_predicted += self.check_true_prediction(a[2], test[b_][1])
                cost_per_epoch += calculating_cost(a[2], test[b_][1])
                print(b_, cost_per_epoch)
            mean_costs.append(cost_per_epoch / self.batch_size)
        print("Total time: ", time.time() - start_time)
        print("Test accuracy is: ", self.calculate_accuracy(test))
        plotting_costs(mean_costs)


n = NeuralNetwork(0.5, 100, 5)
#n.training_with_feedforward(train[:200])
#n.training_nonvectorized(train[:200])
#n.training_vectorized(train[:200])
n.training_vectorized(train)
n.predicting_test_data(test)