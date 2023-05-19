# Author: Rambod Azimi
# May 2023

"""
Linear Regression (supervised learning algorithm) using Gradient Descent and Cost Function in Python

This program will make use of a simple dataset consisting of multiple input feature variables (4 features)
(house size, number of bedrooms, number of floors, age of home) and one output target variable (house price).
Then, it finds a function f = w1x1 + w2x2 + w3x3 + w4x4 + b which fits the datapoints using Gradient Descent.
"""


# importing the python libraries that we need in this program
import numpy as np
import matplotlib.pyplot as plt
import math, copy

def main():
    print("This program uses multiple variable linear regression to predict the price of a house based on a given training set")
    print("Author: Rambod Azimi | May 2023")
    
    # defining our sample dataset
    x_train = np.array([[2104, 5, 1, 45],
                        [1416, 3, 2, 40], 
                        [852, 2, 1, 35]])
    # each row of x_train represents one example

    y_train = np.array([460, 232, 178])
    m = x_train.shape[0] # number of training examples

    print("\nThe given dataset:")
    for i in range(m):
        print(f"House size: {x_train[i]} \t House price: {y_train[i]}")

    # initializing variables
    w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618]) # now, W is a vector which has 4 elements
    b_init = 785.1811367994083
    alpha = 5.0e-7
    iterations = 1000

    w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)
    print(f"b, w found by gradient descent: {b_final}, {w_final}")

    for k in range(m):
        print(f"prediction: {np.dot(x_train[i], w_final) + b_final}, target value: {y_train[i]}")

# simple function to compute the f = w1x1 + w2x2 + ..... + wnxn + b (or simply w.x + b)
# this implementation is slow becasue it uses a for loop to compute the dot product
def predict_single_loop(x, w, b):
    n = x.shape[0] # number of elements
    result = 0
    for i in range(n):
        result += x[i] * w[i]
    result += b
    return result


# a faster version of predict_single_loop() function which utilizes dot() in NumPy library which is much faster
def predict (x, w, b):
    result = np.dot(x, w) + b
    return result


# computing the cost function to see how well does the algorithm predict the house price
def compute_cost(x, y, w, b):
    m = x.shape[0] # number of elements
    cost = 0
    for i in range(m):
        f = np.dot(x[i], w) + b
        cost += (f - y[i]) ** 2
    cost /= (2*m)
    return cost


# compute gradient function which finds both dj/dw and dj/db to be used in the actual gradient descent algorithm
def compute_gradient(x, y, w, b):
    m = x.shape[0] # number of training examples
    n = x.shape[1] # number of features
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        f = np.dot(x[i], w) + b
        error = f - y[i]

        for j in range(n):
            dj_dw[j] += error * x[i, j]
        dj_db += error
        dj_dw /= m
        dj_db /= m

        return dj_db, dj_dw


def gradient_descent(x, y, w_in, b_in, alpha, iterations):

    # copying the parameters into temp variables to make sure the actual variables do not change
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(iterations):
        dj_db, dj_dw = compute_gradient(x, y, w, b)

        # updating both parameters w and b (vectorization)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b


if __name__ == "__main__":
    main()
