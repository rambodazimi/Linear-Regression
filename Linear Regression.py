# Author: Rambod Azimi
# May 2023

"""
Linear Regression (supervised learning algorithm) using Gradient Descent and Cost Function in Python

This program will make use of a simple dataset consisting of one input feature variable (house size)
and one output target variable (house price). Then, it finds a function f = wx+b which fits the datapoints
using Gradient Descent algorithm.
Finally, we check the function with cost function to see how well did the function fits the data

I decided to use numpy library in Python because it has many built-in methods to deal with scientific computation
in this program as well as matplotlib to plot the results in Python
"""


# importing the python libraries that we need in this program
import numpy as np
import matplotlib.pyplot as plt
import math, copy


# a function which calculates the squared error cost function to see how well did the function fit the data
def compute_cost_function(x, y, w, b):
    sum = 0
    m = x.shape[0] # size of the dataset
    for i in range(m): # iterate over each element and adds the calculated value to the variable sum
        sum += (w * x[i] + b - y[i]) ** 2

    sum /= (2*m)
    return sum


def compute_gradient_descent(x, y, w, b):
    m = x.shape[0] # size of the dataset
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f = x[i] * w + b
        # storing the result in temp variables befcause both w and b need to be upodated simultaneously
        dj_dw_temp = f - y[i] * x[i]
        dj_db_temp = f - y[i]

        # now, storing the calculated values to their actual variable at the same time
        dj_dw = dj_dw_temp
        dj_db = dj_db_temp

    # divide by m at the end
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db # tuple


def gradient_descent(x, y, w_in, b_in, alpha, iters):

    w = w_in
    b = b_in
    
    for i in range(iters): # repeaqt the same process for a specified number of times if not convergent
        dj_dw, dj_db = compute_gradient_descent(x, y, w, b)

        # update the parameters w and b
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        sum = compute_cost_function(x, y, w, b)
        print(f"The value of the cost function is: {sum}")


# Now that we implemented the cost function and gradient descent algorithms above, it's time to
# complete the main function

def main():
    print("This program uses linear regression to predict the price of a house based on a given training set")
    print("Author: Rambod Azimi | May 2023")

    
    # defining our sample dataset
    x_train = np.array([1.0, 2.0]) # features (input)
    y_train = np.array([300.0, 500.0]) # targets (output)
    m = x_train.shape[0] # size of the dataset


    print("\nThe given dataset:")

    for i in range(m):
        print(f"House size: {x_train[i]} \t House price: {y_train[i]}")


    w_init = 0
    b_init = 0
    iterations = 100
    alpha_value = 0.01
    gradient_descent(x_train, y_train, w_init, b_init, alpha_value, iterations)


if __name__ == "__main__":
    main()
