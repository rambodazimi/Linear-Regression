# Author: Rambod Azimi
# May 2023

"""
Linear Regression (supervised learning algorithm) using Gradient Descent and Cost Function in Python
This program will make use of a very popular machiine learning library in Python called scikit-learn
It uses SGDRegressor which is a linear regression model using the Gradient Descent algorithm
Also, it uses StandardScaler to scale the featyure variables to better fit to the data model
"""

# importing the python libraries that we need in this program
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor # linear regression model library in scikit-learn
from sklearn.preprocessing import StandardScaler # Z-score scaling library in scikit-learn

def main():
    print("Linear regression using scikit-learn library in Python")
    print("Author: Rambod Azimi | May 2023")
    print()

    # defining the training data
    x_train = np.array([[100, 2, 1, 20], [200, 4, 2, 12], [300, 5, 2, 6], [43, 1, 1, 10], [170, 3, 1, 7]])
    # x1: size of the house (square m), x2: number of bedrooms, x3: number of floors, x4: age of the house

    x_features = np.array(['size', 'bedrooms', 'floors', 'age'])

    """
    Because we have 4 feature variables x1, x2, x3, and x4, our function looks like the following:
    f = w1x1 + w2x2 + w3x3 + w4x4 + b
    Therefore, we have to find the values for 5 parameters w1, w2, w3, w4, and b
    """

    y_train = np.array([200000, 400000, 900000, 130000, 300000]) # house price

    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x_train) # normalized (Z-score) version of x_train 

    gradient = SGDRegressor(max_iter=int(1e07)) # defining the gradient descent regressor and setting the max number of iterations to 1000

    gradient.fit(x_normalized, y_train)
    print(f"The algorithm completed {gradient.n_iter_} iterations in total.")

    w = gradient.coef_
    b = gradient.intercept_
    print(f"The parameter values of W are: {w}")
    print(f"The parameter value of b is: {b}")

    # printing the calculated function (linear)
    print(f"f = {w[0]}x1 + {w[1]}x2 + {w[2]}x3 + {w[3]}x4 + {b[0]}")

    # now, let's predict the target values to see if the predicted values are close to the actual valus or not
    actual_values = y_train
    predicted_values = gradient.predict(x_normalized)
    f = np.dot(x_normalized, w) + b
    print((f == predicted_values).all()) # True means they match

    print(f"Prediction: {predicted_values}")
    print(f"Actual Target Values: {actual_values}")

    # plot predictions and targets vs original features    
    fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(x_train[:,i],y_train, label = 'target')
        ax[i].set_xlabel(x_features[i])
        ax[i].scatter(x_train[:,i],predicted_values, label = 'predict')
    ax[0].set_ylabel("Price")
    ax[0].legend()
    fig.suptitle("target versus prediction using z-score normalized model")
    plt.show()

if __name__ == "__main__":
    main()