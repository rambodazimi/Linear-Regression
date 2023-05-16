# a simple linear regression program which predicts the price of a house based on its size
import numpy as np
import matplotlib.pyplot as plt

# creating our sample dataset with 2 entries
x_trains = np.array([1, 2]) # house size
y_trains = np.array([300, 500]) # house price
m = x_trains.shape[0] # number of training examples

# f = wx + b (w and b are constants). The main goal of linear regression is to find a best fit line,
# which has the mentioned equation

# setting w and b to some values (actually the hardest part of linear regression is to find these constants!)
w = 200
b = 100
# we could use cost function (squared error cost function) to see how well the function f fits the datapoints
f = np.zeros(m)

# go over each data point and find the equation
for i in range(m):
    f[i] = w*x_trains[i] + b


# Now, let's compute the cost function to see how well does the calculated function f response
# Note that lower cost function means we have a better function f which fits the dataset
sum = 0
for i in range(m):
    sum += (f[i] - y_trains[i])**2

sum /= (2*m)

print(f"The value of the cost function is: {sum}")


# plotting the result
plt.plot(x_trains, f, c='b', label="Our prediction")
plt.scatter(x_trains, y_trains, marker='x', c='r', label="Actual values")
plt.title("Housing prices")
plt.xlabel(f"Size\nCost function: {sum}")
plt.ylabel("Price")
plt.legend()
plt.show()


