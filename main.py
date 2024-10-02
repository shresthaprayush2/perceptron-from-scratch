from Preceptron import Perceptron  # Import the Perceptron class
import numpy as np  # Import NumPy for handling arrays

# Define the training data (inputs for AND gate)
Xtrain = np.array([
    [0, 0],  # Both inputs are 0
    [0, 1],  # One input is 0, the other is 1
    [1, 0],  # One input is 1, the other is 0
    [1, 1],  # Both inputs are 1
])

# Define the corresponding labels (outputs for AND gate)
ytrain = np.array([0, 0, 0, 1])  # AND gate only returns 1 when both inputs are 1

# Define the test data (new inputs to check after training)
Xtest = np.array([
    [1, 0],  # Testing input with one 1 and one 0
    [1, 1]   # Testing input with both 1s
])

# Initialize the Perceptron with 2 input features
model = Perceptron(input_size=2)

# Train the Perceptron model using the training data
model.train(Xtrain, ytrain)

# Test the model with new inputs and print the predicted values
for inp in Xtest:
    print(f'Input: {inp}, Predicted Value: {model.predict(inp)}')