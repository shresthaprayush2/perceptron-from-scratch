import numpy as np  # Import NumPy for mathematical operations like dot products


class Perceptron:

    # Constructor to initialize the perceptron
    def __init__(self, input_size, learning_rate=0.01, epoch=100):
        """
        Parameters:
        - input_size: Number of features (or inputs) the perceptron will use
        - learning_rate: Controls how much we adjust the weights with each step
        - epoch: Number of times the training loop will run
        """
        # Initialize weights as a zero vector, adding +1 for the bias term
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epoch = epoch

    # Step/Activation function to determine output
    def activation_function(self, x):
        """
        Activation function: A step function that returns:
        - 1 if the input is greater than or equal to 0
        - 0 if the input is less than 0
        """
        if x >= 0:
            return 1
        else:
            return 0

    # Predict function to calculate the perceptron's output for a given input
    def predict(self, x):
        """
        Prediction function:
        - Calculates the weighted sum of the inputs (z = w*x + b)
        - Passes the result through the activation function to get the output (1 or 0)

        Parameters:
        - x: Input feature vector

        Returns:
        - 1 or 0 (the prediction)
        """
        # z is the weighted sum of inputs + bias (w0)
        z = np.dot(x, self.weights[1:]) + self.weights[0]  # Bias weight is self.weights[0]
        # Return the activation function's result (either 0 or 1)
        return self.activation_function(z)

    # Train function to adjust the weights based on the training data
    def train(self, X, y):
        """
        Training function:
        - Adjusts the weights based on the difference between predicted and actual labels
        - Repeats the process for a given number of epochs

        Parameters:
        - X: Feature matrix (each row is a sample's input vector)
        - y: Target labels (actual outputs)
        """
        # Loop through each epoch (number of training iterations)
        for epoch in range(self.epoch):
            print(f'Epoch {epoch + 1}')  # Trace the epoch number for debugging
            # For each input-output pair in the training set
            for inputs, label in zip(X, y):
                # Predict the output using current weights
                prediction = self.predict(inputs)
                # Trace the inputs, predictions, and current weights
                print(f'Input: {inputs}, Prediction: {prediction}, Actual: {label}, Weights: {self.weights}')
                # Update the weights based on the prediction error (label - prediction)
                # For the weights corresponding to inputs (w1, w2,...)
                self.weights[1:] = self.weights[1:] + self.learning_rate * (label - prediction) * inputs
                # For the bias weight (w0)
                self.weights[0] += self.learning_rate * (label - prediction)
            # Trace the updated weights after each epoch
            print(f'Updated Weights: {self.weights}\n')