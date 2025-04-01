import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate, weights, threshold):
        self.learning_rate = learning_rate
        self.weights = np.array(weights)
        self.threshold = threshold

    def activation(self, x):
        # sgn activation function
        return np.where(x >= self.threshold, 1, -1)

    def predict(self, inputs):
        # Predict result
        weighted_sum = np.dot(inputs, self.weights)
        return self.activation(weighted_sum)

    def fit(self, X, y, epochs):
        for epoch in range(epochs):
            for input_vector, label in zip(X, y):
                prediction = self.predict(input_vector)
                # Update weights
                self.weights += self.learning_rate * (label - prediction) * input_vector
            
            # Plot decision boundary at the end of each epoch
            self.plot_decision_boundary(X, y, epoch)

    def plot_decision_boundary(self, X, y, epoch):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        # Predict values on the grid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
        plt.title(f'Perceptron Decision Boundary (Epoch {epoch + 1})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid()
        plt.show()

# Data generation
def generate_data():
    # Randomly generate 100 points
    np.random.seed(0)  # Fix random seed for reproducibility
    x1 = np.random.randn(50, 2) + np.array([1, 1])  # Positive class
    x2 = np.random.randn(50, 2) + np.array([-1, -1])  # Negative class
    X = np.vstack((x1, x2))
    y = np.array([1] * 50 + [-1] * 50)  # Labels
    return X, y

if __name__ == "__main__":
    # Parameter settings
    learning_rate = 0.8
    initial_weights = [1, 0.2]
    threshold = -1

    # Generate data
    X, y = generate_data()

    # Create perceptron
    perceptron = Perceptron(learning_rate, initial_weights, threshold)

    # Train perceptron
    perceptron.fit(X, y, epochs=10)