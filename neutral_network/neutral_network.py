
import numpy as np
import matplotlib.pyplot as plt


def relu(Z):
    return np.maximum(0, Z)


def gradient_relu(Z):
    return (Z > 0).astype(float)


def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


class NeuralNetwork:

    def _initialize(self, input_size, list_layers):
        self.layers = len(list_layers)

        layer_sizes = [input_size] + list_layers

        self.W = []
        self.B = []

        for i in range(self.layers):
            # He initialization for ReLU layers
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            if i == self.layers - 1:
                wi = np.random.randn(fan_out, fan_in) * np.sqrt(1 / fan_in)
            else:
                wi = np.random.randn(fan_out, fan_in) * np.sqrt(2 / fan_in)

            bi = np.zeros(fan_out)

            self.W.append(wi)
            self.B.append(bi)

    def _forward(self, X):
        self.A = []
        self.Z = []

        A = X

        for i in range(self.layers - 1):
            Z = A @ self.W[i].T + self.B[i]
            A = relu(Z)

            self.Z.append(Z)
            self.A.append(A)

        # Output layer: Softmax
        Z = self.A[-1] @ self.W[-1].T + self.B[-1]
        A = softmax(Z)

        self.Z.append(Z)
        self.A.append(A)

        return A

    def _loss(self, pred, Y):
        pred = np.clip(pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(Y * np.log(pred), axis=1))

    def fit(
        self,
        X,
        Y,
        layers=None,
        list_layers=None,
        learning_rate=0.001,
        epochs=1,
        batch_size=10
    ):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be 2D arrays with the same number of samples.")

        if list_layers is None:
            raise ValueError("list_layers must be provided.")

        if layers is not None and layers != len(list_layers):
            raise ValueError("layers must equal len(list_layers).")

        if list_layers[-1] != Y.shape[1]:
            raise ValueError("The last layer size must equal the number of classes.")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self._initialize(X.shape[1], list_layers)

        self.loss_history = []
        N = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(N)
            total_loss = 0

            for start in range(0, N, batch_size):
                batch_indices = indices[start:start + batch_size]

                x = X[batch_indices]
                y = Y[batch_indices]

                m = x.shape[0]

                # 1. Forward propagation
                pred = self._forward(x)

                total_loss += self._loss(pred, y) * m

                # 2. Backward propagation
                # Softmax + Cross Entropy
                E = (pred - y) / m

                grad_W = [None] * self.layers
                grad_B = [None] * self.layers

                grad_W[-1] = E.T @ self.A[-2]
                grad_B[-1] = np.sum(E, axis=0)

                for l in range(self.layers - 2, 0, -1):
                    E = (E @ self.W[l + 1]) * gradient_relu(self.Z[l])

                    grad_W[l] = E.T @ self.A[l - 1]
                    grad_B[l] = np.sum(E, axis=0)

                # First hidden layer
                E = (E @ self.W[1]) * gradient_relu(self.Z[0])

                grad_W[0] = E.T @ x
                grad_B[0] = np.sum(E, axis=0)

                # 3. Update parameters
                for j in range(self.layers):
                    self.W[j] -= learning_rate * grad_W[j]
                    self.B[j] -= learning_rate * grad_B[j]

            epoch_loss = total_loss / N
            self.loss_history.append(epoch_loss)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {epoch_loss:.6f}"
            )

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = self._forward(X)
        return np.argmax(pred, axis=1)

    def accuracy(self, X, Y):
        pred = self.predict(X)
        true = np.argmax(Y, axis=1)

        return np.mean(pred == true)

    def print_wrong(self, X, Y):
        pred = self.predict(X)
        true = np.argmax(Y, axis=1)

        wrong = np.where(pred != true)[0][:10]

        if len(wrong) == 0:
            print("No incorrect predictions.")
            return

        fig, axes = plt.subplots(2, 5, figsize=(12, 6))

        for ax in axes.flat:
            ax.axis("off")

        for ax, i in zip(axes.flat, wrong):
            ax.imshow(X[i].reshape(28, 28), cmap="gray")
            ax.set_title(f"True: {true[i]} | Pred: {pred[i]}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()
