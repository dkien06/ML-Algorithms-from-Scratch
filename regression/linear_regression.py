
import numpy as np


class LinearRegression:

    def fit(self, X, Y, lr=0.01, epochs=1000, tol=1e-6):
        """
        Train Linear Regression using Batch Gradient Descent.

        Parameters
        ----------
        X : ndarray, shape (N, D)
        Y : ndarray, shape (N,) or (N, 1)
        lr : float
            Learning rate.
        epochs : int
            Maximum number of iterations.
        tol : float
            Convergence threshold.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).reshape(-1, 1)

        if X.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("Invalid input shapes.")

        # Normalize features
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1

        X_scaled = (X - self.mean) / self.std

        # Add bias
        X_new = np.concatenate(
            (np.ones((X.shape[0], 1)), X_scaled),
            axis=1
        )

        self.w = np.zeros((X_new.shape[1], 1))
        N = X_new.shape[0]

        self.loss_history = []

        for epoch in range(epochs):
            grad = self.gradient(X_new, Y, self.w, N)

            self.w -= lr * grad

            loss = self.loss(X_new, Y, self.w)
            self.loss_history.append(loss)

            if np.linalg.norm(grad) < tol:
                break

        self.n_epochs = epoch + 1
        return self

    def gradient(self, X, Y, w, N):
        return (X.T @ (X @ w - Y)) / N

    def loss(self, X, Y, w):
        error = X @ w - Y
        return np.sum(error ** 2) / (2 * X.shape[0])

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if not hasattr(self, "w"):
            raise RuntimeError("Model has not been trained. Call fit() first.")

        X_scaled = (X - self.mean) / self.std

        X_new = np.concatenate(
            (np.ones((X.shape[0], 1)), X_scaled),
            axis=1
        )

        return X_new @ self.w
