import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    poissonRegressor = PoissonRegression(step_size=lr)
    poissonRegressor.fit(x_train, y_train)
    validLableArray = poissonRegressor.predict(x_val)
    np.savetxt(save_path, validLableArray)

    def plotData(y_pred, y_valid, save_path):
        """Plot dataset
        Args:
            y_pred: predicted counts.
            y_valid: observed true counts.
            save_path: Path to save the plot.
        """
        # Plot dataset
        plt.figure()
        plt.plot(y_valid, y_pred, 'bx', linewidth=1)

        plt.xlim(y_valid.min() - .1, y_valid.max() + .1)
        plt.ylim(y_pred.min() - .1, y_pred.max() + .1)

        plt.xlabel('observed count')
        plt.ylabel('predicted count')
        plt.savefig(save_path)
    plotData(validLableArray, y_val, save_path[:-4] + '.png')

    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        numObs, dimObs = x.shape
        # siteLambda = np.mean(y)
        self.theta, gradient_theta, = [0] * dimObs, [0] * dimObs
        stepLen = np.abs(self.eps * 10000)
        numIter = 0
        xMat = np.matrix(x)
        xMatT = xMat.transpose()

        while ((stepLen > self.eps) and
               (numIter < self.max_iter)):
            numIter += 1
            gradientTerm3 = np.exp(np.array(np.matmul(np.matrix(self.theta), xMatT))[0])
            gradientTerm2 = np.array(np.matmul(np.matrix(self.step_size * np.add(y, -1 * gradientTerm3)),
                                               xMat))[0]
            self.theta = np.add(self.theta, gradientTerm2)
            stepLen = np.linalg.norm(gradientTerm2)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        predArray = np.exp(np.array(np.matmul(np.matrix(self.theta),
                                              np.matrix(x).transpose()))[0])
        return(predArray)

        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
