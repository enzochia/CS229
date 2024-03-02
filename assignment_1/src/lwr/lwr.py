import matplotlib.pyplot as plt
import numpy as np
import util


def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    # Fit a LWR model
    lwrRegressor = LocallyWeightedLinearRegression(0.5)
    lwrRegressor.fit(x_train, y_train)
    y_pred = []
    for idxObs in range(x_valid.shape[0]):
        # print(lwrRegressor.predict(x_valid[idxObs])[0, 0])
        y_pred.append(lwrRegressor.predict(x_valid[idxObs])[0, 0])
    y_pred = np.array(y_pred)
    # yPredList = [(y_pred[idxObs], y_valid[idxObs]) for idxObs in range(x_valid.shape[0])]
    # print(np.matrix(yPredList))
    # Get MSE value on the validation set
    mse = (np.square(y_pred - y_valid)).mean()
    print(f'MSE: {mse}')
    # Plot validation predictions on top of training set
    import matplotlib.pyplot as plt
    def plotData(x_train, y_train, x_valid, y_valid, save_path):
        """Plot dataset and fitted logistic regression parameters.
        Args:
            x: Matrix of training examples, one per row.
            y: Vector of labels in {0, 1}.
            theta: Vector of parameters for logistic regression model.
            save_path: Path to save the plot.
            correction: Correction factor to apply, if any.
        """
        # Plot dataset
        plt.figure()
        plt.plot(x_train[:, 1], y_train, 'bx', linewidth=1)
        plt.plot(x_valid[:, 1], y_valid, 'ro', linewidth=1)

        plt.xlim(x_train[:, 1].min() - .1, x_train[:, 1].max() + .1)
        plt.ylim(y_train.min() - .1, y_train.max() + .1)

        # Add labels and save to disk
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(save_path)
    plotData(x_train, y_train, x_valid, y_pred, 'lwr_pred_val.png')
    plotData(x_train, y_train, x_valid, y_valid, 'lwr_actual_val.png')
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***


        numObs, dimObs = self.x.shape
        wMat = np.zeros((numObs, numObs))

        for idxDiag in range(numObs):
            wMat[idxDiag, idxDiag] = np.exp(((-1) / (2 * self.tau ** 2)) *
                                            (np.linalg.norm(self.x[idxDiag] - x) ** 2))
        xMat = np.matrix(self.x)
        xMatT = xMat.transpose()
        theta = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(xMatT, wMat),
                                                                      xMat)),
                                              xMatT),
                                    wMat),
                          np.matrix(self.y).transpose())

        yPred = np.matmul(np.matrix(x), theta)
        return(yPred)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='./train.csv',
         eval_path='./valid.csv')
