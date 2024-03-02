import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    # Search tau_values for the best tau (lowest MSE on the validation set)
    def plotData(x_train, y_train, x_valid, y_valid, save_path):
        """Plot datasets
        Args:
            x: Matrix of training examples, one per row.
            y: Vector of labels in {0, 1}.
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
    def validateTestTau(x_valid, y_valid, tau, x_train, y_train):
        lwrRegressor = LocallyWeightedLinearRegression(tau)
        lwrRegressor.fit(x_train, y_train)
        y_pred = []
        for idxObs in range(x_valid.shape[0]):
            y_pred.append(lwrRegressor.predict(x_valid[idxObs])[0, 0])
        y_pred = np.array(y_pred)
        mse = (np.square(y_pred - y_valid)).mean()
        plotData(x_train, y_train, x_valid, y_pred, 'pred_tau' + str(tau) +'.png')
        return(mse)
    mseList = []
    bestTau, bestMSE = 0, float('inf')
    for tau in tau_values:
        mse = validateTestTau(x_valid, y_valid, tau, x_train, y_train)
        if mse < bestMSE:
            bestMSE = mse
            bestTau = tau
        mseList.append(mse)
    # Fit a LWR model with the best tau value
    lwrRegressor = LocallyWeightedLinearRegression(bestTau)
    lwrRegressor.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    y_pred = []
    for idxObs in range(x_test.shape[0]):
        y_pred.append(lwrRegressor.predict(x_test[idxObs])[0, 0])
    y_pred = np.array(y_pred)
    mse = (np.square(y_pred - y_test)).mean()
    print(f'All MSEs: {mseList}')
    print(f'MSE for best tau which is {bestTau}: {mse}')

    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)
    # Plot data
    plotData(x_train, y_train, x_test, y_pred, 'test_tau' + str(bestTau) + '.png')
    plotData(x_train, y_train, x_test, y_test, 'test_data.png')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
