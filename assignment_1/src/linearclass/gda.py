import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    import matplotlib.pyplot as plt
    def plotBoundry(x, y, gdaModel, save_path):
        """Plot dataset and fitted GDA parameters.

        Args:
            x: Matrix of training examples, one per row.
            y: Vector of labels in {0, 1}.
            gdaModel: GDA model.
            save_path: Path to save the plot.
        """
        # Plot dataset
        plt.figure()
        plt.plot(x[y == 1, 0], x[y == 1, 1], 'bx', linewidth=10)
        plt.plot(x[y == 0, 0], x[y == 0, 1], 'go', linewidth=10)

        # Plot decision boundary (found by solving for theta^T x = 0)
        x1 = np.arange(min(x[:, 0]), max(x[:, 0]), 0.01)
        x2 = np.array([(gdaModel.theta[1][1] ** 2 - gdaModel.theta[2][1] ** 2 +
               (x1_obs - gdaModel.theta[1][0]) ** 2 - (x1_obs - gdaModel.theta[2][0]) ** 2)
              / (2 * gdaModel.theta[1][1] - 2 * gdaModel.theta[2][1])
              for x1_obs in x1])
        plt.plot(x1, x2, c='red', linewidth=2)
        plt.xlim(x[:, 0].min() - .1, x[:, 0].max() + .1)
        plt.ylim(x[:, 1].min() - .1, x[:, 1].max() + .1)

        # Add labels and save to disk
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.savefig(save_path)

    x_val, y_val = util.load_dataset(valid_path, add_intercept=False)
    clf_gda = GDA()
    clf_gda.fit(x_train, y_train)
    validLableArray = clf_gda.predict(x_val)
    np.savetxt(save_path, validLableArray)
    # printMat = np.c_[np.array(x_val), np.array(y_val), np.array(validLableArray)]
    # print(printMat)

    plotBoundry(x_val, np.array(y_val), clf_gda, 'plot_' + save_path[:(len(save_path) - 4)] + '.png')
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        numObs, dimObs = x.shape
        # theta: phi, mu_0, mu_1, sigma
        self.theta = [0, np.zeros(dimObs), np.zeros(dimObs), np.zeros([dimObs, dimObs])]
        self.theta[0] = sum([y_i == 1 for y_i in y]) / numObs
        self.theta[1] = np.array([sum([x[idxObs, idxDim] for idxObs in range(numObs) if y[idxObs] == 0])
                                  / (numObs * (1 - self.theta[0]))
                         for idxDim in range(dimObs)])
        self.theta[2] = np.array([sum([x[idxObs, idxDim] for idxObs in range(numObs) if y[idxObs] == 1])
                                  / (numObs * self.theta[0])
                         for idxDim in range(dimObs)])

        deviationList = np.array([[x[idxObs, idxDim] - self.theta[1][idxDim] if y[idxObs] == 0 else x[idxObs, idxDim] - self.theta[2][idxDim]
                          for idxDim in range(dimObs)]
                         for idxObs in range(numObs)])
        for idxObs in range(numObs):
            covObs = np.outer(deviationList[idxObs], deviationList[idxObs]) / numObs
            self.theta[3] = np.add(self.theta[3], covObs)



        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        predList = []
        sigma_inv = np.linalg.inv(self.theta[3])
        for obs in x:
            prob_0 = np.matmul(np.matmul(np.array(obs) - self.theta[1], sigma_inv),
                               np.array(np.array(obs) - self.theta[1]).transpose())
            prob_1 = np.matmul(np.matmul(np.array(obs) - self.theta[2], sigma_inv),
                               np.array(np.array(obs) - self.theta[2]).transpose())
            thisLabel = 0 if prob_0 < prob_1 else 1
            predList.append(thisLabel)
        return(np.array(predList))




        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
