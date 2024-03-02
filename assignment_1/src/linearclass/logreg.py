import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    clf_logReg = LogisticRegression()
    clf_logReg.fit(x_train, y_train)
    validLableArray = clf_logReg.predict(x_val)
    np.savetxt(save_path, validLableArray)
    util.plot(np.array(x_val[:, 1:]), np.array(y_val), clf_logReg.theta,
              'plot_' + save_path[:(len(save_path) - 4)] + '.png')
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.`

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        # self.eps *= 1
        # case1: [-2.2571775093031685, 1.1549738707688126, 0.1726321295146096]
        # case2: [-2.315926615087517, 0.9619998044271986, 0.9049935740869497]
        # self.eps *= 10
        # case1: [-2.628025141803172, 0.8091487081200437, 0.38349722874658543]
        # case2: [-2.6780377961945754, 1.1326464906275564, 1.0209874334562643]
        # self.eps *= 50
        # case1: [-1.799753805448378, 1.048254096952203, 0.0936434812044019]
        # case2: [-2.826168108941117, 1.198299492610236, 1.0753366133325712]
        # self.eps *= 100
        # case1: [-3.3333014526807307, 1.0024374676626713, 0.4813713928703548]
        # case2: [-2.9604789089713415, 1.2551592311064035, 1.1282066974841891]
        # self.eps *= 200
        # case1: [-3.5282383148988377, 0.9944992928377696, 0.5241256791051411]
        # case2: [-3.1058156650113298, 1.3165432113116227, 1.1849271769899983]
        # self.eps *= 1000
        # case1: [-1.2851434824642904, 0.9535905567969644, 0.016835758383769318]
        # case2: [-4.01802635085829, 1.685591903159928, 1.5582048505708146]


        numObs, dimObs = x.shape
        self.theta, gradient_theta, = [0] * dimObs, [0] * dimObs
        hessian_theta = [[0] * dimObs for _ in range(dimObs)]
        stepLen = np.abs(self.eps * 10000)
        numIter = 0

        while ((stepLen > self.eps) and
               (numIter < self.max_iter)):
            numIter += 1

            for idx, obs_i in enumerate(x):
                theta_multiply_x = np.matmul(self.theta, obs_i)
                h_theta_x = 1 / (1 + np.exp(-theta_multiply_x))
                gradient_theta = [gradient_theta[idx_dim] - (y[idx] - h_theta_x) * obs_i[idx_dim] / numObs
                                  for idx_dim in range(len(obs_i))]
                hessian_theta = [[hessian_theta[idx_row][idx_col] + (obs_i[idx_row] * obs_i[idx_col] *
                                                                    h_theta_x * (1 - h_theta_x)) / numObs
                                  for idx_col in range(dimObs)]
                                 for idx_row in range(dimObs)]
            hessian_theta_inv = np.linalg.inv(hessian_theta)
            gradientStep = np.matmul(np.matrix(gradient_theta),
                                               hessian_theta_inv)

            stepLen = np.linalg.norm(gradientStep)
            self.theta = [self.theta[idx_dim] - gradientStep[0, idx_dim] for idx_dim in range(dimObs)]
        # print(numIter)
        # print(self.theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        x = [[1] + obs_i for obs_i in x]
        numObs = len(x)
        yHat = [1 / (1 + np.exp(-np.matmul(self.theta, np.matrix(x[idxObs]).transpose()))[0, 0])
                for idxObs in range(numObs)]
        yHatArray = np.array(yHat)
        return(yHatArray)



        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
