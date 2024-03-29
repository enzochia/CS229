import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import multivariate_normal

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    n = x.shape[0]
    numObs = x.shape[0]
    w = np.ones([numObs, K]) * (1.0 / K)
    phi = np.ones(K) * (1.0 / K)
    init_label = np.array([int(x) for x in np.random.uniform(0, K, numObs)])
    # init_cluster = np.array([x[init_label == idx_cluster, :]
    #                          for idx_cluster in range(K)])
    # mu = [init_cluster[idx_cluster].mean(axis=0)
    #       for idx_cluster in range(K)]
    # sigma = [np.cov(init_cluster[idx_cluster][:, 0], init_cluster[idx_cluster][:, 1])
    #          for idx_cluster in range(K)]
    mu, sigma = [], []
    for idx_cluster in range(K):
        cluster = np.array(x[init_label == idx_cluster, :])
        mu.append(cluster.mean(axis=0))
        sigma.append(np.cov(cluster[:, 0],
                            cluster[:, 1]))

    # print(mu)
    # print(sigma)
    # print(phi)
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        print(f'iteration # {it} has log-likelihood {ll}')
        # E STEP
        mvn_list = []
        for idx_cluster in range(K):
            mvn_list.append(multivariate_normal(np.squeeze(np.asarray(mu[idx_cluster])),
                                                sigma[idx_cluster]))
        for idx_obs in range(w.shape[0]):
            density = np.zeros(w.shape[1])
            for idx_cluster in range(w.shape[1]):
                density[idx_cluster] = mvn_list[idx_cluster].pdf(x[idx_obs]) * phi[idx_cluster]
            density = np.divide(density, np.sum(density))
            w[idx_obs] = density

        # M STEP
        phi = w.mean(axis=0)
        mu_list = []
        sigma_list = []
        for idx_cluster in range(K):
            mu_list.append(np.multiply(np.matrix(x),
                                       np.matrix(w[:, idx_cluster]).transpose()).mean(axis=0))
            sigma_this = np.zeros([x.shape[1], x.shape[1]])
            for idx_obs in range(w.shape[0]):
                sigma_this = np.add(sigma_this,
                                    np.multiply(np.matmul(np.matrix(np.add(x[idx_obs],
                                                                           -mu[idx_cluster])).transpose(),
                                                          np.matrix(np.add(x[idx_obs],
                                                                           -mu[idx_cluster]))),
                                                w[idx_obs, idx_cluster]))
            sigma_list.append(np.divide(sigma_this, x.shape[0]))
        mu = []
        sigma = []
        for idx_cluster in range(K):
            mu.append(np.divide(mu_list[idx_cluster],
                                phi[idx_cluster]))
            sigma.append(np.divide(sigma_list[idx_cluster],
                                   phi[idx_cluster]))
        # Calculate log-likelihood
        prev_ll = ll
        ll = 0
        for idx_obs in range(x.shape[0]):
            for idx_cluster in range(K):
                ll += w[idx_obs, idx_cluster] * \
                      (np.log(phi[idx_cluster]) +
                       (-x.shape[1] / 2) * np.log(2 * np.pi) -
                       0.5 * np.log(np.linalg.det(sigma[idx_cluster])) -
                       0.5 * np.matmul(np.matmul(np.matrix(np.add(x[idx_obs],
                                                                  -mu[idx_cluster])),
                                                 np.linalg.inv(sigma[idx_cluster])),
                                       np.matrix(np.add(x[idx_obs],
                                                        -mu[idx_cluster])).transpose())[0, 0] -
                       np.log(w[idx_obs, idx_cluster]))
                       # np.log(w[idx_obs, idx_cluster])) * (1 / x.shape[0])
        it += 1
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        print(f'iteration # {it} has log-likelihood {ll}')
        # E STEP
        mvn_list = []
        for idx_cluster in range(K):
            mvn_list.append(multivariate_normal(np.squeeze(np.asarray(mu[idx_cluster])),
                                                sigma[idx_cluster]))
        for idx_obs in range(w.shape[0]):
            density = np.zeros(w.shape[1])
            for idx_cluster in range(w.shape[1]):
                density[idx_cluster] = mvn_list[idx_cluster].pdf(x[idx_obs]) * phi[idx_cluster]
            density = np.divide(density, np.sum(density))
            w[idx_obs] = density

        # M STEP
        w_sum = w.sum(axis=0)
        phi_list = []
        mu_list = []
        sigma_list = []
        size_tilde = []
        for idx_cluster in range(K):
            z_tilde.astype(int)
            cluster_tilde = np.array(x_tilde[z_tilde[:, 0] == idx_cluster, :])
            phi_list.append((w_sum[idx_cluster] + alpha * cluster_tilde.shape[0]) /
                            (x.shape[0] + alpha * x_tilde.shape[0]) )
            size_tilde.append(cluster_tilde.shape[0])
            mu_tilde = cluster_tilde.sum(axis=0)
            mu_list.append(np.add(np.multiply(np.matrix(x),
                                              np.matrix(w[:, idx_cluster]).transpose()).sum(axis=0),
                                  np.multiply(mu_tilde,
                                              alpha)))



            sigma_this = np.zeros([x.shape[1], x.shape[1]])
            for idx_obs in range(w.shape[0]):
                sigma_this = np.add(sigma_this,
                                    np.multiply(np.matmul(np.matrix(np.add(x[idx_obs],
                                                                           -mu[idx_cluster])).transpose(),
                                                          np.matrix(np.add(x[idx_obs],
                                                                           -mu[idx_cluster]))),
                                                w[idx_obs, idx_cluster]))
            for idx_obs in range(cluster_tilde.shape[0]):
                sigma_this = np.add(sigma_this,
                                    np.multiply(np.matmul(np.matrix(np.add(cluster_tilde[idx_obs],
                                                                           -mu[idx_cluster])).transpose(),
                                                          np.matrix(np.add(cluster_tilde[idx_obs],
                                                                           -mu[idx_cluster]))),
                                                alpha))
            sigma_list.append(sigma_this)
        phi = phi_list
        mu = []
        sigma = []
        for idx_cluster in range(K):
            mu.append(np.divide(mu_list[idx_cluster],
                                w_sum[idx_cluster] + alpha * size_tilde[idx_cluster]))
            sigma.append(np.divide(sigma_list[idx_cluster],
                                   w_sum[idx_cluster] + alpha * size_tilde[idx_cluster]))

        # Calculate log-likelihood
        prev_ll = ll
        ll = 0
        for idx_obs in range(x.shape[0]):
            for idx_cluster in range(K):
                w[idx_obs, idx_cluster] = max(w[idx_obs, idx_cluster], 1e-10)
                ll += w[idx_obs, idx_cluster] * \
                      (np.log(phi[idx_cluster]) +
                       (-x.shape[1] / 2) * np.log(2 * np.pi) -
                       0.5 * np.log(np.linalg.det(sigma[idx_cluster])) -
                       0.5 * np.matmul(np.matmul(np.matrix(np.add(x[idx_obs],
                                                                  -mu[idx_cluster])),
                                                 np.linalg.inv(sigma[idx_cluster])),
                                       np.matrix(np.add(x[idx_obs],
                                                        -mu[idx_cluster])).transpose())[0, 0] -
                       np.log(w[idx_obs, idx_cluster]))
        for idx_tilde in range(x_tilde.shape[0]):
            ll += alpha * \
                  (np.log(phi[int(z_tilde[idx_tilde, 0])]) +
                   (-x_tilde.shape[1] / 2) * np.log(2 * np.pi) -
                   0.5 * np.log(np.linalg.det(sigma[int(z_tilde[idx_tilde, 0])])) -
                   0.5 * np.matmul(np.matmul(np.matrix(np.add(x_tilde[idx_tilde],
                                                              -mu[int(z_tilde[idx_tilde, 0])])),
                                             np.linalg.inv(sigma[int(z_tilde[idx_tilde, 0])])),
                                   np.matrix(np.add(x_tilde[idx_tilde],
                                                    -mu[int(z_tilde[idx_tilde, 0])])).transpose())[0, 0]
                   )
        it += 1
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
