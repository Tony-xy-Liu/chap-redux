"""
Variational Bayes for Correlated Topic Model
"""

import copy
import logging
import os
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import psi, gammaln, logsumexp
from sklearn.utils import check_array
from sklearn.utils.validation import check_non_negative
from utility.access_file import save_data

logger = logging.getLogger(__name__)
EPSILON = np.finfo(np.float).eps
UPPER_BOUND = np.log(sys.float_info.max) * 0.05
LOWER_BOUND = np.log(sys.float_info.min) * 10000
np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class CorrelatedTopicModel:
    def __init__(self, vocab, num_components, alpha_mu, alpha_sigma, alpha_beta, optimization_method="L-BFGS-B",
                 cost_threshold=1e-6, component_threshold=0.0001, forgetting_rate=0.9, delay_factor=1.0,
                 subsample_input_size=0.1, batch=10, num_epochs=5, max_inner_iter=10, num_jobs=1, display_interval=2,
                 shuffle=True, random_state=12345, log_path='../../log', verbose=0):
        logging.basicConfig(filename=os.path.join(
            log_path, 'CTM_events'), level=logging.DEBUG)
        self.vocab = vocab
        self.num_features = len(self.vocab)
        self.num_components = num_components
        self.alpha_mu = alpha_mu
        self.alpha_sigma = alpha_sigma
        self.alpha_beta = alpha_beta
        self.optimization_method = optimization_method
        self.cost_threshold = cost_threshold
        self.component_threshold = component_threshold
        self.forgetting_rate = forgetting_rate
        self.delay_factor = delay_factor
        self.display_interval = display_interval
        self.subsample_input_size = subsample_input_size
        self.batch = batch
        self.max_inner_iter = max_inner_iter
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.num_jobs = num_jobs
        self.verbose = verbose
        self.log_path = log_path
        warnings.filterwarnings("ignore", category=Warning)

    def __print_arguments(self, **kwargs):
        argdict = dict()
        argdict.update(
            {'num_features': 'Number of features: {0}'.format(self.num_features)})
        argdict.update({'num_components': 'Number of mixture components.: {0}'.format(
            self.num_components)})
        argdict.update(
            {'alpha_mu': 'Prior of component mean `mu`: {0}'.format(self.alpha_mu)})
        argdict.update(
            {'alpha_sigma': 'Prior of component correlation `sigma`: {0}'.format(self.alpha_sigma)})
        argdict.update(
            {'alpha_beta': 'Prior of component feature distribution `omega`: {0}'.format(self.alpha_beta)})
        argdict.update({'optimization_method': 'Optimization algorithm used? {0}'.format(
            self.optimization_method)})
        argdict.update(
            {'cost_threshold': 'Perplexity tolerance: {0}'.format(self.cost_threshold)})
        argdict.update({'forgetting_rate': 'Forgetting rate to control how quickly old '
                                           'information is forgotten: {0}'.format(self.forgetting_rate)})
        argdict.update(
            {'delay_factor': 'Delay factor down weights early iterations: {0}'.format(self.delay_factor)})
        argdict.update({'subsample_input_size': 'Subsampling inputs: {0}'.format(
            self.subsample_input_size)})
        argdict.update(
            {'batch': 'Number of examples to use in each iteration: {0}'.format(self.batch)})
        argdict.update({'max_inner_iter': 'Number of inner loops inside an optimizer: {0}'.format(
            self.max_inner_iter)})
        argdict.update(
            {'num_epochs': 'Number of loops over training set: {0}'.format(self.num_epochs)})
        argdict.update(
            {'num_jobs': 'Number of parallel workers: {0}'.format(self.num_jobs)})
        argdict.update(
            {'display_interval': 'How often to evaluate? {0}'.format(self.display_interval)})
        argdict.update(
            {'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update(
            {'random_state': 'The random number generator: {0}'.format(self.random_state)})
        argdict.update(
            {'log_path': 'Logs are stored in: {0}'.format(self.log_path)})

        for key, value in kwargs.items():
            argdict.update({key: value})
        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1]
                for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print('\t>> The following arguments are applied:\n\t\t{0}'.format(
            args), file=sys.stderr)
        logger.info(
            '\t>> The following arguments are applied:\n\t\t{0}'.format(args))

    def __shffule(self, num_samples):
        if self.shuffle:
            idx = np.arange(num_samples)
            np.random.shuffle(idx)
            return idx

    def __check_non_neg_array(self, X, whom):
        """check X format

        check X format and make sure no negative value in X.

        Parameters
        ----------
        X :  array-like or sparse matrix

        """
        X = check_array(X, accept_sparse='csr')
        check_non_negative(X, whom)
        return X

    def __check_bounds(self, X):
        if len(X.shape) == 0:
            np.nan_to_num(X, copy=False)
            X = EPSILON if 0 <= X <= 1e-9 else X
            if np.log(X) > UPPER_BOUND:
                X = np.exp(UPPER_BOUND)
            if X < LOWER_BOUND:
                X = LOWER_BOUND
        else:
            shape = X.shape
            temp = X
            if len(shape) == 1:
                temp = X[:, np.newaxis]
            np.nan_to_num(temp, copy=False)
            row, col = np.where(np.logical_and(0.0 <= temp, temp <= EPSILON))
            if len(row) != 0:
                temp[row, col] = EPSILON
            temp[np.log(temp) > UPPER_BOUND] = np.exp(UPPER_BOUND)
            temp[temp < LOWER_BOUND] = LOWER_BOUND
            X = temp.reshape((shape))
        return X

    def __init_latent_variables(self):
        """Initialize latent variables."""

        alpha_beta = self.alpha_beta
        if alpha_beta <= 0:
            self.alpha_beta = 1.0 / self.num_components

        alpha_mu = self.alpha_mu
        if alpha_mu <= 0:
            self.alpha_mu = 1.0 / self.num_components
        alpha_sigma = self.alpha_sigma
        if alpha_sigma <= 0:
            self.alpha_sigma = 1.0 / self.num_components

        # initialize a model with zero-mean, diagonal covariance gaussian and
        # random topics seeded from the corpus
        self.mu = np.zeros(self.num_components) + self.alpha_mu
        self.sigma = np.eye(self.num_components) + self.alpha_sigma
        self.sigma_inv = np.linalg.pinv(self.sigma)

        # initialize a t-dimensional vector.
        self.alpha_beta_vec = np.zeros(self.num_features) + self.alpha_beta

        # initialize variational parameters
        init_gamma = 100.
        init_var = 1. / init_gamma
        self.omega = np.random.gamma(shape=init_gamma, scale=init_var, size=(
            self.num_components, self.num_features))
        self.omega = self.omega / self.omega.sum(axis=1)[:, np.newaxis]
        self.expected_log_beta = self.__dirichlet_expectation(alpha=self.omega)
        self.lam = np.zeros(self.num_components) + self.alpha_mu
        self.nu_square = np.ones(self.num_components) + self.alpha_sigma
        self.zeta = np.sum(np.exp(self.lam + 0.5 * self.nu_square))

    def __set_variables(self, batch_size, num_jobs):
        self.batch = batch_size
        self.num_jobs = num_jobs
        if batch_size < 0:
            self.batch = 30
        if num_jobs < 0:
            self.num_jobs = 1

    def __dirichlet_expectation(self, alpha, beta=None):
        if beta is not None:
            return psi(beta) - psi(np.sum(alpha))
        if len(alpha.shape) == 1:
            return psi(alpha) - psi(np.sum(alpha))
        return (psi(alpha) - psi(np.sum(alpha, axis=1))[:, np.newaxis])

    def __optimize_lambda(self, lam, arguments):
        def __func_lambda(lam, *args):
            (zeta, sum_phi, num_features) = args
            tmp = np.exp(lam + 0.5 * self.nu_square) / zeta
            func_lambda = np.dot(lam, sum_phi)
            mean_adjustment = lam - self.mu
            func_lambda += -0.5 * \
                           np.dot(np.dot(mean_adjustment, self.sigma_inv), mean_adjustment.T)
            func_lambda -= num_features * np.sum(tmp)
            return np.asscalar(-func_lambda)

        def __func_jac_lambda(lam, *args):
            (zeta, sum_phi, num_features) = args
            tmp = np.exp(lam + 0.5 * self.nu_square) / zeta
            jac_lambda = -np.dot((lam - self.mu), self.sigma_inv)
            jac_lambda += sum_phi
            jac_lambda -= num_features * tmp
            return np.asarray(-jac_lambda)

        def __func_hess_lambda(lam, *args):
            (zeta, sum_phi, num_features) = args
            tmp = np.exp(lam + 0.5 * self.nu_square) / zeta
            hess_lambda = -self.sigma_inv
            hess_lambda -= num_features * np.diag(tmp)
            return np.asarray(-hess_lambda)

        optimize_result = minimize(__func_lambda, lam, args=arguments,
                                   method=self.optimization_method,
                                   jac=__func_jac_lambda,
                                   hess=__func_hess_lambda,
                                   options={'maxiter': self.max_inner_iter,
                                            'disp': False})

        optimize_lambda = optimize_result.x
        self.lam = optimize_lambda
        return optimize_lambda

    def __optimize_nu_square(self, nu_square, arguments):
        def __func_nu_square(nu_square, *args):
            (zeta, num_features) = args
            tmp = np.exp(self.lam + 0.5 * nu_square) / zeta
            tmp = self.__check_bounds(X=tmp)
            func_nu_square = 0.5 * np.sum(np.log(nu_square))
            func_nu_square += -0.5 * \
                              np.trace(np.diag(nu_square) * self.sigma_inv)
            func_nu_square += -num_features * \
                              self.__check_bounds(X=np.sum(tmp))
            return np.asscalar(-func_nu_square)

        def __func_jac_nu_square(nu_square, *args):
            (zeta, num_features) = args
            tmp = np.exp(self.lam + 0.5 * nu_square) / zeta
            tmp = self.__check_bounds(X=tmp)
            jac_nu_square = np.array(np.diag(self.sigma_inv))
            jac_nu_square += num_features * tmp
            jac_nu_square -= 1 / nu_square
            jac_nu_square = -0.5 * jac_nu_square
            return np.asarray(-jac_nu_square)

        def __func_hess_nu_square(nu_square, *args):
            (zeta, num_features) = args
            tmp = np.exp(self.lam + 0.5 * nu_square) / zeta
            tmp = self.__check_bounds(X=tmp)
            hess_nu_square = 1 / (nu_square ** 2)
            hess_nu_square += 0.5 * num_features * tmp
            hess_nu_square = np.diag(hess_nu_square)
            hess_nu_square = -0.5 * hess_nu_square
            return np.asarray(-hess_nu_square)

        bounds = tuple([(0, None)] * self.num_components)
        nu_square = self.__check_bounds(X=nu_square)
        optimization_method = "L-BFGS-B"
        if optimization_method == "L-BFGS-B":
            optimize_result = minimize(__func_nu_square, nu_square,
                                       args=arguments,
                                       method=optimization_method,
                                       jac=__func_jac_nu_square,
                                       bounds=bounds,
                                       options={'maxiter': self.max_inner_iter,
                                                'disp': False})
        else:
            optimize_result = minimize(__func_nu_square, nu_square,
                                       args=arguments,
                                       method="L-BFGS-B",
                                       jac=__func_jac_nu_square,
                                       hess=__func_hess_nu_square,
                                       bounds=bounds,
                                       options={'maxiter': self.max_inner_iter,
                                                'disp': False})
        optimize_nu_square = optimize_result.x
        self.nu_square = optimize_nu_square
        return optimize_nu_square

    def __optimize_log_nu_square(self, nu_square, arguments):
        def __func_log_nu_square(log_nu_square, *args):
            nu_square = np.exp(log_nu_square)
            (zeta, num_features) = args
            tmp = np.exp(self.lam + 0.5 * nu_square) / zeta
            tmp = self.__check_bounds(X=tmp)
            func_nu_square = 0.5 * np.sum(log_nu_square)
            func_nu_square += -0.5 * \
                              np.trace(np.diag(nu_square) * self.sigma_inv)
            func_nu_square += -num_features * \
                              self.__check_bounds(X=np.sum(tmp))
            return np.asscalar(-func_nu_square)

        def __func_jac_log_nu_square(log_nu_square, *args):
            nu_square = np.exp(log_nu_square)
            (zeta, num_features) = args
            tmp = np.exp(self.lam + 0.5 * nu_square) / zeta
            tmp = self.__check_bounds(X=tmp)
            jac_nu_square = np.copy(np.diag(self.sigma_inv))
            jac_nu_square += num_features * tmp - 1
            jac_nu_square = -0.5 * nu_square * jac_nu_square
            return np.asarray(-jac_nu_square)

        def __func_hess_log_nu_square(log_nu_square, *args):
            nu_square = np.exp(log_nu_square)
            (zeta, num_features) = args
            tmp = np.exp(self.lam + 0.5 * nu_square) / zeta
            tmp = self.__check_bounds(X=tmp)
            hess_log_nu_square = -0.5 * nu_square * np.diag(self.sigma_inv)
            hess_log_nu_square -= 0.5 * num_features * \
                                  tmp * nu_square * (1 + 0.5 * nu_square)
            hess_log_nu_square = np.diag(hess_log_nu_square)
            return np.asarray(-hess_log_nu_square)

        nu_square = self.__check_bounds(X=nu_square)
        log_nu_square = np.log(nu_square + EPSILON)
        optimize_result = minimize(__func_log_nu_square, log_nu_square,
                                   args=arguments, method=self.optimization_method,
                                   jac=__func_jac_log_nu_square,
                                   hess=__func_hess_log_nu_square,
                                   options={'maxiter': self.max_inner_iter,
                                            'disp': False})
        optimize_nu_square = np.exp(optimize_result.x)
        self.nu_square = optimize_nu_square
        return optimize_nu_square

    def __e_step(self, X, current_batch=-1, total_batches=-1, transform=False, verbose=True):
        """E-step in EM update.
        :param verbose:
        """
        if current_batch != -1:
            if verbose:
                desc = '\t       --> Computing E-step: {0:.2f}%...'.format(
                    ((current_batch + 1) / total_batches) * 100)
            else:
                desc = '\t       --> Re-Computing E-step with updated parameters: {0:.2f}%...'.format(
                    ((current_batch + 1) / total_batches) * 100)
            if (current_batch + 1) != total_batches:
                print(desc, end="\r")
            if (current_batch + 1) == total_batches:
                print(desc)
                logger.info(desc)

        num_samples = X.shape[0]

        # initialize two n-by-b matrices
        component_distribution = np.ones((num_samples, self.num_components))

        # initialize empty sufficient statistics for the M-step.
        mu_sstats = np.zeros(self.num_components)
        sigma_sstats = np.zeros((self.num_components, self.num_components))
        phi_sstats = np.zeros(
            (self.num_components, self.num_features)) + self.alpha_beta_vec
        zeta = self.zeta

        # iterate over all samples
        for idx in np.arange(num_samples):
            feature_idx = X[idx].indices
            feature_count = X[idx].data

            # compute the total number of vocab
            sample_feature_count = np.sum(X[idx])

            # initialize values
            sample_lambda = np.zeros(self.num_components)
            sample_nu_square = np.zeros(self.num_components)
            if transform:
                sample_lambda = self.mu
                sample_nu_square = self.sigma

            prev_comp_distr = component_distribution[idx]

            for iter in np.arange(self.max_inner_iter):

                # update log_phi in a closed form
                log_phi = sample_lambda[:, np.newaxis] + \
                          self.expected_log_beta[:, feature_idx] + EPSILON
                log_phi = log_phi - logsumexp(log_phi, axis=0)

                # sum phi over components
                sum_phi = np.exp(
                    logsumexp(log_phi + np.log(feature_count), axis=1))

                if not transform:
                    # update sample_lambda
                    arguments = (zeta, sum_phi, sample_feature_count)
                    sample_lambda = self.__optimize_lambda(
                        lam=self.lam, arguments=arguments)

                    # update zeta
                    zeta = np.sum(
                        np.exp(sample_lambda + 0.5 * sample_nu_square))

                    # update sample_nu_square
                    arguments = (zeta, sample_feature_count)
                    sample_nu_square = self.__optimize_nu_square(
                        nu_square=self.nu_square, arguments=arguments)

                    # update zeta
                    zeta = np.sum(
                        np.exp(sample_lambda + 0.5 * sample_nu_square))

                    # if curr_cmp_distr hasn't changed much, we're done.
                    curr_cmp_distr = (sum_phi + EPSILON) / \
                                     (np.sum(sum_phi) + EPSILON)
                    meanchange = np.abs(curr_cmp_distr, prev_comp_distr)
                    if np.mean(meanchange) < self.component_threshold:
                        break

            # contribution of an example to the expected sufficient
            # statistics for the M step.
            mu_sstats += sample_lambda
            temp = np.dot(sample_lambda[:, np.newaxis],
                          sample_lambda[:, np.newaxis].T)
            if not transform:
                sigma_sstats += np.diag(sample_nu_square) + temp
            else:
                sigma_sstats += sample_nu_square + temp

            temp = np.exp(log_phi + np.log(feature_count))
            temp = temp / temp.sum(axis=0)
            temp = self.__check_bounds(X=temp)
            phi_sstats[:, feature_idx] += temp

            # compute the normalized components weights across samples
            component_distribution[idx, :] = (
                                                     sum_phi + EPSILON) / (np.sum(sum_phi) + EPSILON)

        # store sufficient_stats in a dictionary
        sufficient_stats = {"mu_sstats": mu_sstats,
                            "sigma_sstats": sigma_sstats, "phi_sstats": phi_sstats}
        if not transform:
            component_distribution = None

        return sufficient_stats, component_distribution

    def __batch_e_step(self, X, list_batches, transform=False, verbose=True):
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads",
                            verbose=max(0, self.verbose - 1))
        results = parallel(delayed(self.__e_step)(X[batch:batch + self.batch],
                                                  idx, len(list_batches),
                                                  transform, verbose)
                           for idx, batch in enumerate(list_batches))

        # merge result
        sstats_list, component_distribution = zip(*results)
        del results

        mu_sstats = 0
        sigma_sstats = 0
        phi_sstats = 0

        for lst in sstats_list:
            mu_sstats += lst["mu_sstats"]
            sigma_sstats += lst["sigma_sstats"]
            phi_sstats += lst["phi_sstats"]

        if transform:
            component_distribution = np.vstack(component_distribution)
            component_distribution /= component_distribution.sum(1)[
                                      :, np.newaxis]

        sufficient_stats = {"mu_sstats": mu_sstats,
                            "sigma_sstats": sigma_sstats, "phi_sstats": phi_sstats}

        return sufficient_stats, component_distribution

    def __m_step(self, sstats, learning_rate, num_samples):
        """
        Optimize model's parameters using the statictics collected during the e-step
        """
        desc = '\t       --> Computing M-step...'
        print(desc)

        # compute mean values
        mean_mu_sstats = sstats["mu_sstats"] / num_samples
        mean_sigma_sstats = sstats["sigma_sstats"] / num_samples

        # update component feature distributions
        phi = sstats["phi_sstats"] + self.alpha_beta
        self.omega = (1 - learning_rate) * self.omega + learning_rate * phi
        self.omega = self.omega / np.sum(self.omega, axis=1)[:, np.newaxis]
        self.expected_log_beta = self.__dirichlet_expectation(alpha=self.omega)

        # update mu and sigma
        self.mu = (1 - learning_rate) * self.mu + \
                  mean_mu_sstats * learning_rate
        self.mu = self.__check_bounds(X=self.mu)
        temp = mean_sigma_sstats + \
               np.dot(self.mu[:, np.newaxis], self.mu[:, np.newaxis].T)
        self.sigma = (1 - learning_rate) * self.sigma + temp * learning_rate
        self.sigma = self.__check_bounds(X=self.sigma)
        self.sigma_inv = np.linalg.pinv(self.sigma)

        # update lam and nu square
        self.lam = (1 - learning_rate) * self.lam + self.mu * learning_rate
        self.nu_square = (1 - learning_rate) * self.nu_square + \
                         np.diag(self.sigma) * learning_rate

        # update zeta
        sum_zeta = np.sum(np.exp(self.lam + 0.5 * self.nu_square))
        self.zeta = (1 - learning_rate) * self.zeta + sum_zeta * learning_rate

    def __elbo(self, num_samples, num_features, sstats):
        score = 0.0

        # phi statistics
        phi = sstats + EPSILON

        # E[log p(beta | alpha)] - E[log q(beta | omega)]
        temp = self.alpha_beta_vec - 1
        score += (gammaln(np.sum(self.alpha_beta_vec)) -
                  np.sum(gammaln(self.alpha_beta_vec))) * self.num_components
        score += np.sum(np.multiply(temp[np.newaxis,
                                    :], self.expected_log_beta))
        score -= np.sum(gammaln(np.sum(self.omega, axis=1))) + \
                 np.sum(gammaln(self.omega))
        score -= np.sum(np.multiply((self.omega - 1), self.expected_log_beta))

        # E[log p(eta | mu, sigma)] + E[log q(eta | lam, nu_square)]
        det = np.linalg.slogdet(self.sigma_inv + EPSILON)
        score += 0.5 * det[0] * det[1]
        score -= 0.5 * self.num_components * np.log(2 * np.pi)
        score -= 0.5 * np.trace(np.diag(self.nu_square) * self.sigma_inv)
        temp = np.dot((self.mu - self.lam), self.sigma_inv)
        score -= 0.5 * np.dot(temp, (self.mu - self.lam).T) * num_samples
        temp = 0.5 * (np.log(self.nu_square + EPSILON) + np.log(2 * np.pi) + 1)
        score += (np.sum(temp) * num_samples)

        # E[log p(z | eta)] - E[log q(z | phi)]
        score += ((1 - np.log(self.zeta + EPSILON)) * num_features)
        score += np.sum(np.dot(self.lam, phi))
        score -= (np.sum(np.exp(self.lam + 0.5 * self.nu_square)) /
                  self.zeta) * num_features
        score -= np.sum(np.multiply(phi, np.log(phi + EPSILON)))

        # E[log p(w | z, beta)]
        score += np.sum(np.multiply(phi, self.expected_log_beta))

        return float(score)

    def fit(self, X, model_name='CTM', model_path="../../model", result_path=".", display_params: bool = True):
        if X is None:
            raise Exception("Please provide a dataset.")
        assert X.shape[1] == self.num_features

        X = self.__check_non_neg_array(X, "CorrelatedTopicModel.fit")

        # collect properties from data
        self.__init_latent_variables()
        num_samples = int(X.shape[0] * self.subsample_input_size)
        list_batches = np.arange(start=0, stop=num_samples, step=self.batch)

        if display_params:
            self.__print_arguments()
            time.sleep(2)

        cost_file_name = model_name + "_cost.txt"
        save_data('', file_name=cost_file_name, save_path=result_path,
                  mode='w', w_string=True, print_tag=False)

        print('\t>> Training by CTM model...')
        logger.info('\t>> Training by CTM model...')
        n_epochs = self.num_epochs + 1
        old_bound = np.inf

        timeref = time.time()

        for epoch in np.arange(start=1, stop=n_epochs):
            desc = '\t   {0:d})- Epoch count ({0:d}/{1:d})...'.format(
                epoch, n_epochs - 1)
            print(desc)
            logger.info(desc)

            learning_rate = np.power(
                (epoch + self.delay_factor), -self.forgetting_rate)

            # Subsample dataset
            idx = np.random.choice(X.shape[0], num_samples, False)
            start_epoch = time.time()

            # E-step
            sstats, tmp = self.__batch_e_step(
                X=X[idx, :], list_batches=list_batches)
            del tmp

            # M-step
            self.__m_step(sstats=sstats, learning_rate=learning_rate,
                          num_samples=num_samples)

            end_epoch = time.time()

            self.is_fit = True

            # Compute predictive perplexity
            new_bound = self.perplexity(
                X=X[idx, :], sstats=sstats["phi_sstats"])

            print('\t\t## Epoch {0} took {1} seconds...'.format(
                epoch, round(end_epoch - start_epoch, 3)))
            logger.info('\t\t## Epoch {0} took {1} seconds...'.format(
                epoch, round(end_epoch - start_epoch, 3)))
            data = str(epoch) + '\t' + str(round(end_epoch -
                                                 start_epoch, 3)) + '\t' + str(new_bound) + '\n'
            save_data(data=data, file_name=cost_file_name, save_path=result_path, mode='a', w_string=True,
                      print_tag=False)
            # Save models parameters based on test frequencies
            if (epoch % self.display_interval) == 0 or epoch == 1 or epoch == n_epochs - 1:
                print(
                    '\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_bound, old_bound))
                logger.info(
                    '\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_bound, old_bound))

                if new_bound <= old_bound or epoch == n_epochs - 1:
                    omega_file_name = model_name + '_exp_omega.npz'
                    sigma_file_name = model_name + '_sigma.npz'
                    mu_file_name = model_name + '_mu.npz'
                    model_file_name = model_name + '.pkl'
                    if epoch == n_epochs - 1:
                        omega_file_name = model_name + '_exp_omega_final.npz'
                        sigma_file_name = model_name + '_sigma_final.npz'
                        mu_file_name = model_name + '_mu_final.npz'
                        model_file_name = model_name + '_final.pkl'

                    print(
                        '\t\t  --> Storing the CTM omega to: {0:s}'.format(omega_file_name))
                    logger.info(
                        '\t\t  --> Storing the CTM omega to: {0:s}'.format(omega_file_name))
                    np.savez(os.path.join(
                        model_path, omega_file_name), self.omega)

                    print(
                        '\t\t  --> Storing the CTM sigma to: {0:s}'.format(sigma_file_name))
                    logger.info(
                        '\t\t  --> Storing the CTM sigma to: {0:s}'.format(sigma_file_name))
                    np.savez(os.path.join(
                        model_path, sigma_file_name), self.sigma)

                    print(
                        '\t\t  --> Storing the CTM mu to: {0:s}'.format(mu_file_name))
                    logger.info(
                        '\t\t  --> Storing the CTM mu to: {0:s}'.format(mu_file_name))
                    np.savez(os.path.join(model_path, mu_file_name), self.mu)

                    print(
                        '\t\t  --> Storing the CTM model to: {0:s}'.format(model_file_name))
                    logger.info(
                        '\t\t  --> Storing the CTM model to: {0:s}'.format(model_file_name))
                    save_data(data=copy.copy(self), file_name=model_file_name, save_path=model_path, mode="wb",
                              print_tag=False)
                    old_bound = new_bound
        print('\t  --> Training consumed %.2f mintues' %
              (round((time.time() - timeref) / 60., 3)))
        logger.info('\t  --> Training consumed %.2f mintues' %
                    (round((time.time() - timeref) / 60., 3)))

    def __transform(self, X):
        num_samples = X.shape[0]
        X = self.__check_non_neg_array(X, "CorrelatedTopicModel.fit")
        list_batches = np.arange(start=0, stop=num_samples, step=self.batch)
        sstats, component_distribution = self.__batch_e_step(X=X, list_batches=list_batches,
                                                             transform=True)
        return sstats, component_distribution

    def transform(self, X, batch_size=30, num_jobs=1):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        self.__set_variables(batch_size=batch_size, num_jobs=num_jobs)
        _, component_distribution = self.__transform(X=X)
        return component_distribution

    def perplexity(self, X, log_space: bool = True, sstats=None, per_feature=True,
                   per_component=False, batch_size=30, num_jobs=1):
        self.__set_variables(batch_size=batch_size, num_jobs=num_jobs)
        # collect properties from data
        num_samples = X.shape[0]
        num_features = np.sum(X)
        num_components = self.num_components

        if sstats is None:
            sstats, _ = self.__transform(X=X)

        perplexity = self.__elbo(
            num_samples=num_samples, num_features=num_features, sstats=sstats)

        if per_feature:
            perplexity = perplexity / num_features
        if per_component:
            perplexity = perplexity / num_components

        perplexity = -1 * perplexity
        if log_space:
            perplexity = np.log(perplexity)
        else:
            perplexity = np.exp(perplexity)

        return perplexity

    def score(self, X, log_space: bool = True, batch_size=30, num_jobs=1):
        """Calculate approximate log-likelihood as score."""
        self.__set_variables(batch_size=batch_size, num_jobs=num_jobs)

        # collect properties from data
        num_samples = X.shape[0]
        num_features = np.sum(X)

        sstats, _ = self.__transform(X=X)
        score = self.__elbo(num_samples=num_samples,
                            num_features=num_features, sstats=sstats["phi_sstats"])
        if log_space:
            score = -1 * perplexity
            score = np.log(score)
        return score

    def get_components(self):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        return self.omega

    def get_component_features(self, component_idx: int, top_k=10):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        component = self.get_components()[component_idx]
        # normalize to probability distribution
        component = component / component.sum()
        best_k = np.argsort(component)[::-1][:top_k]
        return [(self.vocab[idx], component[idx]) for idx in best_k]

    def get_feature_components(self, feature_idx: int, minimum_probability: float = 0.00001):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        # never allow zero values in sparse output
        minimum_probability = max(minimum_probability, 1e-8)
        component_prob = [(idx, self.omega[idx][feature_idx]) for idx in np.arange(0, self.num_components)
                          if self.omega[idx][feature_idx] >= minimum_probability]
        return component_prob

    def get_example_components(self, X, minimum_probability: float = 0.00001, batch_size=30, num_jobs=1):
        assert X.shape[0] == 1
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        self.__set_variables(batch_size=batch_size, num_jobs=num_jobs)
        # never allow zero values in sparse output
        minimum_probability = max(minimum_probability, 1e-8)
        sstats, component_distribution = self.__transform(X=X)
        component_feature = sstats["phi_sstats"]
        component_feature = csr_matrix(X).multiply(component_feature).toarray()
        component_feature[component_feature < minimum_probability] = 0.
        component_feature_id = []
        feature_component_id = []
        for a in np.arange(self.num_components):
            features = np.argsort(component_feature[a])[::-1]
            temp = np.trim_zeros(component_feature[a, features])
            features = features[:len(temp)]
            component_feature_id.append((a, [self.vocab[f] for f in features]))
        for t in np.arange(self.num_features):
            components = np.argsort(component_feature[:, t])[::-1]
            feature_component_id.append(
                (t, components, component_feature[components, t]))
        return component_distribution, component_feature_id, feature_component_id

    def get_component_distribution(self, X, minimum_probability: float = 0.00001,
                                   batch_size=30, num_jobs=1):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        self.__set_variables(batch_size=batch_size, num_jobs=num_jobs)
        # never allow zero values in sparse output
        minimum_probability = max(minimum_probability, 1e-8)
        _, component_distribution = self.__transform(X=X)
        component_distribution[component_distribution <
                               minimum_probability] = 0.
        return component_distribution

    def predictive_distribution(self, X, cal_average=True, batch_size=30, num_jobs=1):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")

        X = self.__check_non_neg_array(X, "CorrelatedTopicModel.fit")
        self.__set_variables(batch_size=batch_size, num_jobs=num_jobs)
        num_samples = X.shape[0]
        _, component_distribution = self.__transform(X=X)
        score = 0
        # iterate over all samples
        for idx in np.arange(num_samples):
            feature_idx = X[idx].indices
            temp = np.multiply(
                component_distribution[idx][:, np.newaxis], self.omega[:, feature_idx])
            score += np.sum(temp)
        if cal_average:
            score = score / num_samples
        return np.log(score + EPSILON)
