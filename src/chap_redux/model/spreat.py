"""
Variational Bayes for Distributed Sparse Correlated Pathway Group Model
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
from scipy.special import psi, gammaln, logsumexp, beta
from sklearn.utils import check_array
from sklearn.utils.validation import check_non_negative
from ..utility.access_file import save_data

logger = logging.getLogger(__name__)
EPSILON = np.finfo(np.float).eps
UPPER_BOUND = np.log(sys.float_info.max) * 0.05
LOWER_BOUND = np.log(sys.float_info.min) * 10000
np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class diSparseCorrelatedBagPathway:
    def __init__(self, vocab, num_components, alpha_mu, alpha_sigma, alpha_phi, gamma=2, kappa=3, xi=0, varpi=1,
                 optimization_method="Newton-CG", cost_threshold=0.001, component_threshold=0.0001, forgetting_rate=0.9,
                 delay_factor=1.0, max_sampling=5, max_inner_iter=10, top_k=100, collapse2ctm=False, use_features=False,
                 subsample_input_size=0.1, batch=10, num_epochs=5, num_jobs=1, display_interval=2, shuffle=True,
                 random_state=12345, log_path='../../log', verbose=0):
        """Distributed Sparse Correlated Bag Pathway model"""
        logging.basicConfig(filename=os.path.join(log_path, 'SPREAT_events'), level=logging.DEBUG)
        self.vocab = vocab
        self.num_features = len(self.vocab)
        self.num_components = num_components
        self.alpha_mu = alpha_mu
        self.alpha_sigma = alpha_sigma
        self.alpha_phi = alpha_phi
        self.gamma = gamma
        self.kappa = kappa
        self.xi = xi
        self.varpi = varpi
        self.optimization_method = optimization_method
        self.cost_threshold = cost_threshold
        self.component_threshold = component_threshold
        self.forgetting_rate = forgetting_rate
        self.delay_factor = delay_factor
        self.max_sampling = max_sampling
        self.top_k = top_k
        if top_k < 0 or top_k > self.num_features:
            self.top_k = self.num_features
        self.collapse2ctm = collapse2ctm
        self.use_features = use_features
        self.display_interval = display_interval
        self.shuffle = shuffle
        self.subsample_input_size = subsample_input_size
        self.batch = batch
        self.max_inner_iter = max_inner_iter
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.num_jobs = num_jobs
        self.verbose = verbose
        self.log_path = log_path
        warnings.filterwarnings("ignore", category=Warning)

    def __print_arguments(self, **kwargs):
        argdict = dict()
        argdict.update({'num_features': 'Number of features: {0}'.format(self.num_features)})
        argdict.update({'num_components': 'Number of mixture components: {0}'.format(self.num_components)})
        argdict.update({'top_k': 'Number of features per component for sparseness: {0}'.format(self.top_k)})
        argdict.update({'alpha_mu': 'Prior of component mean `mu`: {0}'.format(self.alpha_mu)})
        argdict.update({'alpha_sigma': 'Prior of component correlation `sigma`: {0}'.format(self.alpha_sigma)})
        argdict.update({'alpha_phi': 'Prior of component feature distribution `phi`: {0}'.format(self.alpha_phi)})
        argdict.update({'gamma': 'A hyper-parameter for omega parameter (gamma): {0}'.format(self.gamma)})
        argdict.update({'kappa': 'Additional hyper-parameter for omega parameter (kappa): {0}'.format(self.kappa)})
        argdict.update({'xi': 'Prior of supplementary features: {0}'.format(self.xi)})
        argdict.update({'varpi': 'The parameter controlling component feature distribution: {0}'.format(self.varpi)})
        argdict.update({'optimization_method': 'Optimization algorithm used? {0}'.format(self.optimization_method)})
        argdict.update({'cost_threshold': 'Perplexity tolerance: {0}'.format(self.cost_threshold)})
        argdict.update({'max_sampling': 'Maximum number of random samplings.: {0}'.format(self.max_sampling)})
        argdict.update({'collapse2ctm': 'Collapse estimation to CTM model? {0}'.format(self.collapse2ctm)})
        argdict.update({'use_features': 'Employ external component features? {0}'.format(self.use_features)})
        argdict.update({'forgetting_rate': 'Forgetting rate to control how quickly old '
                                           'information is forgotten: {0}'.format(self.forgetting_rate)})
        argdict.update({'delay_factor': 'Delay factor down weights early iterations: {0}'.format(self.delay_factor)})
        argdict.update({'subsample_input_size': 'Subsampling inputs: {0}'.format(self.subsample_input_size)})
        argdict.update({'batch': 'Number of examples to use in each iteration: {0}'.format(self.batch)})
        argdict.update({'max_inner_iter': 'Number of inner loops inside an optimizer: {0}'.format(self.max_inner_iter)})
        argdict.update({'num_epochs': 'Number of loops over training set: {0}'.format(self.num_epochs)})
        argdict.update({'num_jobs': 'Number of parallel workers: {0}'.format(self.num_jobs)})
        argdict.update({'display_interval': 'How often to evaluate? {0}'.format(self.display_interval)})
        argdict.update({'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update({'random_state': 'The random number generator: {0}'.format(self.random_state)})
        argdict.update({'log_path': 'Logs are stored in: {0}'.format(self.log_path)})

        for key, value in kwargs.items():
            argdict.update({key: value})
        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print('\t>> The following arguments are applied:\n\t\t{0}'.format(args), file=sys.stderr)
        logger.info('\t>> The following arguments are applied:\n\t\t{0}'.format(args))

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

        alpha_phi = self.alpha_phi
        if alpha_phi <= 0:
            self.alpha_phi = 1.0 / self.num_components

        alpha_mu = self.alpha_mu
        if alpha_mu <= 0:
            self.alpha_mu = 1.0 / self.num_components
        alpha_sigma = self.alpha_sigma
        if alpha_sigma <= 0:
            self.alpha_sigma = 1.0 / self.num_components

        xi = self.xi
        if xi <= 0:
            self.xi = 1.0 / self.num_features

        # initialize a model with zero-mean, diagonal covariance gaussian and
        # random topics seeded from the corpus
        self.mu = np.zeros(self.num_components) + self.alpha_mu
        self.sigma = np.eye(self.num_components) + self.alpha_sigma
        self.sigma_inv = np.linalg.pinv(self.sigma)

        # initialize a t-dimensional vector.
        self.alpha_phi_vec = np.zeros(self.num_features) + self.alpha_phi
        self.xi_vec = np.zeros(self.num_features) + self.xi

        # initialize variational parameters
        init_gamma = 100.
        init_var = 1. / init_gamma
        self.phi = np.random.gamma(shape=init_gamma, scale=init_var, size=(self.num_components, self.num_features))
        self.phi = self.phi / self.phi.sum(axis=1)[:, np.newaxis]
        self.expected_log_phi = self.__dirichlet_expectation(alpha=self.phi)
        self.nu = np.zeros(self.num_components) + self.alpha_mu
        self.zeta_square = np.ones(self.num_components) + self.alpha_sigma
        self.rho = np.sum(np.exp(self.nu + 0.5 * self.zeta_square))
        self.vartheta_1 = self.gamma
        self.vartheta_2 = self.kappa

    def __set_variables(self, X, M, features, batch_size, num_jobs):
        if not self.collapse2ctm:
            if M is not None:
                assert M.shape == X.shape
            else:
                M = np.zeros((X.shape[0], self.num_features))
            if features is not None:
                assert X.shape[1] == features.shape[0]
            else:
                features = np.ones((self.num_features, 20))
            features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
        self.batch = batch_size
        self.num_jobs = num_jobs
        if batch_size < 0:
            self.batch = 30
        if num_jobs < 0:
            self.num_jobs = 1
        return M, features

    def __dirichlet_expectation(self, alpha, beta=None):
        if beta is not None:
            return psi(beta) - psi(np.sum(alpha))
        if len(alpha.shape) == 1:
            return psi(alpha) - psi(np.sum(alpha))
        return psi(alpha) - psi(np.sum(alpha, axis=1))[:, np.newaxis]

    def __optimize_nu(self, nu, arguments):
        def __func_nu(nu, *args):
            (rho, sum_o, num_features) = args
            tmp = np.exp(nu + 0.5 * self.zeta_square) / rho
            func_nu = np.dot(nu, sum_o)
            mean_adjustment = nu - self.mu
            func_nu += -0.5 * np.dot(np.dot(mean_adjustment, self.sigma_inv), mean_adjustment.T)
            func_nu += -num_features * np.sum(tmp)
            return np.asscalar(-func_nu)

        def __func_jac_nu(nu, *args):
            (rho, sum_o, num_features) = args
            tmp = np.exp(nu + 0.5 * self.zeta_square) / rho
            jac_nu = -np.dot((nu - self.mu), self.sigma_inv)
            jac_nu += sum_o
            jac_nu -= num_features * tmp
            return np.asarray(-jac_nu)

        def __func_hess_nu(nu, *args):
            (rho, sum_o, num_features) = args
            tmp = np.exp(nu + 0.5 * self.zeta_square) / rho
            hess_nu = -self.sigma_inv
            hess_nu -= num_features * np.diag(tmp)
            return np.asarray(-hess_nu)

        optimize_result = minimize(__func_nu, nu, args=arguments,
                                   method=self.optimization_method,
                                   jac=__func_jac_nu,
                                   hess=__func_hess_nu,
                                   options={'maxiter': self.max_inner_iter,
                                            'disp': False})
        optimize_nu = optimize_result.x
        self.nu = optimize_nu
        return optimize_nu

    def __optimize_zeta_square(self, zeta_square, arguments):
        def __func_zeta_square(zeta_square, *args):
            (rho, num_features) = args
            tmp = np.exp(self.nu + 0.5 * zeta_square) / rho
            tmp = self.__check_bounds(X=tmp)
            func_zeta_square = 0.5 * np.sum(np.log(zeta_square))
            func_zeta_square += -0.5 * np.trace(np.diag(zeta_square) * self.sigma_inv)
            func_zeta_square += -num_features * self.__check_bounds(X=np.sum(tmp))
            return np.asscalar(-func_zeta_square)

        def __func_jac_zeta_square(zeta_square, *args):
            (rho, num_features) = args
            tmp = np.exp(self.nu + 0.5 * zeta_square) / rho
            tmp = self.__check_bounds(X=tmp)
            jac_zeta_square = np.array(np.diag(self.sigma_inv))
            jac_zeta_square += num_features * tmp
            jac_zeta_square -= 1 / zeta_square
            jac_zeta_square = -0.5 * jac_zeta_square
            return np.asarray(-jac_zeta_square)

        def __func_hess_zeta_square(zeta_square, *args):
            (rho, num_features) = args
            tmp = np.exp(self.nu + 0.5 * zeta_square) / rho
            tmp = self.__check_bounds(X=tmp)
            hess_zeta_square = 1 / (zeta_square ** 2)
            hess_zeta_square += 0.5 * num_features * tmp
            hess_zeta_square = np.diag(hess_zeta_square)
            hess_zeta_square = -0.5 * hess_zeta_square
            return np.asarray(-hess_zeta_square)

        bounds = tuple([(0, None)] * self.num_components)
        zeta_square = self.__check_bounds(X=zeta_square)

        optimization_method = "L-BFGS-B"
        if optimization_method == "L-BFGS-B":
            optimize_result = minimize(__func_zeta_square, zeta_square,
                                       args=arguments,
                                       method=optimization_method,
                                       jac=__func_jac_zeta_square,
                                       bounds=bounds,
                                       options={'maxiter': self.max_inner_iter,
                                                'disp': False})
        else:
            optimize_result = minimize(__func_zeta_square, zeta_square,
                                       args=arguments,
                                       method=self.optimization_method,
                                       jac=__func_jac_zeta_square,
                                       hess=__func_hess_zeta_square,
                                       bounds=bounds,
                                       options={'maxiter': self.max_inner_iter,
                                                'disp': False})
        optimize_zeta_square = optimize_result.x
        self.zeta_square = optimize_zeta_square
        return optimize_zeta_square

    def __optimize_log_zeta_square(self, zeta_square, arguments):
        def __func_log_zeta_square(log_zeta_square, *args):
            zeta_square = np.exp(log_zeta_square)
            (rho, num_features) = args
            tmp = np.exp(self.nu + 0.5 * zeta_square) / rho
            tmp = self.__check_bounds(X=tmp)
            func_zeta_square = 0.5 * np.sum(log_zeta_square)
            func_zeta_square += -0.5 * np.trace(np.diag(zeta_square) * self.sigma_inv)
            func_zeta_square += -num_features * self.__check_bounds(X=np.sum(tmp))
            return np.asscalar(-func_zeta_square)

        def __func_jac_log_zeta_square(log_zeta_square, *args):
            zeta_square = np.exp(log_zeta_square)
            (rho, num_features) = args
            tmp = np.exp(self.nu + 0.5 * zeta_square) / rho
            tmp = self.__check_bounds(X=tmp)
            jac_zeta_square = np.copy(np.diag(self.sigma_inv))
            jac_zeta_square += num_features * tmp - 1
            jac_zeta_square = -0.5 * zeta_square * jac_zeta_square
            return np.asarray(-jac_zeta_square)

        def __func_hess_log_zeta_square(log_zeta_square, *args):
            zeta_square = np.exp(log_zeta_square)
            (rho, num_features) = args
            tmp = np.exp(self.nu + 0.5 * zeta_square) / rho
            tmp = self.__check_bounds(X=tmp)
            hess_log_zeta_square = -0.5 * zeta_square * np.diag(self.sigma_inv)
            hess_log_zeta_square -= 0.5 * num_features * tmp * zeta_square * (1 + 0.5 * zeta_square)
            hess_log_zeta_square = np.diag(hess_log_zeta_square)
            return np.asarray(-hess_log_zeta_square)

        zeta_square = self.__check_bounds(X=zeta_square)
        log_zeta_square = np.log(zeta_square + EPSILON)
        optimize_result = minimize(__func_log_zeta_square, log_zeta_square,
                                   args=arguments, method=self.optimization_method,
                                   jac=__func_jac_log_zeta_square,
                                   hess=__func_hess_log_zeta_square,
                                   options={'maxiter': self.max_inner_iter,
                                            'disp': False})
        optimize_zeta_square = np.exp(optimize_result.x)
        self.zeta_square = optimize_zeta_square
        return optimize_zeta_square

    def __e_step(self, X, M, features, current_batch=-1, total_batches=-1, transform=False, verbose=True):
        """E-step in EM update.
        """
        if current_batch != -1:
            if verbose:
                desc = '\t       --> Computing E-step: {0:.2f}%...'.format(((current_batch + 1) / total_batches) * 100)
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
        omega = 0
        omega_sstats = 0
        if not self.collapse2ctm:
            omega = np.copy(M) + self.xi_vec
            omega = omega / np.sum(omega, axis=1)[:, np.newaxis]
            omega_sstats = np.zeros(self.num_features)

        # initialize empty sufficient statistics for the M-step.
        mu_sstats = np.zeros(self.num_components)
        sigma_sstats = np.zeros((self.num_components, self.num_components))
        lambda_sstats = np.zeros(self.num_components)
        o_sstats = np.zeros((self.num_components, self.num_features))
        rho = self.rho
        vartheta_1 = self.vartheta_1
        vartheta_2 = self.vartheta_2

        # iterate over all samples
        for idx in np.arange(num_samples):
            feature_idx = X[idx].indices
            feature_count = X[idx].data

            if self.use_features:
                # features of components
                P = np.dot(features[feature_idx, :], features[feature_idx, :].T)
                P = np.mean(P, axis=0)

            # compute the total number of vocab
            sample_feature_count = np.sum(X[idx])

            # initialize nu and zeta_square for this sample
            sample_nu = np.zeros(self.num_components)
            sample_zeta_square = np.zeros(self.num_components)
            if transform:
                sample_nu = self.mu
                sample_zeta_square = self.sigma

            # initialize lambda for this sample
            if not self.collapse2ctm:
                beta = np.random.beta(vartheta_1, vartheta_2)
                sample_lam = np.random.binomial(1, beta, size=(self.num_components))
                # initialize omega for this sample
                sample_omega = omega[idx, feature_idx]
            else:
                sample_lam = np.ones(self.num_components)

            prev_comp_distr = component_distribution[idx]

            for iter in np.arange(start=0, stop=self.max_inner_iter + 1):
                # Update log_o
                if self.collapse2ctm:
                    hold_norm = self.expected_log_phi[:, feature_idx]
                else:
                    # Update (1 - w_c) / (sum_{k=1}^{k=t} 1- w_k)
                    norm_wc = (1 - sample_omega)
                    norm_wc = norm_wc / np.sum(1 - sample_omega)
                    hold_norm = np.multiply(self.expected_log_phi[:, feature_idx], norm_wc)

                hold_norm = np.multiply(hold_norm, sample_lam[:, np.newaxis])
                if not self.collapse2ctm:
                    hold_norm = hold_norm - 1
                if self.use_features:
                    hold_norm = np.multiply(hold_norm, P)

                log_o = sample_nu[:, np.newaxis] + hold_norm + EPSILON
                log_o = log_o - logsumexp(log_o, axis=0)

                # update o
                o = np.exp(log_o + np.log(feature_count))

                # sum o over components
                sum_o = np.exp(logsumexp(log_o + np.log(feature_count), axis=1))

                if not transform:
                    # update sample_nu
                    arguments = (rho, sum_o, sample_feature_count)
                    sample_nu = self.__optimize_nu(nu=self.nu, arguments=arguments)

                    # update rho
                    rho = np.sum(np.exp(sample_nu + 0.5 * sample_zeta_square))

                    # update sample_zeta_square
                    arguments = (rho, sample_feature_count)
                    sample_zeta_square = self.__optimize_zeta_square(zeta_square=self.zeta_square,
                                                                     arguments=arguments)
                else:
                    sample_nu = self.mu
                    sample_zeta_square = self.sigma

                # update rho
                rho = np.sum(np.exp(sample_nu + 0.5 * sample_zeta_square))

                if not self.collapse2ctm:
                    # collect average lambda for this sample
                    if iter % 3 == 0:
                        if not transform:
                            # update
                            vartheta_1 = self.gamma + np.mean(sample_lam, axis=0)
                            vartheta_2 = self.kappa - np.mean(sample_lam, axis=0) + 1

                        # Estimate psi(vartheta_1) - psi(vartheta_2)
                        hold_psi_vartheta = psi(vartheta_1) - psi(vartheta_2)

                        hold_lambda = np.zeros((self.max_sampling, self.num_components))
                        temp = np.multiply(o, hold_norm)
                        temp = np.mean(temp, axis=1)
                        temp = temp / temp.sum(axis=0)
                        for i in np.arange(self.max_sampling):
                            beta = np.random.beta(vartheta_1, vartheta_2)
                            hold_bernoulli = np.random.binomial(1, beta, size=(self.num_components))
                            hold_lambda[i] = 1 / (1 + np.exp(-(hold_psi_vartheta + temp)))
                            hold_lambda[i] = np.random.binomial(1, hold_lambda[i], size=(self.num_components))
                            hold_lambda[i] = hold_lambda[i] * hold_bernoulli
                        sample_lam = np.mean(hold_lambda, axis=0)
                        sample_lam[sample_lam >= 0.5] = 1.
                        sample_lam[sample_lam < 0.5] = EPSILON

                        # Update sample_omega
                        # Update (1 - w_c - (sum_{k=1}^{k=t} 1- w_k)) / (sum_{k=1}^{k=t} 1- w_k)**2
                        norm_wc = (1 - sample_omega - np.sum(1 - sample_omega))
                        norm_wc = norm_wc / np.sum(1 - sample_omega) ** 2
                        temp = np.multiply(o, sample_lam[:, np.newaxis])
                        temp = np.multiply(temp, self.expected_log_phi[:, feature_idx])
                        sum_o_norm = np.sum(np.multiply(temp, norm_wc), axis=0)
                        sample_omega = omega[idx, feature_idx]
                        sample_omega = sample_omega - np.multiply(sum_o_norm, norm_wc)
                        sample_omega = np.array(sample_omega.flat)
                        sample_omega[sample_omega < 0] = EPSILON
                        sample_omega = sample_omega / np.sum(sample_omega)

                # If curr_cmp_distr hasn't changed much, we're done.
                curr_cmp_distr = (sum_o + EPSILON) / np.sum(sum_o + EPSILON)
                meanchange = np.mean(np.abs(curr_cmp_distr, prev_comp_distr))
                if meanchange < self.component_threshold:
                    break

            # Contribution of an example i to the expected sufficient
            # statistics for the M step.
            mu_sstats += sample_nu
            temp = np.dot(sample_nu[:, np.newaxis], sample_nu[:, np.newaxis].T)
            if not transform:
                sigma_sstats += np.diag(sample_zeta_square) + temp
            else:
                sigma_sstats += sample_zeta_square + temp

            lambda_sstats += sample_lam
            lambda_sstats = lambda_sstats / (lambda_sstats.sum(axis=0) + EPSILON)
            if not self.collapse2ctm:
                omega_sstats[feature_idx] += sample_omega
                omega[idx, feature_idx] = sample_omega
            temp = np.exp(log_o + np.log(feature_count))
            temp = temp / temp.sum(axis=0)
            temp = self.__check_bounds(X=temp)
            o_sstats[:, feature_idx] += temp
            # compute the normalized components weights across samples
            if not self.collapse2ctm:
                component_distribution[idx, :] = np.multiply(component_distribution[idx], sample_lam)
            component_distribution[idx, :] = (sum_o + EPSILON) / np.sum(sum_o + EPSILON)
        if not self.collapse2ctm:
            if self.top_k > 0:
                offset = np.arange(o_sstats.shape[1])
                for j in np.arange(self.num_components):
                    temp = np.argsort(-1 * o_sstats)[j, :self.top_k]
                    temp = [idx for idx in offset if idx not in temp]
                    if len(temp) > 0:
                        o_sstats[j, temp] = EPSILON
        # store sufficient_stats in a dictionary
        sufficient_stats = {"mu_sstats": mu_sstats, "sigma_sstats": sigma_sstats,
                            "lambda_sstats": lambda_sstats, "omega_sstats": omega_sstats,
                            "o_sstats": o_sstats}
        if not transform:
            component_distribution = None

        return sufficient_stats, component_distribution

    def __batch_e_step(self, X, M, features, list_batches, transform=False, verbose=True):
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads", verbose=max(0, self.verbose - 1))
        if not self.collapse2ctm:
            results = parallel(delayed(self.__e_step)(X[batch:batch + self.batch],
                                                      M[batch:batch + self.batch],
                                                      features, idx, len(list_batches),
                                                      transform, verbose)
                               for idx, batch in enumerate(list_batches))
        else:
            results = parallel(delayed(self.__e_step)(X[batch:batch + self.batch], M,
                                                      features, idx, len(list_batches),
                                                      transform, verbose)
                               for idx, batch in enumerate(list_batches))
        # merge result
        sstats_list, component_distribution = zip(*results)
        del results

        mu_sstats = 0
        sigma_sstats = 0
        o_sstats = 0
        lambda_sstats = 0
        omega_sstats = 0

        for lst in sstats_list:
            mu_sstats += lst["mu_sstats"]
            sigma_sstats += lst["sigma_sstats"]
            o_sstats += lst["o_sstats"]
            lambda_sstats += lst["lambda_sstats"]
            if not self.collapse2ctm:
                omega_sstats += lst["omega_sstats"]

        if transform:
            component_distribution = np.vstack(component_distribution)
            component_distribution /= component_distribution.sum(1)[:, np.newaxis]

        # store sufficient_stats in a dictionary
        sufficient_stats = {"mu_sstats": mu_sstats, "sigma_sstats": sigma_sstats,
                            "lambda_sstats": lambda_sstats, "omega_sstats": omega_sstats,
                            "o_sstats": o_sstats}

        return sufficient_stats, component_distribution

    def __m_step(self, sstats, num_samples, learning_rate):
        """
        Optimize model's parameters using the statictics collected during the e-step
        :param num_samples: 
        """
        desc = '\t       --> Computing M-step...'
        print(desc)

        # compute mean values
        mean_mu_sstats = sstats["mu_sstats"] / num_samples
        mean_sigma_sstats = sstats["sigma_sstats"] / num_samples
        mean_lam = sstats["lambda_sstats"] / num_samples

        # update component feature distributions
        temp = sstats["o_sstats"]
        phi = temp + self.alpha_phi_vec
        self.phi = (1 - learning_rate) * self.phi + phi * learning_rate
        self.phi = self.phi / np.sum(self.phi, axis=1)[:, np.newaxis]
        self.expected_log_phi = self.__dirichlet_expectation(alpha=self.phi)

        # update mu and sigma
        self.mu = (1 - learning_rate) * self.mu + mean_mu_sstats * learning_rate
        self.mu = self.__check_bounds(X=self.mu)
        temp = mean_sigma_sstats + np.dot(self.mu[:, np.newaxis], self.mu[:, np.newaxis].T)
        self.sigma = (1 - learning_rate) * self.sigma + temp * learning_rate
        self.sigma = self.__check_bounds(X=self.sigma)
        self.sigma_inv = np.linalg.pinv(self.sigma)

        # update rho
        sum_rho = np.sum(np.exp(self.nu + 0.5 * self.zeta_square))
        self.rho = (1 - learning_rate) * self.rho + sum_rho * learning_rate

        # update nu and zeta square
        self.nu = (1 - learning_rate) * self.nu + self.mu * learning_rate
        self.zeta_square = (1 - learning_rate) * self.zeta_square + np.diag(self.sigma) * learning_rate

        # update vartheta_1 and vartheta_2
        curr_vartheta = np.mean((self.gamma * self.num_components + mean_lam) / self.num_components)
        self.vartheta_1 = (1 - learning_rate) * self.vartheta_1 + curr_vartheta * learning_rate
        curr_vartheta = np.mean((self.kappa * self.num_components - mean_lam + 1) / self.num_components)
        self.vartheta_2 = (1 - learning_rate) * self.vartheta_2 + curr_vartheta * learning_rate

    def __elbo(self, num_samples, num_features, M, omega_sstats, lambda_sstats, o_sstats):
        score = 0.0

        # add smoothing term
        omega_sstats = omega_sstats + EPSILON
        lambda_sstats = lambda_sstats + EPSILON
        o_sstats = o_sstats + EPSILON

        # compute mean values
        mean_lam = lambda_sstats / num_samples
        mean_omega = omega_sstats / num_samples

        # E[log p(Phi | alpha)] - E[log q(Phi | phi)]
        temp = self.alpha_phi_vec - 1
        score += (gammaln(np.sum(self.alpha_phi_vec)) - np.sum(gammaln(self.alpha_phi_vec))) * self.num_components
        score += np.sum(np.multiply(temp[np.newaxis, :], self.expected_log_phi))
        score -= np.sum(gammaln(np.sum(self.phi, axis=1))) + np.sum(gammaln(self.phi))
        score -= np.sum(np.multiply((self.phi - 1), self.expected_log_phi))

        # E[log p(eta | mu, Sigma)] - E[log q(eta | lambda, nu_square)]
        det = np.linalg.slogdet(self.sigma_inv + EPSILON)
        score += 0.5 * det[0] * det[1]
        score -= 0.5 * self.num_components * np.log(2 * np.pi)
        score -= 0.5 * np.trace(np.diag(self.zeta_square) * self.sigma_inv)
        temp = np.dot((self.mu - self.nu), self.sigma_inv)
        score -= 0.5 * np.dot(temp, (self.mu - self.nu).T) * num_samples
        temp = 0.5 * (np.log(self.zeta_square + EPSILON) + np.log(2 * np.pi) + 1)
        score += np.sum(temp) * num_samples

        if not self.collapse2ctm:
            # E[log p(Lambda | beta)] - E[log q(Lambda | lambda)]
            hold_vartheta = np.array([self.vartheta_1] + [self.vartheta_2])
            expected_log_beta = self.__dirichlet_expectation(alpha=hold_vartheta, beta=self.vartheta_1)
            expected_log_beta_minus = self.__dirichlet_expectation(alpha=hold_vartheta, beta=self.vartheta_2)
            score += np.sum(mean_lam * expected_log_beta + (1 - mean_lam) * expected_log_beta_minus)
            score -= np.sum(np.multiply(mean_lam, np.log(mean_lam)))
            score -= np.sum(np.multiply(1 - mean_lam, np.log(1 - mean_lam)))

            # E[log p(beta | gamma, kappa)] - E[log q(beta | vartheta_1, vartheta_2)]
            temp = (self.gamma - 1) * expected_log_beta + (self.kappa - 1) * expected_log_beta_minus
            temp -= np.log(beta(self.gamma, self.kappa))
            score += np.sum(temp) * self.num_components * num_samples
            temp = (self.vartheta_1 - 1) * expected_log_beta + (self.vartheta_2 - 1) * expected_log_beta_minus
            temp -= np.log(beta(self.vartheta_1, self.vartheta_2))
            score -= np.sum(temp) * self.num_components * num_samples

            # E[log p(Omega | M, xi)] - E[log q(Omega | omega)]
            ## TODO: uncomment if necessary
            # expected_log_omega = self.__dirichlet_expectation(omega_sstats)
            # score += (np.sum(gammaln(np.sum(self.xi_vec + M, axis=1))) - np.sum(gammaln(self.xi_vec + M)))
            # score += np.sum(np.multiply((self.xi_vec + M - 1), expected_log_omega))
            # score -= (gammaln(np.sum(omega_sstats)) - np.sum(gammaln(omega_sstats)))
            # score -= np.sum(np.multiply((omega_sstats - 1), expected_log_omega))

        # E[log p(z | eta)] - E[log q(z | o)]
        score += ((1 - np.log(self.rho + EPSILON)) * num_features)
        score += np.sum(np.dot(self.nu, o_sstats))
        score -= (np.sum(np.exp(self.nu + 0.5 * self.zeta_square)) / self.rho) * num_features
        score -= np.sum(np.multiply(o_sstats, np.log(o_sstats)))

        # E[log p(y | z, Omega, Phi, Lambda, varpi)]
        temp = np.multiply(o_sstats, self.expected_log_phi)
        if not self.collapse2ctm:
            temp = np.multiply(o_sstats, mean_lam[:, np.newaxis])
            temp = np.multiply(temp, mean_omega)
            temp = np.multiply(temp, self.expected_log_phi)
            score += np.log(self.varpi + EPSILON) * num_features

        score += np.sum(temp)

        return float(score)

    def fit(self, X, M=None, features=None, model_name='soap', model_path="../../model", result_path=".",
            display_params: bool = True):
        # validate inputs
        if X is None:
            raise Exception("Please provide a dataset.")
        assert X.shape[1] == self.num_features
        X = self.__check_non_neg_array(X, "diSparseCorrelatedBagPathway.fit")

        if not self.collapse2ctm:
            if M is not None:
                assert M.shape == X.shape
            else:
                M = np.zeros((X.shape[0], self.num_features))

            if features is not None:
                assert X.shape[1] == features.shape[0]
            else:
                features = np.ones((self.num_features, 20))
            features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]

        # collect properties from data
        self.__init_latent_variables()
        num_samples = int(X.shape[0] * self.subsample_input_size)
        list_batches = np.arange(start=0, stop=num_samples, step=self.batch)

        if display_params:
            self.__print_arguments()
            time.sleep(2)

        cost_file_name = model_name + "_cost.txt"
        save_data('', file_name=cost_file_name, save_path=result_path, mode='w', w_string=True, print_tag=False)

        print('\t>> Training by SPREAT model...')
        logger.info('\t>> Training by SPREAT model...')
        n_epochs = self.num_epochs + 1
        old_bound = np.inf

        timeref = time.time()

        for epoch in np.arange(start=1, stop=n_epochs):
            desc = '\t   {0:d})- Epoch count ({0:d}/{1:d})...'.format(epoch, n_epochs - 1)
            print(desc)
            logger.info(desc)

            learning_rate = np.power((epoch + self.delay_factor), -self.forgetting_rate)

            # Subsample dataset
            idx = np.random.choice(X.shape[0], num_samples, False)
            start_epoch = time.time()

            # E-step
            if not self.collapse2ctm:
                sstats, tmp = self.__batch_e_step(X=X[idx, :], M=M[idx, :], features=features,
                                                  list_batches=list_batches)
            else:
                sstats, tmp = self.__batch_e_step(X=X[idx, :], M=None, features=features, list_batches=list_batches)
            del tmp

            # M-step
            self.__m_step(sstats=sstats, num_samples=num_samples, learning_rate=learning_rate)

            end_epoch = time.time()

            self.is_fit = True

            # Compute approx bound
            if not self.collapse2ctm:
                new_bound = self.perplexity(X=X[idx, :], M=M[idx, :], features=features, sstats=sstats)
            else:
                new_bound = self.perplexity(X=X[idx, :], M=M, features=features, sstats=sstats)
            print('\t\t## Epoch {0} took {1} seconds...'.format(epoch, round(end_epoch - start_epoch, 3)))
            logger.info('\t\t## Epoch {0} took {1} seconds...'.format(epoch, round(end_epoch - start_epoch, 3)))
            data = str(epoch) + '\t' + str(round(end_epoch - start_epoch, 3)) + '\t' + str(new_bound) + '\n'
            save_data(data=data, file_name=cost_file_name, save_path=result_path, mode='a', w_string=True,
                      print_tag=False)
            # Save models parameters based on test frequencies
            if (epoch % self.display_interval) == 0 or epoch == 1 or epoch == n_epochs - 1:
                print('\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_bound, old_bound))
                logger.info('\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_bound, old_bound))
                if new_bound <= old_bound or epoch == n_epochs - 1:
                    phi_file_name = model_name + '_exp_phi.npz'
                    sigma_file_name = model_name + '_sigma.npz'
                    mu_file_name = model_name + '_mu.npz'
                    model_file_name = model_name + '.pkl'
                    if epoch == n_epochs - 1:
                        phi_file_name = model_name + '_exp_phi_final.npz'
                        sigma_file_name = model_name + '_sigma_final.npz'
                        mu_file_name = model_name + '_mu_final.npz'
                        model_file_name = model_name + '_final.pkl'

                    print('\t\t  --> Storing the SPREAT phi to: {0:s}'.format(phi_file_name))
                    logger.info('\t\t  --> Storing the SPREAT phi to: {0:s}'.format(phi_file_name))
                    np.savez(os.path.join(model_path, phi_file_name), self.phi)

                    print('\t\t  --> Storing the SPREAT sigma to: {0:s}'.format(sigma_file_name))
                    logger.info('\t\t  --> Storing the SPREAT sigma to: {0:s}'.format(sigma_file_name))
                    np.savez(os.path.join(model_path, sigma_file_name), self.sigma)

                    print('\t\t  --> Storing the SPREAT mu to: {0:s}'.format(mu_file_name))
                    logger.info('\t\t  --> Storing the SPREAT mu to: {0:s}'.format(mu_file_name))
                    np.savez(os.path.join(model_path, mu_file_name), self.mu)

                    print('\t\t  --> Storing the SPREAT model to: {0:s}'.format(model_file_name))
                    logger.info('\t\t  --> Storing the SPREAT model to: {0:s}'.format(model_file_name))
                    save_data(data=copy.copy(self), file_name=model_file_name, save_path=model_path, mode="wb",
                              print_tag=False)
                    old_bound = new_bound
        print('\t  --> Training consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))
        logger.info('\t  --> Training consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))

    def __transform(self, X, M=None, features=None):
        num_samples = X.shape[0]
        X = self.__check_non_neg_array(X, "diSparseCorrelatedBagPathway.fit")
        list_batches = np.arange(start=0, stop=num_samples, step=self.batch)
        sstats, component_distribution = self.__batch_e_step(X=X, M=M, features=features, list_batches=list_batches,
                                                             transform=True)
        return sstats, component_distribution

    def transform(self, X, M=None, features=None, batch_size=30, num_jobs=1):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        M, features = self.__set_variables(X, M, features, batch_size, num_jobs)
        _, component_distribution = self.__transform(X=X, M=M, features=features)
        component_distribution = component_distribution
        return component_distribution

    def perplexity(self, X, M=None, features=None, log_space: bool = True, sstats=None, per_feature=True,
                   per_component=False, batch_size=30, num_jobs=1):
        M, features = self.__set_variables(X, M, features, batch_size, num_jobs)
        # collect properties from data
        num_samples = X.shape[0]
        num_features = np.sum(X)
        num_components = self.num_components

        if sstats is None:
            sstats, _ = self.__transform(X=X, M=M, features=features)

        perplexity = self.__elbo(num_samples=num_samples, num_features=num_features,
                                 M=M, omega_sstats=sstats["omega_sstats"],
                                 lambda_sstats=sstats["lambda_sstats"],
                                 o_sstats=sstats["o_sstats"])

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

    def score(self, X, M=None, features=None, log_space: bool = True, batch_size=30, num_jobs=1):
        """Calculate approximate log-likelihood as score."""
        M, features = self.__set_variables(X, M, features, batch_size, num_jobs)
        # collect properties from data
        num_samples = X.shape[0]
        num_features = np.sum(X)

        sstats, _ = self.__transform(X=X, M=M, features=features)
        score = self.__elbo(num_samples=num_samples, num_features=num_features,
                            M=M, omega_sstats=sstats["omega_sstats"],
                            lambda_sstats=sstats["lambda_sstats"],
                            o_sstats=sstats["o_sstats"])
        if log_space:
            score = np.log(score)
        return score

    def get_components(self):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        return self.phi

    def get_component_features(self, component_idx: int, top_k=10):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        component = self.get_components()[component_idx]
        component = component / component.sum()  # normalize to probability distribution
        best_k = np.argsort(component)[::-1][:top_k]
        return [(idx, component[idx]) for idx in best_k]

    def get_feature_components(self, feature_idx: int, minimum_probability: float = 0.00001):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output
        component_prob = [(idx, self.phi[idx][feature_idx]) for idx in np.arange(0, self.num_components)
                          if self.phi[idx][feature_idx] >= minimum_probability]
        return component_prob

    def get_example_components(self, X, M=None, features=None, minimum_probability: float = 0.00001,
                               batch_size=30, num_jobs=1):
        assert X.shape[0] == 1
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        M, features = self.__set_variables(X, M, features, batch_size, num_jobs)
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output
        sstats, component_distribution = self.__transform(X=X, M=M, features=features)
        component_feature = sstats["o_sstats"]
        component_feature = csr_matrix(X).multiply(component_feature).toarray()
        component_feature[component_feature < minimum_probability] = 0
        component_feature_id = []
        feature_component_id = []
        for a in np.arange(self.num_components):
            features = np.argsort(component_feature[a])[::-1]
            temp = np.trim_zeros(component_feature[a, features])
            features = features[:len(temp)]
            component_feature_id.append((a, [self.vocab[f] for f in features]))
        for t in np.arange(self.num_features):
            components = np.argsort(component_feature[:, t])[::-1]
            feature_component_id.append((t, components, component_feature[components, t]))
        return component_distribution, component_feature_id, feature_component_id

    def get_component_distribution(self, X, M=None, features=None, minimum_probability: float = 0.00001,
                                   batch_size=30, num_jobs=1):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        M, features = self.__set_variables(X, M, features, batch_size, num_jobs)
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output
        _, component_distribution = self.__transform(X=X, M=M, features=features)
        component_distribution[component_distribution < minimum_probability] = 0.
        return component_distribution

    def predictive_distribution(self, X, M=None, features=None, cal_average=True,
                                batch_size=30, num_jobs=1):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")

        X = self.__check_non_neg_array(X, "diSparseCorrelatedBagPathway.fit")
        M, features = self.__set_variables(X, M, features, batch_size, num_jobs)
        num_samples = X.shape[0]
        _, component_distribution = self.__transform(X=X, M=M, features=features)
        score = 0
        # iterate over all samples
        for idx in np.arange(num_samples):
            feature_idx = X[idx].indices
            temp = np.multiply(component_distribution[idx][:, np.newaxis], self.phi[:, feature_idx])
            score += np.sum(temp)
        if cal_average:
            score = score / num_samples
        return np.log(score + EPSILON)
