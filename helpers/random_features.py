import numpy as np
import pandas as pd
from enum import Enum
import time
import logging
import os
from helpers.auxilliary_functions import vol_adjust_data, sharpe_ratio_by_freq, \
    get_all_random_matrix_quantities_for_the_panel, \
    regression_with_tstats, sharpe_ratio, rank_features_cross_sectionally
from matplotlib import pyplot as plt
from datetime import datetime


# export PYTHONPATH=$PYTHONPATH:/helpers


def select_balanced_panel(raw_signals, labels, date_ids, stock_ids):
    """
    Select the observations of stocks that exist the whole time

    :param raw_signals:
    :type raw_signals:
    :param labels:
    :type labels:
    :param date_ids:
    :type date_ids:
    :param stock_ids:
    :type stock_ids:
    :return:
    :rtype:
    """
    # # only look at stocks with full observations
    date_stock_ids = pd.Series(index=date_ids.flatten(), data=stock_ids.flatten(), name='id')
    stock_obs_counts = date_stock_ids.reset_index().groupby('id').count()['index']
    good_stock_subset = stock_obs_counts[stock_obs_counts == len(np.unique(date_ids))].index

    # filter only observations of full sample stocks
    indicator = np.isin(stock_ids, good_stock_subset)
    date_ids = date_ids[indicator]
    stock_ids = stock_ids[indicator]
    labels = labels[indicator]
    raw_signals = raw_signals[indicator, :]
    return raw_signals, labels, date_ids, stock_ids


class RFTypes(Enum):
    NONE = 1
    GAUSSIAN_MIXTURE = 2
    GAUSSIAN = 3


class RandomFeatures:
    def __init__(self):
        self.number_of_models = None
        self.size = None
        self.length_of_mixture_per_model = None
        self.number_of_random_features = None
        self.activation = None
        self.type = RFTypes.NONE
        self.bias = []
        self.feature_weights = []

    def apply_activation_to_multiplied_signals(self, multiplied_signals, activation):
        """
        this method takes as input signaled already multipled by some weights + cosntant: w*x+b
        and returns act(w*x+b)
        :rtype: object
        """
        if activation in ['cos', 'sin', 'exp', 'arctan', 'tanh']:
            final_random_features = getattr(np, activation)(multiplied_signals)
        elif activation == 'cos_and_sin':
            final_random_features = np.concatenate([np.cos(multiplied_signals),
                                                    np.sin(multiplied_signals)], axis=0)

        elif activation in ['ReLu', 'relu', 'Relu']:
            final_random_features = multiplied_signals * (multiplied_signals > 0)
        elif isinstance(activation, list) and activation[0] in ['Elu' 'ELU', 'Elu', 'elu']:
            final_random_features = (multiplied_signals * (multiplied_signals > 0)) \
                                    + activation[1] * (np.exp(multiplied_signals) - 1) \
                                    * (multiplied_signals < 0)
        elif activation in ['SoftPlus', 'softPlus', 'Softplus', 'softplus']:
            final_random_features = np.log(1 + np.exp(multiplied_signals))
        elif activation.lower() == 'linear':
            print('--- running linear', flush=True)
            final_random_features = multiplied_signals
        else:
            breakpoint()
            raise Exception(f'activation function{activation} is not yet supported')
        return final_random_features

    @staticmethod
    def _generate_random_gaussians(seed,
                                   size,
                                   length_of_mixture_per_model,
                                   sigma_identity_shrink: int = 0,
                                   mu_scale: int = 2):
        """
        This method generate a set of mixture models
        :param seed:
        :param size:
        :param length_of_mixture_per_model:
        :param sigma_identity_shrink: we shrink covariance towards identity,
        so that we could continuously recover i.i.d. features
        (standard i.i.d. features correspond to (mu,\sigma) = (0, 1)
        :return:
        """
        np.random.seed(seed)
        # I am using 2 *size here so that the matrix is sufficiently non-degenerate
        gaussian_mix = list()
        for i in range(length_of_mixture_per_model):
            # this is how we generate a random covariance matrix
            tmp = np.random.normal(size=[size, 2 * size])
            sigma = np.matmul(tmp, tmp.T) / size

            sigma = (1 - 0.001) * sigma + 0.001 * np.diag(np.diag(sigma))
            # a bit of diagonal shrinkage to ensure the matrix is non-degenerate
            sigma = sigma_identity_shrink * np.eye(size) + (1 - sigma_identity_shrink) * sigma
            if sigma_identity_shrink > 1:
                print(f'warning !!!!!! {sigma_identity_shrink} is above 1 !!!!!!!')
                breakpoint()

            mu = np.random.normal(size=[size, 1])
            a = np.random.uniform(0, mu_scale)
            mu *= a
            gaussian_mix += [(mu, sigma)]
        return gaussian_mix

    def generate_lots_of_different_gaussian_mixtures(self,
                                                     number_of_models,
                                                     size,
                                                     length_of_mixture_per_model,
                                                     number_features,
                                                     sigma_identity_shrink: int = 0,
                                                     mu_scale: int = 2
                                                     ):
        list_of_models = list()
        for seed in range(number_of_models):
            model = self._generate_random_gaussians(seed=seed,
                                                    size=size,
                                                    length_of_mixture_per_model=length_of_mixture_per_model,
                                                    sigma_identity_shrink=sigma_identity_shrink,
                                                    mu_scale=mu_scale)
            list_of_models += [model]
            partitions = self._generate_random_partitions(
                number_of_models,
                length_of_mixture_per_model,
                number_features,
                seed=number_of_models * 1000)
        return partitions, list_of_models

    @staticmethod
    def _generate_random_partitions(number_models, length_of_mixture_per_model, number_features, seed=0):
        # Create a new pseudo random number generator
        np.random.seed(seed)
        integers = np.random.uniform(1, int(number_features), size=[number_models, length_of_mixture_per_model])
        integers = integers / np.sum(integers, 1).reshape(-1, 1)
        integers *= number_features
        # now, these integers have a sum equal to number_features
        return integers  # they are not really integers

    def add_bias(self,
                 multiplied_signals,
                 bias_distribution,
                 bias_distribution_parameters,
                 seed=0):
        """
        Careful, multiplied signals are assumed to be P \times n where P is the sumber of signals
        and n the number of observations
        Parameters
        ----------
        multiplied_signals :
        bias_distribution :
        bias_distribution_parameters :
        Returns
        -------
        """
        np.random.seed(seed)
        number_features = multiplied_signals.shape[0]
        random_bias = getattr(np.random, bias_distribution)(*bias_distribution_parameters, [number_features, 1])
        # here we are using numpy broadcasting to add the same bias every time period
        multiplied_signals += random_bias
        self.bias.append(random_bias)
        return multiplied_signals

    @staticmethod
    def gaussian_mixture_weights(partition, list_of_gaussians, seed):
        np.random.seed(seed)
        L = [np.random.multivariate_normal(list_of_gaussians[i][0][:, 0], list_of_gaussians[i][1],
                                           [max(int(number_features), 1)])
             for i, number_features in enumerate(partition)]
        feature_weights = np.concatenate(L, axis=0)
        return feature_weights

    def add_random_features_gaussian_times_gamma(self,
                                                 raw_signals,
                                                 number_of_models,
                                                 number_of_random_features,
                                                 gamma_low: float = 1.0,
                                                 gamma_high: float = 1.5,
                                                 seed=1,
                                                 activation='cos',
                                                 ):
        """
        raw_signals are original (low dimensional) features
        For these, we build complex random features

        Parameters
        ----------
        number_of_models :
        raw_signals :
        number_of_random_features :
        activation :

        Returns
        -------
        """
        # save type of rf in object
        self.type = RFTypes.GAUSSIAN
        self.number_of_random_features = number_of_models * number_of_random_features
        self.activation = activation
        if number_of_models * number_of_random_features > 0:
            # print('generate data with', number_of_models * number_of_random_features, 'number_of_random_features')
            # semyon's version
            intrinsic_dimension = raw_signals.shape[1]
            weight_matrix = self.gaussian_simple_weight(intrinsic_dimension=intrinsic_dimension,
                                                        number_of_random_features=number_of_models * number_of_random_features,
                                                        seed=seed)
            # we generate random gammas from a uniform distribution
            temp = np.tile(np.random.uniform(gamma_low, gamma_low + gamma_high,
                                             size=(number_of_models, 1)), reps=(1, number_of_random_features))

            temp = np.concatenate(np.array_split(temp, temp.shape[0], axis=0), axis=1)
            temp = np.tile(temp, reps=(weight_matrix.shape[0], 1))
            weight_matrix = weight_matrix * temp

            multiplied_signals = raw_signals @ weight_matrix
            # note the transpose before and after, somehow add bias
            if activation == 'cos':
                mult = 1.
            else:
                mult = 0.0001
            multiplied_signals = self.add_bias(multiplied_signals.T, bias_distribution='uniform',
                                               bias_distribution_parameters=[- mult * np.pi, mult * np.pi]).T
            random_features = self.apply_activation_to_multiplied_signals(multiplied_signals, activation)

        else:
            random_features = raw_signals
        return random_features

    def add_random_features_gaussian_mix(self,
                                         raw_signals,
                                         number_of_models,
                                         length_of_mixture_per_model,
                                         number_of_random_features,
                                         activation='cos',
                                         mu_scale: int = 2,
                                         sigma_identity_shrink: int = 0):
        """
        This function is a rough adaptation of the one I found in the old random_features. I didn't change many things.
        Just a few transpose that didn't work.
        :param raw_signals:
        :param number_of_models:
        :param length_of_mixture_per_model:
        :param number_of_random_features: This is the number of random features PER MODEL. So, the actual
        final number of random features is number_of_models * number_of_random_features

        :param activation:
        :param mu_scale: very important (should not be too large), determines the size of mu in the Gaussian mixtures
        :param sigma_identity_shrink: very important as well. Should be between 0 and 1
        when equal to 1, means we always generate Gaussian featured with identity covariance matrix
        :return:
        """
        # save type of rf in object
        self.type = RFTypes.GAUSSIAN_MIXTURE
        # saving the basic_parameters in object
        self.number_of_models = number_of_models
        self.size = raw_signals.shape[1]
        self.length_of_mixture_per_model = length_of_mixture_per_model
        self.number_of_random_features = number_of_random_features
        self.activation = activation
        partitions, list_of_models = self.generate_lots_of_different_gaussian_mixtures(
            number_of_models=number_of_models,
            size=self.size,
            length_of_mixture_per_model=length_of_mixture_per_model,
            number_features=number_of_random_features,
            mu_scale=mu_scale,
            sigma_identity_shrink=sigma_identity_shrink)
        # now, if number_of_models is large, then storing all random features in memory becomes impossible
        # but for a small number of models, we can still play with it
        # TODO: I USED LIST COMPREHENSION TO DRASTICALLY SPEED UP THIS LOOP
        giant_set_of_random_features = list()
        all_weights = np.concatenate([self.gaussian_mixture_weights(partitions[i],
                                                                    list_of_models[i],
                                                                    seed=1000 * i + number_of_models).T * np.sqrt(
            1 / self.size)
                                      for i in range(number_of_models)], axis=1)

        multiplied_signals = raw_signals @ all_weights
        # this is standard for RFF
        # now we want to make sure that each block of signals has a different bias
        # hence we need to re-shuffle a bit
        bias_distribution = 'uniform'  # for now it's not an input but we can change this latter when needed
        bias_distribution_parameters = [0, 2 * np.pi]
        if bias_distribution != 'none':
            # TODO careful here: in add_bias, signals MUST BE P TIMES N
            # todo this was a major bug. when incorrectly applied (along wrong dimension), it means that
            # the bias will be different in- and out-of-sample, breaking predictability !!!!
            multiplied_signals = self.add_bias(multiplied_signals.T,
                                               bias_distribution,
                                               bias_distribution_parameters,
                                               seed=len(partitions)).T
        final_random_features = self.apply_activation_to_multiplied_signals(multiplied_signals, activation)
        return final_random_features

    def gaussian_simple_weight(self,
                               intrinsic_dimension,
                               number_of_random_features,
                               seed):
        """
        :param intrinsic_dimension: number of true feature
        :param number_of_random_features: number of random feature to generate
        :return: weights w for rnd feature of type act(wX+b)
        """
        np.random.seed(seed)
        w = np.random.normal(size=[intrinsic_dimension, number_of_random_features]) * np.sqrt(1 / intrinsic_dimension)
        self.feature_weights.append(w)
        return w

    def add_random_features_simple_gaussian(self,
                                            raw_signals,
                                            number_of_random_features,
                                            seed,
                                            gamma=1,
                                            activation='cos',
                                            uniform_bias_interval=[-np.pi, np.pi]
                                            ):
        """
        raw_signals are original (low dimensional) features
        For these, we build complex random features

        Parameters
        ----------
        seed :
        gamma :
        uniform_bias_interval :
        raw_signals :
        number_of_random_features :
        activation :

        Returns
        -------

        """
        # save type of rf in object
        self.type = RFTypes.GAUSSIAN
        self.number_of_random_features = number_of_random_features
        self.activation = activation

        if number_of_random_features > 0:
            # print('generate data with', number_of_random_features, 'number_of_random_features')
            intrinsic_dimension = raw_signals.shape[1]
            weights = self.gaussian_simple_weight(intrinsic_dimension=intrinsic_dimension,
                                                  number_of_random_features=number_of_random_features,
                                                  seed=seed) * gamma
            multiplied_signals = raw_signals @ weights
            # note the transpose before and after, somehow add bias
            multiplied_signals = self.add_bias(multiplied_signals.T, bias_distribution='uniform',
                                               bias_distribution_parameters=uniform_bias_interval).T
            random_features = self.apply_activation_to_multiplied_signals(multiplied_signals, activation)

        else:
            random_features = raw_signals
        return random_features

    def optimal_weight_for_ensembling_across_different_bstar(self, in_sample_signals, bstar, residual_bstar,
                                                             shrinkage_list):
        """
        We use theoretical values from random matrix theory to compute optimal weight
        in_sample_signals are assumed to be n \times P.
        So that their covariance matrix is np.matmul(in_sample_signals.T, in_sample_signals)
        Parameters
        ----------
        in_sample_signals :
        bstar :
        residual_bstar :
        shrinkage_list :

        Returns
        -------

        """

        c_ = in_sample_signals.shape[1] / in_sample_signals.shape[0]  # complexity

        z_ = np.array(shrinkage_list).reshape(-1, 1)  # vertical

        if in_sample_signals.shape[0] < in_sample_signals.shape[1]:
            data = in_sample_signals.T
        else:
            data = in_sample_signals

        signals_covariance = np.matmul(data.T, data) / in_sample_signals.shape[0]
        eigenvalues, _ = np.linalg.eigh(signals_covariance)  # eigenvalues are horizontal
        if in_sample_signals.shape[0] < in_sample_signals.shape[1]:
            true_eigenvalues = np.concatenate(
                [np.zeros([1, in_sample_signals.shape[1] - in_sample_signals.shape[0]]),
                 eigenvalues.reshape(1, -1)], axis=1)

        ce_from_paper, cl_from_paper \
            = self.ce_and_cl_from_paper_based_on_eigenvalues(true_eigenvalues, bstar, residual_bstar, z_, c_)
        optimal_weight = ce_from_paper / cl_from_paper
        return optimal_weight

    @staticmethod
    def naive_in_sample_bstar_estimate(in_sample_signals, in_sample_returns):
        """
        This is the naive approach akin to naive Bias: we ignore the correlation structure of signals!
        the assumption is that each \beta \sim N(0, b_*/P). Hence, E[\sum_i \beta_i^2] = b_*
        Parameters
        ----------
        in_sample_signals :
        in_sample_returns :

        Returns
        -------

        """
        all_betas = (in_sample_signals * in_sample_returns.reshape(-1, 1)).mean(0) \
                    / (in_sample_returns.reshape(-1, 1) ** 2).mean(0)
        bstar = (all_betas ** 2).reshape(-1, 1)
        return bstar

    @staticmethod
    def clip_number_at_zero(number):
        return number * (number > 0) + 0.000001 * (number <= 0)

    @staticmethod
    def smart_in_sample_bstar_estimate_based_on_mean(observed_mean,
                                                     z_,
                                                     c_,
                                                     xi,
                                                     sigma_hat):
        """
        THE MATHEMATICS ASSUMES THAT EVERYTHING HAS MEAN ZERO IN SAMPLE!
        IN PARTICULAR, IDEALLY, ALL FEATURES SHOULD BE DEMEANED,
        AND IN PARTICULAR, LABELS (1, 0) SHOULD BE REPLACED WITH LABELS - MEAN(LABELS)
        (WHERE MEAN(LABELS) IS TRIVIALLY JUST THE FREQUENCY OF OCCURRENCE)

        We are recovering true bstar from the observed norm.
        In theory,
        observed mean (labels * features) is given by an explicit formula

        Parameters
        ----------
        c_ :
        z_ :
        xi :
        Returns
        -------

        """
        z_ = np.array(z_).reshape(-1, 1)
        numerator = observed_mean - xi * sigma_hat / (1 + xi * sigma_hat)
        denominator = xi * (sigma_hat ** 2) / (1 + xi * sigma_hat) \
                      + sigma_hat * (1 - z_ * (1 / c_) * xi)
        theoretical_bstar = numerator / denominator
        theoretical_bstar = theoretical_bstar * (theoretical_bstar > 0)
        theoretical_bstar = theoretical_bstar.clip(10 ** (-10))
        return theoretical_bstar

    @staticmethod
    def smart_in_sample_bstar_estimate_based_on_bnorm(observed_norm_of_beta,
                                                      z_,
                                                      c_,
                                                      m,
                                                      m_prime,
                                                      xi,
                                                      xi_prime,
                                                      residual_bstar=0,
                                                      core_z_values=None):
        """
        We are recovering true bstar from the observed norm.
        In theory, observed_norm_of_beta
        = bstar (P_1/P) * (1 - 2 * z_ * m + (z_ ** 2) * m_prime) + c_ * m
        - z_ * c_ * m_prime - residual_bstar * xi_prime / ((1 + xi) ** 2)

        Parameters
        ----------
        xi_prime :
        c_ :
        z_ :
        observed_norm_of_beta :
        m :
        m_prime :
        xi :
        residual_bstar :

        Returns
        -------

        """
        z_ = np.array(z_).reshape(-1, 1)
        q1 = (1 - 2 * z_ * m + (z_ ** 2) * m_prime)
        q3 = - xi_prime / ((1 + xi) ** 2)
        q2 = c_ * m - z_ * c_ * m_prime
        theoretical = (observed_norm_of_beta - residual_bstar * q3 - q2) / q1

        # now we try to recover both the residual bstar
        if core_z_values is None:
            return theoretical
        else:
            z_index1 = np.argmin(np.abs(z_ - core_z_values[0]))
            z_index2 = np.argmin(np.abs(z_ - core_z_values[1]))
            matrix = np.array([[float(q1[z_index1]), float(q3[z_index1])], [float(q1[z_index2]), float(q3[z_index2])]])
            vector = np.array([observed_norm_of_beta[z_index1] - q2[z_index1],
                               observed_norm_of_beta[z_index2] - q2[z_index2]]).reshape(-1, 1)
            if np.min(np.linalg.eigh(matrix @ matrix.T)[0]) < 10 ** (-10):
                solution = np.array([0.00001, 0.00001]).reshape(-1, 1)
            else:
                solution = np.linalg.inv(matrix) @ vector
            bstar_estimate = RandomFeatures.clip_number_at_zero(solution.sum())
            residual_bstar_estimate = RandomFeatures.clip_number_at_zero(float(solution[1]))
            return theoretical, bstar_estimate, residual_bstar_estimate  # here we return bstar and residual_bstar

    def split_signals_according_to_betas(self, in_sample_signals, in_sample_returns, number_models):
        """
        we compute betas per signal using naive bias; then we split signals into groups
        Parameters
        ----------
        in_sample_signals :
        in_sample_returns :
        number_models :

        Returns
        -------

        """
        bstar = self.naive_in_sample_bstar_estimate(in_sample_signals, in_sample_returns)
        """
        the assumption is that each beta \sim N(0, b_*/P). Hence, E[\sum_i beta_i^2] = b_* 
        """
        quantiles = np.quantile(bstar, np.arange(0, 1 + 1 / number_models, 1 / number_models))
        signal_groups = [in_sample_signals[:, ((bstar > quantiles[i]) * (bstar <= quantiles[i + 1])).flatten()]
                         for i in range(len(quantiles) - 1)]
        bstar_per_group = [bstar[((bstar > quantiles[i]) * (bstar <= quantiles[i + 1])).flatten()].sum()
                           for i in range(len(quantiles) - 1)]
        return signal_groups, bstar_per_group

    @staticmethod
    def optimal_weight_vector_across_ridge_parameters(bstar,
                                                      residual_bstar,
                                                      psi_star,
                                                      xi,
                                                      xi_prime,
                                                      z_,
                                                      c_,
                                                      regularize=True):
        """
        this function uses random matrix theory to compute
        optimal combination of predictors with different ridge basic_parameters
        Parameters
        ----------
        bstar :
        residual_bstar :
        psi_star :
        xi :
        xi_prime :
        z_ :
        c_ :

        Returns
        -------

        """
        z_ = np.array(z_).reshape(1, -1)

        nu, nu_hat, nu_prime = RandomFeatures.all_the_different_nu_functions(psi_star, xi, xi_prime, z_, c_)
        function = (1 + residual_bstar) * (z_ * xi) - (bstar / c_) * (z_ ** 2) * xi
        diff = (function - function.T)
        divisor = (z_ - z_.T)
        divisor[divisor == 0] = np.inf
        diff = diff / divisor
        diff += bstar * psi_star  # every single element of the matrix must be added bstar * psi_star
        # now we add the diagonal, which is just the derivative
        diag = (1 + residual_bstar) * (xi + z_ * xi_prime) - (bstar / c_) * (2 * z_ * xi + (z_ ** 2) * xi_prime)
        # equivalently we could use
        # diag1 = - (1 + residual_bstar) * c_ * nu_prime + bstar * nu_hat - bstar * psi_star
        # # and, finally, the last equivalent formulation is
        ce_from_paper, cl_from_paper = \
            RandomFeatures.cl_and_ce_from_paper(bstar, residual_bstar, c_, nu, nu_hat, nu_prime)
        # diag2 = cl_from_paper - bstar * psi_star # this subtraction is important because
        # print(f'{diag - diag1}, {diag - diag2}')
        # breakpoint()
        # we already added bstar * psi_star
        diff += np.diag(diag.flatten())
        # now we regularize a bit
        eigval, _ = np.linalg.eigh(diff)

        # regularize a bit because we will be inverting this theoretical matrix
        if regularize:
            diff += np.eye(diff.shape[0]) * (eigval.sum() * 0.01 - min(np.min(eigval), 0))
        else:
            print(f'eigenvalues of the diff matrix are {eigval}')
        # print(f'eigenvalues of the diff matrix are {eigval}')

        # now we invert the matrix to get optimal_weights w=\Gamma^{-1}\gamma, according to the notation in the paper
        inverted = np.linalg.inv(diff)
        optimal_weights = np.matmul(inverted, bstar * nu.T)

        # this is the theoretical expected return on the portfolio
        ce_portfolio = (bstar * nu.T * optimal_weights).sum()

        # this is the theoretical risk of the portfolio
        cl_portfolio = float(optimal_weights.T @ (diff @ optimal_weights))  # (this is gamma'Gamma^{-1} gamma
        if not regularize:
            cl_portfolio_til = float(bstar * nu @ optimal_weights)
            print(f'formal stuff is {diff[-1, -1], bstar * nu[0, -1]}\n '
                  f' and quotient={(bstar * nu[0, -1]) ** 2 / diff[-1, -1]}, while cl_portfolio = {cl_portfolio_til}\n'
                  f'should be larger by Matrix Jensen')

        # if np.abs(cl_portfolio - cl_portfolio_til) > 0.0001:
        #     print(f'damn, matrix inversion is not working')
        #     breakpoint()

        return optimal_weights, ce_from_paper, cl_from_paper, ce_portfolio, cl_portfolio

    @staticmethod
    def mse(returns, predictions):
        # print(f'computing mse')
        components = np.zeros([1, 3])
        components[0, 0] = np.mean(returns ** 2)
        components[0, 1] = - 2 * np.mean(returns * predictions)
        components[0, 2] = np.mean(predictions ** 2)
        mse = np.mean((returns - predictions) ** 2, axis=0)
        # print(f'components: {components}; total is {components.sum()} = {mse}')
        return mse

    @staticmethod
    def produce_all_sorts_of_bstar_estimates(smart_bstar,
                                             upper_clip=10):
        smart_bstar = smart_bstar.clip(10 ** (-10), upper_clip).flatten()
        # note that some of them will be negative,
        names = ['mean', 'median', 'max-z']
        return [smart_bstar.mean(), np.quantile(smart_bstar, 0.5), smart_bstar[-1]], names

    @staticmethod
    def produce_all_sorts_of_bstar_estimates_old(result,
                                                 number_signals,
                                                 upper_clip=10):
        all_attempts_to_estimate_bstar = pd.DataFrame()

        # bstar had identical columns; hence .iloc[:, 0] below
        all_attempts_to_estimate_bstar['naive_bstar'] \
            = result['bstar'].iloc[:, 0]  # First good results were received with this
        # it is small and does not need to be adjusted
        # but then I realized that normalizing it is incorrect. In fact E[\sum_i \beta_i^2] = b_*
        # so it is sum and not mean that we need the following:
        all_attempts_to_estimate_bstar['naive_bstar_unadjusted'] \
            = number_signals * result['bstar'].iloc[:, 0].clip(upper=upper_clip)

        smart_bstar = result['smart_bstar'].clip(upper=upper_clip)
        # note that some of them will be negative,
        # but we are not yet clipping at zero: a very negative value is informative

        # so we first average and then clip at zero, like this:
        all_attempts_to_estimate_bstar['mean'] = smart_bstar.mean(1).clip(lower=10 ** (-10))
        all_attempts_to_estimate_bstar['median'] = smart_bstar.median(axis=1).clip(lower=10 ** (-10))
        all_attempts_to_estimate_bstar['max-z'] = smart_bstar.iloc[:, -1].clip(lower=10 ** (-10))
        return all_attempts_to_estimate_bstar

    @staticmethod
    def ce_and_cl_from_paper_based_on_eigenvalues(eigenvalues, bstar, residual_bstar, z_, c_):
        """
        # ce / cl = optimal weight; ce^2 / cl = monotone transformation of sharpe ratio
        # say, we can select strategies with the highest ce^2 / cl; or we can use the optimal weight
        Parameters
        ----------
        eigenvalues :
        bstar :
        residual_bstar :
        z_ :
        c_ :

        Returns
        -------

        """
        psi_star, m, m_prime, xi, xi_prime = RandomFeatures.m_and_xi_functions(eigenvalues, z_, c_)

        nu, nu_hat, nu_prime = RandomFeatures.all_the_different_nu_functions(psi_star, xi, xi_prime, z_, c_)

        ce_from_paper, cl_from_paper = \
            RandomFeatures.cl_and_ce_from_paper(bstar, residual_bstar, c_, nu, nu_hat, nu_prime)
        ###################################################################################################################
        return ce_from_paper, cl_from_paper

    @staticmethod
    def m_and_xi_functions(eigenvalues, z_, c_):
        eigenvalues = eigenvalues.reshape(1, -1)
        z_ = np.array(z_).reshape(-1, 1).clip(0, 10 ** 10)
        ###################################################################################################################
        # THE NEXT BLOCK IS THE MAGIC OF RANDOM MATRIX THEORY (IGNORING CORRELATIONS ACROSS GROUPS)
        tmp = (1 / (eigenvalues + z_))  # we sum over eigenvalues
        m = tmp.mean(1).reshape(-1, 1)  # \tr (z+
        m_prime = (tmp ** 2).mean(1).reshape(-1, 1)
        # xi_alt = -1 + (1 / c_) / ((1 / c_) - 1 + z_ * m) #
        # since z_ * m is increasing in z, xi = xi_alt is decreasing in z_
        xi = ((1 - z_ * m) / ((1 / c_) - 1 + z_ * m)).clip(0, 10 ** 10)
        xi_prime = (- (1 / c_) * (m - z_ * m_prime) / (((1 / c_) - 1 + z_ * m) ** 2)).clip(- 10 ** 10, 0)
        psi_star = eigenvalues.mean(1)
        return psi_star, m, m_prime, xi, xi_prime

    @staticmethod
    def all_the_different_nu_functions(psi_star, xi, xi_prime, z_, c_):
        """
        from theoretical results, we also know that nu, nu_hat \in [0, \psi_star]
        Parameters
        ----------
        psi_star :
        xi :
        xi_prime :
        z_ :
        c_ :

        Returns
        -------

        """
        z_ = np.array(z_).reshape(1, -1)
        nu = (psi_star - (1 / c_) * z_ * xi).clip(0, psi_star)
        nu_prime = - (1 / c_) * (xi + z_ * xi_prime).clip(0, 10 ** 10)
        nu_hat = (nu + z_ * nu_prime).clip(0, psi_star)
        return nu, nu_hat, nu_prime

    @staticmethod
    def recover_psi_star_and_xi_from_predictions(predictions, complexity):
        psi_star = predictions['nu'] + (1 / complexity) * np.array(predictions['z_grid']).reshape(1, -1) * predictions[
            'xi']
        # we have used the identity nu = psi_star - (1 / c_) * z_ * xi
        xi_prime = (- predictions['nu_prime'] * complexity - predictions['xi']) \
                   / np.array(predictions['z_grid']).reshape(1, -1)
        # we have used the identity nu_prime = - (1 / c_) * (xi + z_ * xi_prime)
        return psi_star, xi_prime

    @staticmethod
    def cl_and_ce_from_paper(bstar, residual_bstar, c_, nu, nu_hat, nu_prime):
        ce_from_paper = bstar * nu
        cl_from_paper = bstar * nu_hat - c_ * (1 + residual_bstar) * nu_prime
        return ce_from_paper, cl_from_paper

    @staticmethod
    def generate_a_random_covariance_matrix_with_given_eigenvalues(eigenvalues: np.ndarray, seed=0) -> np.ndarray:
        """

        Parameters
        ----------
        eigenvalues :

        Returns
        -------
        a random matrix with prescribed eigenvalues
        """
        eigenvalues = eigenvalues.flatten()
        size = len(eigenvalues)
        random_shocks = np.random.normal(size=[size, 2 * size])
        sigma = random_shocks @ random_shocks.T
        _, eigenvectors = np.linalg.eigh(sigma)
        # t1 = time.time()
        # eigenvectors = ortho_group.rvs(size, random_state=seed)
        # matrix = eigenvectors @ (np.diag(eigenvalues) @ eigenvectors.T)
        # t2 = time.time()
        # print(f'got the random O matrix in {t2 - t1}')
        matrix = eigenvectors @ (np.diag(eigenvalues) @ eigenvectors.T)
        return matrix

    @staticmethod
    def produce_betas(multiplied,
                      eigvec,
                      eigval,
                      shrinkage_list):
        normalized = np.concatenate([(1 / (eigval + z)).reshape(-1, 1) * multiplied for z in shrinkage_list], axis=1)
        # here it is subtle as the dimension of eigvec might be lower than that of beta !!!
        # but normalized has the right dimension !!
        betas = np.matmul(eigvec, normalized).T
        return betas

    @staticmethod
    def produce_inverses_signals_mult_mu(signals_mu,
                                         eigvec,
                                         eigval,
                                         shrinkage_list):

        normalized = np.concatenate(
            [np.matmul((1 / (eigval + z)).reshape(-1, 1) * eigvec.T, signals_mu) for z in shrinkage_list], axis=1)
        # here it is subtle as the dimension of eigvec might be lower than that of beta !!!
        # but normalized has the right dimension !!
        inverses_signals_mu = np.matmul(eigvec, normalized)
        # return inverses ## dimension = (windows) * (num of shrinkage*windows)
        return inverses_signals_mu

    @staticmethod
    def produce_gamma_q_xis_and_kappas(b,
                                       signals_mu: float,
                                       inverses_signal_mu,
                                       labels_mu,
                                       betas,
                                       shrinkage_list: list):
        '''
        This function produces necessary quantities for prediction construction in changing kappa test.

        Parameters
        ----------
        b: the constant signal
        signals_mu: Time average of signals
        inverses_signal_mu: inverse of covariance matrix multiplied by
        labels_mu: mean of the return
        betas: betas of signals if there is no shrinkage
        shrinkage_list: list of shrinkages

        Returns
        -------
        GAMMA, Q, xi1, xi2, kappa2, kappa3

        '''
        GAMMA = np.matmul(signals_mu.T, inverses_signal_mu).T  # dim = (# of z) * 1
        Q = np.matmul(betas, signals_mu) / labels_mu  # dim = (# of z) * 1
        xi1 = np.concatenate([np.array([1 / (b * b + z)]) for z in shrinkage_list]).reshape(-1,
                                                                                            1)  # dim = (# of z) * 1
        xi2 = np.concatenate([np.array([b * b / (b * b + z)]) for z in shrinkage_list]).reshape(-1,
                                                                                                1)  # dim = (# of z) * 1
        kappa2 = 1 / (1 - xi2 * GAMMA) * (-xi1 * b * b + xi2 * Q)  # dim = (# of z) * 1
        kappa3 = 1 / (1 - xi2 * GAMMA) * b * b * xi1 * (1 - Q)  # dim = (# of z) * 1
        return GAMMA, Q, xi1, xi2, kappa2, kappa3

    @staticmethod
    def get_eigen_decomposition_of_covariance_in_smart_way(signals,
                                                           labels,
                                                           use_msrr,
                                                           test_fixed_kappa=False,
                                                           fixed_kappas=[]):
        """
        This function even allows us to run multiple regressions at the same time
        when labels have more than one column
        Parameters
        ----------
        signals :
        labels :
        use_msrr :
        fixed_kappas: list of fixed kappas, specified in fixed kappa test

        Returns
        -------

        """
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        if labels.shape[1] > 1:
            labels_mu = labels.mean(0).reshape(-1, 1)
            mu = np.concatenate([(signals * labels[:, i].reshape(-1, 1)).mean(0).reshape(-1, 1)
                                 for i in range(labels.shape[1])], axis=1)
            if test_fixed_kappa:
                mu = np.concatenate(
                    [(signals * labels[:, i].reshape(-1, 1) - labels_mu * fixed_kappas[1]).mean(0).reshape(-1, 1)
                     for i in range(labels.shape[1])], axis=1)  # kappa_2
            managed_returns = None
        else:
            labels_mu = labels.mean()
            managed_returns = signals * labels.reshape(-1, 1)
            if test_fixed_kappa:
                labels_mu = labels.mean()
                managed_returns = signals * (labels.reshape(-1, 1) - fixed_kappas[1] * labels_mu)
            mu = managed_returns.mean(0).reshape(-1, 1)
        # this is the expected return of the timing strategy for each column on labels

        data_for_covariance = managed_returns if use_msrr else signals
        large_covariance = signals.shape[1] > signals.shape[0]
        signals_mu = signals.mean(0).reshape(-1, 1)

        # if test fixed kappa, we need to shrink the RHS data referring to kappas
        if test_fixed_kappa:
            data_for_covariance_shrinkage = data_for_covariance.mean(0).reshape(-1, 1).T * fixed_kappas[
                4]  # kappa 5, will use this to shrink future signals as well
            data_for_covariance = data_for_covariance - data_for_covariance_shrinkage
            data_for_covariance_mean = data_for_covariance.mean(0).reshape(-1, 1).T

            if large_covariance:
                data_for_covariance = data_for_covariance.T
                data_for_covariance_mean = data_for_covariance_mean.T

            ### KAPPA TIME! kappa 1
            covariance = np.matmul((data_for_covariance - data_for_covariance_mean * fixed_kappas[0]).T,
                                   data_for_covariance - data_for_covariance_mean * fixed_kappas[0]) / signals.shape[0]
            # signals.shape[0] is the number of observations
            eigval, eigvec1 = np.linalg.eigh(covariance)
            eigvec1 = eigvec1[:, eigval > 10 ** (-10)]
            eigval = eigval[eigval > 10 ** (-10)]
            if len(eigval) == 0:
                breakpoint()
            if not large_covariance:
                data_for_covariance_mean = data_for_covariance_mean.T
            return large_covariance, eigvec1, eigval, data_for_covariance, data_for_covariance_mean.T, data_for_covariance_shrinkage, mu, labels_mu

        else:
            if large_covariance:
                data_for_covariance = data_for_covariance.T

            covariance = np.matmul(data_for_covariance.T, data_for_covariance) / signals.shape[0]
            # signals.shape[0] is the number of observations
            eigval, eigvec1 = np.linalg.eigh(covariance)

            # now we filter away low eigenvalues.
            # Why is this legal?
            # So we are using (zI + X'X)^{-1} X'y
            # we have the polar decomposition X'=HV
            # (we do not use it in the analysis, so this is just for mathematical explanations)
            # Then, X'X= HVV'H=H^2.
            # So we are doing (zI+H^2)^{-1}H Vy
            # Then, the key observation is that if H has a kernel we can ignore its action on the kernel
            # Namely, H^2 = U D U' and (zI+H^2)^{-1}H = U (z+D)^{-1} D U'
            # and we see that whatever happens on where D=0 gets annihilated

            eigvec1 = eigvec1[:, eigval > 10 ** (-10)]
            eigval = eigval[eigval > 10 ** (-10)]
            if len(eigval) == 0:
                breakpoint()
            return large_covariance, eigvec1, eigval, data_for_covariance, mu, labels_mu, signals_mu

    @staticmethod
    def populate_output(norm_of_beta,
                        predictions,
                        betas,
                        signals,
                        return_in_sample_pred,
                        return_beta,
                        compute_smart_weights,
                        all_the_names,
                        beta_names,
                        all_the_bstar,
                        ce_from_paper,
                        cl_from_paper,
                        shrinkage_list,
                        future_signals):

        output = {'norm_of_beta': norm_of_beta,
                  'predictions': predictions}
        if return_in_sample_pred:
            output['in_sample_predictions'] = np.matmul(betas, signals.T).T
        if return_beta:
            output['betas'] = betas
        if compute_smart_weights:
            output.update({'names_for_bstar_estimation': all_the_names,
                           'beta_names': beta_names,
                           'smart_bstar': np.array(all_the_bstar).reshape(1, -1),
                           'ce_from_paper': np.array(ce_from_paper).reshape(1, -1),
                           'cl_from_paper': np.array(cl_from_paper).reshape(1, -1),
                           'original_z_grid': shrinkage_list})
            # now we tile them for creating time series
            # future_signals.shape[0] = stepp; needed for computing speed up
            for key in ['smart_bstar', 'ce_from_paper', 'cl_from_paper']:
                output[key] = np.tile(output[key], [future_signals.shape[0], 1])
        return output

    @staticmethod
    def compute_smart_betas_with_rmt(in_sample_means,
                                     norm_of_beta,
                                     sigma_hat,
                                     shrinkage_list,
                                     signals,
                                     eigval,
                                     labels,
                                     large_covariance,
                                     core_z_values,
                                     multiplied,
                                     eigvec,
                                     betas,
                                     clip_bstar=100
                                     ):
        complexity = signals.shape[1] / signals.shape[0]
        if large_covariance:
            true_eigenvalues = np.concatenate(
                [np.zeros([1, signals.shape[1] - signals.shape[0]]), eigval.reshape(1, -1)], axis=1)
        else:
            true_eigenvalues = eigval
        naive_bstar = RandomFeatures.naive_in_sample_bstar_estimate(signals, labels).mean()
        if np.isnan(naive_bstar):
            naive_bstar = 10 ** (-10)
        """
        the assumption is that each beta \sim N(0, b_*/P). Hence, E[\sum_i beta_i^2] = b_* 
        """
        psi_star, m, m_prime, xi, xi_prime = RandomFeatures.m_and_xi_functions(true_eigenvalues,
                                                                               z_=shrinkage_list,
                                                                               c_=complexity)
        # nu and nu_hat are inputs into RandomFeatures.cl_and_ce_from_paper()
        # ce and cl are super useful
        # ce / cl = optimal weight; ce^2 / cl = monotone transformation of sharpe ratio
        # say, we can select strategies with the highest ce^2 / cl; or we can use the optimal weight
        smart_bstar = RandomFeatures.smart_in_sample_bstar_estimate_based_on_bnorm(
            norm_of_beta.reshape(-1, 1),
            z_=shrinkage_list,
            c_=complexity,
            m=m,
            m_prime=m_prime,
            xi=xi,
            xi_prime=xi_prime,
            residual_bstar=0,
            core_z_values=core_z_values)

        smart_bstar_based_on_mean \
            = RandomFeatures.smart_in_sample_bstar_estimate_based_on_mean(in_sample_means,
                                                                          z_=shrinkage_list,
                                                                          c_=complexity,
                                                                          xi=xi,
                                                                          sigma_hat=sigma_hat)
        if core_z_values is not None:
            fitted_bstar_estimate = smart_bstar[1]
            fitted_residual_bstar_estimate = smart_bstar[2]
            smart_bstar = smart_bstar[0]

        smart_bstar, names = RandomFeatures.produce_all_sorts_of_bstar_estimates(smart_bstar,
                                                                                 upper_clip=clip_bstar)
        # I used to clip at 10, but it was too small

        all_the_bstar = [naive_bstar, naive_bstar * signals.shape[1]] + smart_bstar \
                        + [np.mean(smart_bstar_based_on_mean), np.median(smart_bstar_based_on_mean)]
        residual_bstars = [0] * len(all_the_bstar)

        all_the_bstar += [fitted_bstar_estimate]
        residual_bstars += [fitted_residual_bstar_estimate]

        # optimal shrinkage given bstar
        optimal_z = complexity * (1 + np.array(residual_bstars)) / np.array(all_the_bstar)
        betas_with_optimal_z = RandomFeatures.produce_betas(multiplied,
                                                            eigvec,
                                                            eigval,
                                                            optimal_z)

        _, _, _, xi_with_optimal_z, xi_prime_with_optimal_z \
            = RandomFeatures.m_and_xi_functions(true_eigenvalues,
                                                z_=optimal_z,
                                                c_=complexity)
        # now we build stuff merging initial z grid with optimal z candidates based on bstar estimates
        xi = np.concatenate([xi, xi_with_optimal_z], axis=0)
        xi_prime = np.concatenate([xi_prime, xi_prime_with_optimal_z])
        betas = np.concatenate([betas, betas_with_optimal_z], axis=0)
        ###############################################################################################

        # now that we got the betas with optimal z, we can build their optimal combinations
        # we do this in a tricky way, allowing to combine both the original grid of z
        # and also candidate optimal values of z: z_=shrinkage_list + list(optimal_z)
        everything = [RandomFeatures.optimal_weight_vector_across_ridge_parameters(
            all_the_bstar[i],
            residual_bstar=residual_bstars[i],
            psi_star=psi_star,
            xi=xi.reshape(1, -1),
            xi_prime=xi_prime.reshape(1, -1),
            z_=shrinkage_list + list(optimal_z),
            c_=complexity) for i in range(len(all_the_bstar))]

        optimal_weights = np.concatenate([everything[i][0]
                                          for i in range(len(all_the_bstar))], axis=1)
        # now, for each bstar, we have a commitment to an optimal z;
        # hence for that choice of bstar, we should use that value of z and nothing else
        # indeed, either we take that value seriously or not
        ce_from_paper = np.array([everything[i][1][0, len(shrinkage_list) + i]
                                  for i in range(len(all_the_bstar))]).T

        cl_from_paper = np.array([everything[i][2][0, len(shrinkage_list) + i]
                                  for i in range(len(all_the_bstar))]).T
        # so now we have produced ce and cl computed always at (bstar_estimate, optimal_z(bstar_estimate))

        # expected return on the efficient portfolio of ridge is:
        ce_portfolio = np.array([everything[i][3] for i in range(len(all_the_bstar))])
        ce_from_paper = np.concatenate([ce_from_paper.reshape(1, -1), ce_portfolio.reshape(1, -1)], axis=1)

        # expected second moment of the efficient portfolio of ridge is:
        cl_portfolio = np.array([everything[i][4] for i in range(len(all_the_bstar))])
        cl_from_paper = np.concatenate([cl_from_paper.reshape(1, -1), cl_portfolio.reshape(1, -1)], axis=1)

        optimal_betas = (betas.T @ optimal_weights).T  # this is the optimal combination of betas
        betas = np.concatenate([betas, optimal_betas], axis=0)
        return betas, all_the_bstar, ce_from_paper, cl_from_paper

    @staticmethod
    def build_the_q_vector(psi_matrix: np.ndarray,
                           labels: np.ndarray,
                           shrinkage_list: list,
                           number_random_features: int,
                           normalize_p: bool = False):
        sample_size = psi_matrix.shape[0]
        covariance = psi_matrix / (sample_size * (number_random_features if normalize_p else 1))  # this is SS'/(P * T)

        # this is T \times T
        # signals.shape[0] is the number of observations
        eigval, eigvec1 = np.linalg.eigh(covariance)

        # now we filter away low eigenvalues.
        # Why is this legal?
        # So we are using (zI + X'X)^{-1} X'y
        # we have the polar decomposition X'=HV
        # (we do not use it in the analysis, so this is just for mathematical explanations)
        # Then, X'X= HVV'H=H^2.
        # So we are doing (zI+H^2)^{-1}H Vy
        # Then, the key observation is that if H has a kernel we can ignore its action on the kernel
        # Namely, H^2 = U D U' and (zI+H^2)^{-1}H = U (z+D)^{-1} D U'
        # and we see that whatever happens on where D=0 gets annihilated

        eigvec1 = eigvec1[:, eigval > 10 ** (-10)]
        eigval = eigval[eigval > 10 ** (-10)]
        # now eigvec1 is a bit smaller, T \times T1 for some T1<T

        multiplied = (1 / eigval).reshape(-1, 1) * (eigvec1.T @ (covariance @ labels))
        # this vector is now T1 \times 1

        normalized = np.concatenate([(1 / (eigval + z)).reshape(-1, 1) * multiplied for z in shrinkage_list], axis=1)
        # this matrix should now be T1 \times len(shrinkage)

        # here it is subtle as the dimension of eigvec might be lower than that of beta !!!
        # but normalized has the right dimension !!
        q_vector = eigvec1 @ normalized
        # this is (T \times T1) \times (T1 \times len(shrinkage))  which should give T \times len(shrinkage)

        if len(eigval.tolist()) < number_random_features:
            # the true psi matrix is P \times P/ If P>T, then it will have zeros
            eigval = np.array(eigval.tolist() + [0] * (number_random_features - len(eigval.tolist())))
        else:
            # otherwise the first number_random_features - len(psi_hat_eig.tolist()) eigenvalues are identically zero
            eigval = eigval[(number_random_features - len(eigval.tolist())):]

        return q_vector, eigval

    @staticmethod
    def update_sigma_matr(sigma_matr: np.ndarray,
                          random_features: np.ndarray,
                          date_ids: np.ndarray,
                          stock_ids: np.ndarray
                          ):
        """
        The function assumes that all stocks are available each date
        important!! the code assumes stock_ids are ordered conditional on a given date !! Make sure it is true!

        Parameters
        ----------
        sigma_matr :
        random_features :
        date_ids :
        stock_ids :

        Returns
        -------

        """
        all_stocks = np.unique(stock_ids)
        if len(all_stocks) * len(np.unique(date_ids)) > len(stock_ids):
            raise Exception(f'!!!!!!!!!!!!!!!! stock panel is imbalanced !!!!!!!!!!!!!!')
        # important!! the code assumes stock_ids are ordered conditional on a given date !! Make sure it is true!
        for date in np.unique(date_ids):
            indicator = date_ids.flatten() == date
            features_per_date = random_features[indicator, :]  # first select date, then stocks in that date
            # the code assumes stock_ids are ordered conditional on a given date
            sigma_matr = sigma_matr + features_per_date @ features_per_date.T
        return sigma_matr

    @staticmethod
    def compute_psi_and_sigma_matrix(seed: int,
                                     number_random_features: int,
                                     small_subset_size: int,
                                     sample_size: int,
                                     stock_ids: np.ndarray,
                                     date_ids: np.ndarray,
                                     raw_signals: np.ndarray,
                                     labels: np.ndarray,
                                     random_features_dict_parameters: dict,
                                     shrinkage_list: list,
                                     ranking: str,
                                     voc_grid: list,
                                     do_ranking_before_full_panel: bool,
                                     test_mode: bool = False,
                                     produce_voc_curve: bool = False,
                                     normalize_p: bool = False):

        np.random.seed(seed)
        block_sizes = np.arange(0, number_random_features, small_subset_size).astype(int).tolist()
        if number_random_features not in block_sizes:
            block_sizes += [number_random_features]

        # if grid point in voc_grid is not in block_sizes, we add them in the block_sizes
        block_sizes = list(set(block_sizes + voc_grid))
        block_sizes.sort()  # sort grid points

        # if step_for_voc_curve > 1:
        #     voc_block_sizes = block_sizes[::step_for_voc_curve] + block_sizes[:step_for_voc_curve]
        #     voc_block_sizes = np.unique(voc_block_sizes).astype(int).tolist()
        #     if number_random_features not in voc_block_sizes:
        #         voc_block_sizes += [number_random_features]
        if do_ranking_before_full_panel:
            # this is just to get the dimensions to construct psi_matrix and sigma_matrix
            raw_signals_post_select, labels_post_select, date_ids_post_select, stock_ids_post_select = \
                select_balanced_panel(raw_signals=raw_signals,
                                      labels=labels,
                                      date_ids=date_ids,
                                      stock_ids=stock_ids)
            sample_size = raw_signals_post_select.shape[0]
            psi_matrix = np.zeros([sample_size, sample_size])
            num_stocks = len(np.unique(stock_ids_post_select))
            number_dates = len(np.unique(date_ids_post_select))
            sigma_matr = np.zeros([num_stocks, num_stocks])
        else:
            psi_matrix = np.zeros([sample_size, sample_size])

            num_stocks = len(np.unique(stock_ids))
            number_dates = len(np.unique(date_ids))
            sigma_matr = np.zeros([num_stocks, num_stocks])

        if produce_voc_curve:
            psi_eigenvalues_for_expanding_complexity = dict()
            q_vectors_for_expanding_complexity = dict()
            sigma_hat_eigenvalues_for_expanding_complexity = dict()

        random_features_all = []
        k = 0
        for block in range(len(block_sizes) - 1):
            k += 1
            # now we loop through blocks of features
            number_features_in_subset = block_sizes[block + 1] - block_sizes[block]
            print(number_features_in_subset)

            # random_weights = np.random.randn(number_raw_signals, number_features_in_subset)

            rft = RandomFeaturesTesting(raw_signals=pd.DataFrame(raw_signals),
                                        returns=pd.DataFrame(np.zeros(shape=(raw_signals.shape[0], 1))),
                                        asset_class='JKP',
                                        balanced=False,
                                        settings=BasicRandomFeatureSettingForJKP())
            spec = random_features_dict_parameters.copy()
            spec.update({'number_features': number_features_in_subset})
            # spec['number_features'] = number_features_in_subset
            rft.generate_random_features_from_list([(int((seed + 1) * 1e3) + k, spec)])
            print('spec', (int((seed + 1) * 1e3) + k, spec))
            random_features = rft.random_features[list(rft.random_features.keys())[0]].values
            print('random_features size', number_features_in_subset, random_features.shape, flush=True)

            # random_features.shape
            # breakpoint()

            # print('HHHERE', np.abs(np.max((random_features-pd.read_pickle('temp_old.p').values))))

            if ranking is not None:
                # pd.DataFrame(random_features).to_pickle('t_rf_lag.p')
                random_features = rank_features_cross_sectionally(random_features, date_ids, ranking)
                # pd.DataFrame(random_features).to_pickle('t_rf.p')
                # pd.DataFrame(date_ids).to_pickle('t_dates.p')
                # pd.DataFrame([ranking]).to_pickle('t_ranking.p')

                # print('THHHERE', np.abs(np.max((random_features-pd.read_pickle('temp_old_rank.p').values))))
            if do_ranking_before_full_panel:
                print('We post-select stocks with full observations after generating random features and ranking')
                random_features, _, _, _ = select_balanced_panel(raw_signals=random_features,
                                                                 labels=labels,
                                                                 date_ids=date_ids,
                                                                 stock_ids=stock_ids)

            if test_mode:
                random_features_all.append(random_features)

            psi_matrix += random_features @ random_features.T

            if do_ranking_before_full_panel:
                sigma_matr = RandomFeatures.update_sigma_matr(sigma_matr,
                                                              random_features,
                                                              date_ids_post_select,
                                                              stock_ids_post_select)
            else:
                sigma_matr = RandomFeatures.update_sigma_matr(sigma_matr,
                                                              random_features,
                                                              date_ids,
                                                              stock_ids)

            if produce_voc_curve and (block_sizes[block + 1] in voc_grid):
                # so now we are running the regression on the intermediate result with a subset of random features
                if do_ranking_before_full_panel:
                    q_vector, psi_hat_eig = RandomFeatures.build_the_q_vector(
                        psi_matrix,
                        labels_post_select,
                        shrinkage_list,
                        number_random_features=block_sizes[block + 1],
                        normalize_p=normalize_p)
                else:
                    q_vector, psi_hat_eig = RandomFeatures.build_the_q_vector(
                        psi_matrix,
                        labels,
                        shrinkage_list,
                        number_random_features=block_sizes[block + 1],
                        normalize_p=normalize_p)

                sigma_hat = sigma_matr / (block_sizes[block + 1] * number_dates)
                # here we are dividing by the actual number of used random features
                sigma_hat_eig_, _ = np.linalg.eigh(sigma_hat)

                q_vectors_for_expanding_complexity.update({block_sizes[block + 1]: q_vector})
                psi_eigenvalues_for_expanding_complexity.update({block_sizes[block + 1]: psi_hat_eig})
                sigma_hat_eigenvalues_for_expanding_complexity.update({block_sizes[block + 1]: sigma_hat_eig_})

        sigma_hat = sigma_matr / (number_random_features * number_dates)
        if not produce_voc_curve:
            sigma_hat_eig_, _ = np.linalg.eigh(sigma_hat)
        else:
            sigma_hat_eig_ = sigma_hat_eigenvalues_for_expanding_complexity[number_random_features]

        if test_mode:
            random_features = np.concatenate(random_features_all, axis=1)
            true_psi_matr = random_features.T @ random_features
        else:
            true_psi_matr = None
            random_features = None
        if produce_voc_curve:
            voc_curve = {'psi_eig': psi_eigenvalues_for_expanding_complexity,
                         'q_vectors': q_vectors_for_expanding_complexity,
                         'sigma_eig': sigma_hat_eigenvalues_for_expanding_complexity}
        else:
            voc_curve = dict()
        return psi_matrix, sigma_hat_eig_, true_psi_matr, random_features, voc_curve

    @staticmethod
    def compute_betas_and_predictions(seed: int,
                                      shrinkage_list: list,
                                      future_raw_signals: np.ndarray,
                                      number_random_features: int,
                                      small_subset_size: int,
                                      raw_signals: np.ndarray,
                                      date_ids: np.ndarray,
                                      stock_ids: np.ndarray,
                                      ranking: str,
                                      random_features_dict_parameters: dict,
                                      future_date_ids: np.ndarray,
                                      labels: np.ndarray,
                                      number_dates: int,
                                      voc_grid: list,
                                      do_ranking_before_full_panel: bool,
                                      voc_curve: dict = {},
                                      produce_betas: bool = False,
                                      test: bool = False,
                                      normalize_p: bool = True) -> tuple:
        labels_pre_select = labels
        number_raw_signals = raw_signals.shape[1]
        # here it is very impotant that we re-build the same seeds !!!!
        np.random.seed(seed)  # hence we re-initiate the seed
        # I am afraid to single out the next loop into a function so that the seed is not lost
        # first we initialize the output with empty lists and zeros
        betas = {key: [] for key in voc_curve['q_vectors'].keys()}
        realized_in_sample_returns \
            = {key: np.zeros([1, len(shrinkage_list)]) for key in voc_curve['q_vectors'].keys()}
        future_predictions = {key: np.zeros([future_raw_signals.shape[0], len(shrinkage_list)])
                              for key in voc_curve['q_vectors'].keys()}

        block_sizes = np.arange(0, number_random_features, small_subset_size).astype(int).tolist()
        if number_random_features not in block_sizes:
            block_sizes += [number_random_features]
        # if grid point in voc_grid is not in block_sizes, we add them in the block_sizes
        block_sizes = list(set(block_sizes + voc_grid))
        block_sizes.sort()  # sort grid points

        future_random_features_all = list()
        k = 0
        for block in range(len(block_sizes) - 1):
            k += 1
            # now we loop through blocks of features
            number_features_in_subset = block_sizes[block + 1] - block_sizes[block]
            # random_weights = np.random.randn(number_raw_signals, number_features_in_subset)
            spec = random_features_dict_parameters.copy()
            spec['number_features'] = number_features_in_subset
            # spec.update({'number_features':number_features_in_subset})
            spec = (int((seed + 1) * 1e3) + k, spec)

            rft = RandomFeaturesTesting(raw_signals=pd.DataFrame(raw_signals),
                                        returns=pd.DataFrame(np.zeros(shape=(raw_signals.shape[0], 1))),
                                        asset_class='JKP', balanced=False,
                                        settings=BasicRandomFeatureSettingForJKP())
            rft.generate_random_features_from_list([spec])
            random_features = rft.random_features[list(rft.random_features.keys())[0]].values

            if ranking is not None:
                random_features = rank_features_cross_sectionally(random_features,
                                                                  date_ids,
                                                                  ranking)
            if do_ranking_before_full_panel:
                # this is just to get the dimensions to construct psi_matrix and sigma_matrix
                random_features, labels, _, _ = select_balanced_panel(raw_signals=random_features,
                                                                      labels=labels_pre_select,
                                                                      date_ids=date_ids,
                                                                      stock_ids=stock_ids)
            # random_weights = np.random.randn(number_raw_signals, number_features_in_subset)
            # # this is P \times P1
            #
            # random_features = RandomFeatures.random_ranked_features(raw_signals,
            #                                                         random_weights,
            #                                                         date_ids,
            #                                                         ranking)
            # this is supposed to be T \times P1

            rft = RandomFeaturesTesting(raw_signals=pd.DataFrame(future_raw_signals),
                                        returns=pd.DataFrame(np.zeros(shape=(future_raw_signals.shape[0], 1))),
                                        asset_class='JKP', balanced=False,
                                        settings=BasicRandomFeatureSettingForJKP())
            rft.generate_random_features_from_list([spec])
            future_random_features = rft.random_features[list(rft.random_features.keys())[0]].values
            if ranking is not None:
                future_random_features = rank_features_cross_sectionally(future_random_features,
                                                                         future_date_ids,
                                                                         ranking)
            # future_random_features = RandomFeatures.random_ranked_features(future_raw_signals,
            #                                                                random_weights,
            #                                                                future_date_ids,
            #                                                                ranking)
            if test:
                future_random_features_all.append(future_random_features)

            # q_vector is T \times len(shrinkage_list)
            # random_features is T \times P1
            # hence beta_chunk \in \R^{P_1\times len(shrinkage_list)}
            # so the betas for the chunk will only matter for a model with hih enough complexity
            # hence the condition key >= block_sizes[block + 1]
            beta_chunks = {key: random_features.T @ voc_curve['q_vectors'][key]
                                / (random_features.shape[0] * (np.sqrt(key) if normalize_p else 1))
                           for key in voc_curve['q_vectors'] if key >= block_sizes[block + 1]}

            # same here: only stuff with high complexity, if key >= block_sizes[block + 1], gets updated
            realized_in_sample_returns.update(
                {key: realized_in_sample_returns[key]
                      + (beta_chunks[key].T @ random_features.T @ labels / (np.sqrt(key) if normalize_p else 1)).T
                 for key in realized_in_sample_returns if
                 key >= block_sizes[block + 1]})
            future_predictions.update(
                {key: future_predictions[key] +
                      future_random_features @ beta_chunks[key] / (np.sqrt(key) if normalize_p else 1)
                 for key in future_predictions
                 if key >= block_sizes[block + 1]})

            # so the amazing thing is that we do not need to actually store the betas.
            # we update predictions chunk-by-chunk and can forget them
            if produce_betas:
                betas.update({key: betas[key] + [beta_chunks[key]]
                              for key in betas if key >= block_sizes[block + 1]})

        realized_in_sample_returns = {key: realized_in_sample_returns[key]
                                           / number_dates for key in realized_in_sample_returns}
        # here we divide by T, because the estimator of b_star_hat_in_sample
        # is designed to take stuff normalized by T
        if produce_betas:
            betas = {key: np.concatenate(betas[key], axis=0) for key in betas.keys()}

        return betas, realized_in_sample_returns, future_predictions, future_random_features_all

    @staticmethod
    def get_predictions_given_beta_with_giant_number_of_random_features(
            betas_pre_trained: np.ndarray,
            shrinkage_list: list,
            number_random_features: int,
            small_subset_size: int,
            seed: int,
            ranking: str,
            voc_grid: list,
            future_raw_signals: np.ndarray,
            future_date_ids: np.ndarray,
            normalize_p: bool = True,
            random_features_parameters_distribution_weights: str = 'gaussian_mixture',
            random_features_parameters_gamma: [] = None,
            random_features_parameters_activation_function: str = 'cos_and_sin',
            random_features_parameters_distribution_bias: str = None,
            random_features_parameters_distribution_bias_parameters: [] = None
    ) -> np.ndarray:
        """
        This function make a chunk based prediction given a pre-train set of beta.
        You need to have the same hpyerparameters (including seed and activation) as in the first training for this funciton to make sens.
        :param betas_pre_trained: an nd.array containing the betas obtained with ridge_regression_with_giant_number_of_random_features
        :param number_random_features: the number of random features
        :param small_subset_size: the small subset sized used in training
        :param seed: same seed as in training
        :param ranking: 'rank' or 'cdf' or 'none' if it's the same
        :param voc_grid: even if we give back only one prediction here, you need the voc_grid to get the same chunk size!
        :param future_raw_signals:  that doesn't have to be same as the old
        :param future_date_ids: ibidem
        :param normalize_p: this should be same as in training
        :param random_features_parameters_distribution_weights: this and the rest  other random_features_parameters* input should be the same as in triaing,
        unless you are doing something funky
        :param random_features_parameters_gamma:
        :param random_features_parameters_activation_function:
        :param random_features_parameters_distribution_bias:
        :param random_features_parameters_distribution_bias_parameters:
        :return: a T*N by Z matrix of predictions.
        """
        # creating random_features_dict_parameters as a function
        if random_features_parameters_gamma is None:
            random_features_parameters_gamma = [0.1, 0.5, 1, 2, 4, 8, 16]
        if random_features_parameters_distribution_bias_parameters is None:
            random_features_parameters_distribution_bias_parameters = [-3.141592653589793, 3.141592653589793]
        random_features_dict_parameters = {'distribution': random_features_parameters_distribution_weights,
                                           'distribution_parameters': random_features_parameters_gamma,
                                           'activation': random_features_parameters_activation_function,
                                           'number_features': small_subset_size,
                                           'bias_distribution': random_features_parameters_distribution_bias,
                                           'bias_distribution_parameters': random_features_parameters_distribution_bias_parameters}

        # q_vector \in R^{T\times len(shrinkage_list)}
        # but psi_hat_eig have lots of missing zeros. We should add them
        # here it is very impotant that we re-build the same seeds !!!!
        np.random.seed(seed)  # hence we re-initiate the seed
        # I am afraid to single out the next loop into a function so that the seed is not lost
        # first we initialize the output with empty lists and zeros

        future_predictions = np.zeros([future_raw_signals.shape[0], len(shrinkage_list)])
        block_sizes = np.arange(0, number_random_features, small_subset_size).astype(int).tolist()
        if number_random_features not in block_sizes:
            block_sizes += [number_random_features]
        # if grid point in voc_grid is not in block_sizes, we add them in the block_sizes
        block_sizes = list(set(block_sizes + voc_grid))
        block_sizes.sort()  # sort grid points

        future_random_features_all = list()
        k = 0
        cumulated_block_size_done = 0
        for block in range(len(block_sizes) - 1):
            k += 1
            # now we loop through blocks of features
            number_features_in_subset = block_sizes[block + 1] - block_sizes[block]
            # random_weights = np.random.randn(number_raw_signals, number_features_in_subset)
            spec = random_features_dict_parameters.copy()
            spec['number_features'] = number_features_in_subset
            # spec.update({'number_features':number_features_in_subset})
            spec = (int((seed + 1) * 1e3) + k, spec)

            rft = RandomFeaturesTesting(raw_signals=pd.DataFrame(future_raw_signals),
                                        returns=pd.DataFrame(np.zeros(shape=(future_raw_signals.shape[0], 1))),
                                        asset_class='JKP', balanced=False,
                                        settings=BasicRandomFeatureSettingForJKP())
            rft.generate_random_features_from_list([spec])
            future_random_features = rft.random_features[list(rft.random_features.keys())[0]].values

            if ranking is not None:
                future_random_features = rank_features_cross_sectionally(future_random_features,
                                                                         future_date_ids,
                                                                         ranking)
            # q_vector is T \times len(shrinkage_list)
            # random_features is T \times P1
            # hence beta_chunk \in \R^{P_1\times len(shrinkage_list)}
            # so the betas for the chunk will only matter for a model with hih enough complexity
            # hence the condition key >= block_sizes[block + 1]
            # beta_chunks = {key: random_features.T @ voc_curve['q_vectors'][key]
            #                     / (random_features.shape[0] * (np.sqrt(key) if normalize_p else 1))
            #                for key in voc_curve['q_vectors'] if key >= block_sizes[block + 1]}

            # print('here')
            # breakpoint()
            beta_chunks = betas_pre_trained[
                          cumulated_block_size_done:(cumulated_block_size_done + number_features_in_subset), :]
            cumulated_block_size_done += number_features_in_subset
            # if normalize_p:
            #     beta_chunks *= (np.sqrt(number_random_features)/np.sqrt(cumulated_block_size_done))

            # same here: only stuff with high complexity, if key >= block_sizes[block + 1], gets updated

            future_predictions += future_random_features @ beta_chunks / (
                np.sqrt(number_random_features) if normalize_p else 1)

        return future_predictions

    @staticmethod
    def ridge_regression_with_giant_number_of_random_features(raw_signals: np.ndarray,
                                                              labels: np.ndarray,
                                                              shrinkage_list: list,
                                                              stock_ids: np.array,
                                                              date_ids: np.ndarray,
                                                              number_random_features: int,
                                                              small_subset_size: int,
                                                              seed: int,
                                                              ranking: str,
                                                              future_raw_signals: np.ndarray,
                                                              future_date_ids: np.ndarray,
                                                              voc_grid: list,
                                                              do_ranking_before_full_panel: bool = False,
                                                              test_mode: bool = False,
                                                              produce_voc_curve: bool = False,
                                                              produce_betas: bool = False,
                                                              run_linear_model: bool = True,
                                                              normalize_p: bool = True,
                                                              random_features_parameters_distribution_weights: str = 'gaussian_mixture',
                                                              random_features_parameters_gamma: [] = None,
                                                              random_features_parameters_activation_function: str = 'cos_and_sin',
                                                              random_features_parameters_distribution_bias: str = None,
                                                              random_features_parameters_distribution_bias_parameters: [] = None,
                                                              save_sigma_and_psi_eig_in_rmt_stuff: bool = True,
                                                              random_features_parameters_use_bining_random_features: bool = False,
                                                              random_features_parameters_bining_distribution: str = 'normal',
                                                              random_features_parameters_bining_random_rotation: bool = False,
                                                              ):
        """
        Important: the code assumes that stock ids are already sorted!
        Pre-process the data so that stock ids are increasing !!
        so the original data must first be sorted on dates. Then, conditional on any date we sort stock ids.
        And this pre-sorted data comes into the function

        Same for date_ids: We are appending data!
        So we assume that

        Parameters
        ----------
        voc_grid: grid for producing VOC curve. Must be multiples of small_subset_size
        produce_betas : If True, then we also output the giant beta vector.
        It could be huge (size = number_random_features, which could be a million or so)
        produce_voc_curve : If True, then we actually output predictions for a giant grid of numbers of random features
        (with a step size of roughly number_random_features / small_subset_size)
        test_mode : If True, then we run a test to see if the output coincides with a simple, naive linear ridge
        future_date_ids : dates for the chunk of out-of-sample (test) data on which we produce OOS predictions
        future_raw_signals : the chunk of out-of-sample (test) data on which we produce OOS predictions
        date_ids : ids of dates. Must be ordered !!!
        shrinkage_list : list of ridge shrinkage basic_parameters
        raw_signals : in-sample raw signals from which random features are constructed
        labels : in-sample returns to be predicted
        stock_ids : stock ids: We need them to build the sigma-matrix . Must be ordered !!!
        number_random_features : how many random features we want to produce. Could be a very large number
        small_subset_size : we split random features into sub-groups so that they fit in memory and
        running it becomes feasible even on a small machine
        seed : random seed. One should run this for a fixed seed, and then average predictions across seeds
        ranking : once we have produced random linear combinations of features, we take a non-linear transformation
        of them. We could either cross-sectionally rank them (corresponds to ranking='rank'),
        or we apply a sigmoid (corresponds to ranking ='cdf')

        Returns dictionary
        output = {'rmt_stuff': rmt_stuff,
                  'betas': betas,
                  'future_predictions': future_predictions}

        -------
        rmt_stuff: random matrix theory stuff that can be used to compute the optimal shrinkage parameter z_*
        'betas': actual regression betas (for each shrinkage level)
        'future_predictions': Actual predictions for each (date-stock)
        Each of these is itself a dictionary, indexed by a grid of "numbers of random features"
        If produce_voc_curve = True, then this is an actual grid.
        If produce_voc_curve = False, then this is just one point, = number_random_features
        For each number_of_features, the corresponding future_predictions[number_of_features]
        is a matrix, dimension (OOS sample size) \times (shrinkage_list), so that for each value of
        the shrinkage parameter we have one prediction.

        Similarly for the rmt_stuff and betas.

        Why would the OOS sample size be big? Well, we do not need to re-compute betas every period.
        It is enough to do it every few month (say, every 3 months), in which case OOS sample size =
        number_of_stocks \times 3
        :param save_sigma_and_psi_eig_in_rmt_stuff: if true, we add these to the list of saved rmt_stuff
        :param random_features_parameters_bining_random_rotation: when random_features_parameters_use_bining_random_features = true, we use this to define the spec
        :param random_features_parameters_bining_distribution: the distribution of the spec when we use random bining
        :param random_features_parameters_use_bining_random_features: when true, we use the Binning random features
        :param random_features_parameters_distribution_bias_parameters: with default bias distribution,
        this defined the minimum and maximum values of the biases in the random featueres. Default: [-3.141592653589793, 3.141592653589793]
        :param random_features_parameters_distribution_bias: define the disitrubtion of the bias in the random features (default unifom)
        :param random_features_parameters_activation_function: define the acitaviton function of the random features
        :param random_features_parameters_gamma: define the distribution paramters of the random features, IF WE USE BINING IT BECOMES THE GAMMA AND WE SELECT ONLY THE FIRST ONE IN THE LIST
        :param random_features_parameters_distribution_weights: defines distribution of the weights of the random features
        :param random_features_dict_parameters: define the paramters of the distrubution of the random features' weights

        """
        # creating random_features_dict_parameters as a function
        if random_features_parameters_gamma is None:
            random_features_parameters_gamma = [0.1, 0.5, 1, 2, 4, 8, 16]
        if random_features_parameters_distribution_bias_parameters is None:
            random_features_parameters_distribution_bias_parameters = [-3.141592653589793, 3.141592653589793]
        if random_features_parameters_use_bining_random_features:
            random_features_dict_parameters = {'distribution': random_features_parameters_bining_distribution,
                                               'distribution_parameters': [0, random_features_parameters_gamma[0]],
                                               'number_features': small_subset_size,
                                               'random_rotation': random_features_parameters_bining_random_rotation}

        else:
            random_features_dict_parameters = {'distribution': random_features_parameters_distribution_weights,
                                               'distribution_parameters': random_features_parameters_gamma,
                                               'activation': random_features_parameters_activation_function,
                                               'number_features': small_subset_size,
                                               'bias_distribution': random_features_parameters_distribution_bias,
                                               'bias_distribution_parameters': random_features_parameters_distribution_bias_parameters}

        if not do_ranking_before_full_panel:
            print('Now preselect the balanced panel before doing random features and ranking')
            raw_signals, labels, date_ids, stock_ids = select_balanced_panel(raw_signals=raw_signals,
                                                                             labels=labels,
                                                                             date_ids=date_ids,
                                                                             stock_ids=stock_ids)

        sample_size = raw_signals.shape[0]
        psi_matrix, sigma_hat_eig, true_psi_matr, random_features, voc_curve \
            = RandomFeatures.compute_psi_and_sigma_matrix(seed=seed,
                                                          number_random_features=number_random_features,
                                                          small_subset_size=small_subset_size,
                                                          sample_size=sample_size,
                                                          stock_ids=stock_ids,
                                                          date_ids=date_ids,
                                                          raw_signals=raw_signals,
                                                          labels=labels,
                                                          shrinkage_list=shrinkage_list,
                                                          ranking=ranking,
                                                          test_mode=test_mode,
                                                          do_ranking_before_full_panel=do_ranking_before_full_panel,
                                                          produce_voc_curve=produce_voc_curve,
                                                          voc_grid=voc_grid,
                                                          random_features_dict_parameters=random_features_dict_parameters,
                                                          normalize_p=normalize_p)
        if not produce_voc_curve:
            if do_ranking_before_full_panel:
                _, labels_post_select, _, _ = select_balanced_panel(raw_signals=raw_signals,
                                                                    labels=labels,
                                                                    date_ids=date_ids,
                                                                    stock_ids=stock_ids)
                q_vector, psi_hat_eig = RandomFeatures.build_the_q_vector(psi_matrix,
                                                                          labels_post_select,
                                                                          shrinkage_list,
                                                                          number_random_features,
                                                                          normalize_p=normalize_p)
            else:
                q_vector, psi_hat_eig = RandomFeatures.build_the_q_vector(psi_matrix,
                                                                          labels,
                                                                          shrinkage_list,
                                                                          number_random_features,
                                                                          normalize_p=normalize_p)
            voc_curve['psi_eig'] = {number_random_features: q_vector}  # todo: this seems like a bug
            voc_curve['sigma_eig'] = {number_random_features: sigma_hat_eig}
            voc_curve['q_vectors'] = {number_random_features: q_vector}

        # q_vector \in R^{T\times len(shrinkage_list)}
        # but psi_hat_eig have lots of missing zeros. We should add them

        number_dates = len(np.unique(date_ids))
        betas, realized_in_sample_returns, future_predictions, future_random_features_all \
            = RandomFeatures.compute_betas_and_predictions(
            seed=seed,
            shrinkage_list=shrinkage_list,
            future_raw_signals=future_raw_signals,
            number_random_features=number_random_features,
            small_subset_size=small_subset_size,
            raw_signals=raw_signals,
            date_ids=date_ids,
            stock_ids=stock_ids,
            ranking=ranking,
            future_date_ids=future_date_ids,
            labels=labels,
            number_dates=number_dates,
            voc_grid=voc_grid,
            voc_curve=voc_curve,
            produce_betas=produce_betas,
            do_ranking_before_full_panel=do_ranking_before_full_panel,
            test=test_mode,
            random_features_dict_parameters=random_features_dict_parameters,
            normalize_p=normalize_p
        )

        # to compute bstar we need in-sample portfolio returns, port_ret_ins
        # and for that we need to compute betas.
        rmt_stuff = dict()
        for key in future_predictions.keys():
            # here key is always the number of random features
            c_ = key / sample_size
            rmt_stuff[key] \
                = get_all_random_matrix_quantities_for_the_panel(
                c_,
                voc_curve['sigma_eig'][key],
                voc_curve['psi_eig'][key],
                shrinkage_list,
                realized_in_sample_returns[key].flatten(),
                save_sigma_and_psi_eig_in_rmt_stuff=save_sigma_and_psi_eig_in_rmt_stuff)

        # b_star_hat_in_sample has now same length as shrinkage_list

        output = {'rmt_stuff': rmt_stuff,
                  'betas': betas,
                  'future_predictions': future_predictions}

        raw_signals, labels, _, _ = select_balanced_panel(raw_signals=raw_signals,
                                                          labels=labels,
                                                          date_ids=date_ids,
                                                          stock_ids=stock_ids)
        sample_size = raw_signals.shape[0]
        if run_linear_model:
            bench = RandomFeatures.ridge_regression_single_underlying(signals=raw_signals, labels=labels,
                                                                      future_signals=future_raw_signals,
                                                                      shrinkage_list=shrinkage_list)
            output['benchmark_pred'] = bench['predictions']

        if test_mode:
            future_random_features_all = np.concatenate(future_random_features_all, axis=1)
            beta_true = np.concatenate([np.linalg.inv(z * np.eye(number_random_features)
                                                      + true_psi_matr / sample_size / number_random_features)
                                        @ (random_features.T / np.sqrt(number_random_features)) @ labels
                                        for z in shrinkage_list], axis=1) / raw_signals.shape[0]
            future_predictions_true = future_random_features_all @ beta_true / np.sqrt(number_random_features)
            print(f'Please enjoy the power of math: \n'
                  f'{betas}\n versus \n '
                  f'{beta_true}')
            print(f'and predictions:\n'
                  f'{output["future_predictions"]}\n'
                  f'and '
                  f'{future_predictions_true}')
            return output, beta_true, random_features, future_random_features_all, date_ids
        return output

    @staticmethod
    def ridge_regression_single_underlying(signals: np.ndarray,
                                           labels: np.ndarray,
                                           future_signals: np.ndarray,
                                           shrinkage_list: list,
                                           use_msrr: bool = False,
                                           return_in_sample_pred: bool = False,
                                           compute_smart_weights: bool = False,
                                           test_fixed_kappas: bool = False,
                                           fixed_kappas=[],
                                           test_changing_kappas: bool = False,
                                           constant_signal=None,
                                           print_time: bool = False,
                                           return_beta: bool = False,
                                           keep_only_big_beta: bool = False,
                                           core_z_values=[1, 100],
                                           clip_bstar=100):
        """
        WARNING: I AM ASSUMING THAT SIGNALS ARE ALREADY SHIFTED RELATIVE TO RETURNS
        SO THAT WE ARE TRADING R * S
        signals are assumed to be T \times M
        (in the machine learning jargon, T = N, M =P )

        We would like to invert A'A+shrinkage *I
        But this is too costly.
        So we use smart algebra:
        AA'v = lambda v implies
        A'A (A'v) = lambda (A'v)
        and \|A'v\|^2=v'AA'v=lambda. Thus,
        if AA' = V D V'
        we have A'A = U (D)U'
        where U = [A'v D^{-1/2}]
        and the quasi-inverse will be
        (A'A)^{-1} = U (D^{-1})U' = A'v D^{-2} v'A
        and shrinkage just replaces here
        (A'A)^{-1} = U ((D+shrinkage)^{-1})U'
        therefore
        (A'A)^{-1} =  A'v  ((D+shrinkage)^{-1}) D^{-1} v'A


        :param signals: in sample features
        :param labels: in sample labels
        :param future_signals: OOS features
        :param shrinkage_list: list of ridge basic_parameters
        :param use_msrr: if use MSRR, then betas are determined by MSRR regression
        :param compute_smart_weights :
        :param return_in_sample_pred :
        :param future_signals :
        :param core_z_values : if not None: # in this case we fit both bstar and residual bstar to the
        :param test_fixed_kappas: True if testing fixed kappas
        :param fixed_kappas: list of kappas, will refer to this when test_fixed_kappas is True
        \|\hat\beta(z)\| evaluated at core_z_values
        :return:


        """
        # print(f'upps, running signals.shape={signals.shape}')
        t1__ = time.time()
        signals[np.isnan(signals)] = 0
        if print_time:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Start regression at =", current_time)
            start = time.time()
        # this is the estimate of the Sigma = S'S
        sigma_hat = (signals ** 2).sum().sum() / (signals.shape[0] * signals.shape[1])
        names = ['mean', 'median', 'max-z', 'based_on_mean_mean', 'based_on_mean_median']
        if core_z_values is not None:
            names += ['fitted']  # in this case we fit both bstar and residual bstar to the \|\hat\beta(z)\| evaluated
            # at core_z_values
        all_the_names = ['naive', 'naive_unadjusted'] + names

        beta_names = shrinkage_list + [f'optimal_z_with_{x}' for x in all_the_names] \
                     + [f'optimal_combination_with_{x}' for x in all_the_names]
        if signals.shape[1] < 2 or np.max(np.max(np.abs(signals))) > 10 ** 12:
            # this means data is bad and we produce zeros as output
            norm_of_beta = np.zeros([1, len(shrinkage_list)])
            betas = np.zeros([len(beta_names), signals.shape[1]])
            predictions = np.zeros([future_signals.shape[0], len(beta_names)])
            all_the_bstar = np.zeros([1, len(all_the_names)])
            ce_from_paper = np.zeros([1, 2 * len(all_the_names)])
            cl_from_paper = np.zeros([1, 2 * len(all_the_names)])

            output = RandomFeatures.populate_output(norm_of_beta,
                                                    predictions,
                                                    betas,
                                                    signals,
                                                    return_in_sample_pred,
                                                    return_beta,
                                                    compute_smart_weights,
                                                    all_the_names,
                                                    beta_names,
                                                    all_the_bstar,
                                                    ce_from_paper,
                                                    cl_from_paper,
                                                    shrinkage_list,
                                                    future_signals)

            return output
        t1 = time.time()
        if test_fixed_kappas:
            large_covariance, eigvec1, eigval, data_for_covariance, data_for_covariance_mean, \
            data_for_covariance_shrinkage, mu, labels_mu \
                = RandomFeatures.get_eigen_decomposition_of_covariance_in_smart_way(
                signals,
                labels,
                use_msrr,
                test_fixed_kappas,
                fixed_kappas)
        else:
            large_covariance, eigvec1, eigval, data_for_covariance, mu, labels_mu, signals_mu \
                = RandomFeatures.get_eigen_decomposition_of_covariance_in_smart_way(
                signals,
                labels,
                use_msrr)

        t2 = time.time()
        if print_time:
            print(f'signals are {signals.shape}-sized and eigen-decomposition took {t2 - t1}', flush=True)

        t1 = time.time()
        if large_covariance:
            eigvec = np.matmul(data_for_covariance, eigvec1 * ((eigval * signals.shape[0]) ** (-1 / 2)).reshape(1, -1))
        else:
            eigvec = eigvec1

        multiplied = np.matmul(eigvec.T, mu)  # for a window of 2000 and 4000 signals, the cost is 4000 * 4000
        t2 = time.time()

        if print_time:
            print(f'with large_covariance, multiplying eigenvectors took {t2 - t1}')
        betas = RandomFeatures.produce_betas(multiplied,
                                             eigvec,
                                             eigval,
                                             shrinkage_list)
        t3 = time.time()
        if print_time:
            print(f'computing betas took {t3 - t2}')
        if len(labels.shape) <= 1:
            norm_of_beta = (betas ** 2).sum(1).reshape([1, len(shrinkage_list)])
        elif labels.shape[1] == 1:
            norm_of_beta = (betas ** 2).sum(1).reshape([1, len(shrinkage_list)])
        else:
            norm_of_beta = 0

        # this is (1/T) \sum_t \beta' S_t R_t
        # it is needed for random matrix theory
        in_sample_means = betas @ mu.reshape(-1, 1)

        # now we recover bstar from norm_of_beta
        all_the_bstar = None
        ce_from_paper = None
        cl_from_paper = None
        if compute_smart_weights:
            betas, all_the_bstar, ce_from_paper, cl_from_paper = \
                RandomFeatures.compute_smart_betas_with_rmt(in_sample_means,
                                                            norm_of_beta,
                                                            sigma_hat,
                                                            shrinkage_list,
                                                            signals,
                                                            eigval,
                                                            labels,
                                                            large_covariance,
                                                            core_z_values,
                                                            multiplied,
                                                            eigvec,
                                                            betas,
                                                            clip_bstar
                                                            )

            # so now, our beta estimates come in three blocks
            # raw for each fixed z; with optimal z; and
        else:
            all_the_names = None
            beta_names = None
        if keep_only_big_beta:
            q_ = 0.05
            print(f'TRUNCATE BETA AT Q{q_}')
            q_low = np.quantile(betas, q_)
            q_high = np.quantile(betas, 1 - q_)
            betas[(betas < q_high) & (betas > q_low)] = 0
        t1 = time.time()

        if test_fixed_kappas:
            predictions = fixed_kappas[2] * labels_mu + \
                          np.matmul(betas, fixed_kappas[3] * data_for_covariance_mean + \
                                    future_signals.T - data_for_covariance_shrinkage.T).T
        elif test_changing_kappas:
            inverses_signal_mult_mu = RandomFeatures.produce_inverses_signals_mult_mu(signals_mu, eigvec, eigval,
                                                                                      shrinkage_list)  ## inverse_mult_signal_mu
            GAMMA, Q, xi1, xi2, kappa2, kappa3 = RandomFeatures.produce_gamma_q_xis_and_kappas(constant_signal,
                                                                                               signals_mu,
                                                                                               inverses_signal_mult_mu,
                                                                                               labels_mu,
                                                                                               betas, shrinkage_list)
            predictions = (kappa3 * labels_mu + np.matmul(betas + kappa2 * inverses_signal_mult_mu.T * labels_mu,
                                                          future_signals.T)).T  # dim = (# of z) * 1
        else:
            predictions = np.matmul(betas, future_signals.T).T

        output = RandomFeatures.populate_output(norm_of_beta,
                                                predictions,
                                                betas,
                                                signals,
                                                return_in_sample_pred,
                                                return_beta,
                                                compute_smart_weights,
                                                all_the_names,
                                                beta_names,
                                                all_the_bstar,
                                                ce_from_paper,
                                                cl_from_paper,
                                                shrinkage_list,
                                                future_signals)
        t2 = time.time()
        if print_time:
            print(f'and finally producing predictions took {t2 - t1}')

        if print_time:
            print('Ridge took:', np.round((time.time() - start) / 60, 2), 'm', flush=True)
        t2__ = time.time()
        # print(f'----------------------------------------\n '
        #       f'just ran ridge for output_layer with signals.shape={signals.shape} in {t2__ - t1__} seconds\n '
        #       f'-----------------------------------------')
        return output

    @staticmethod
    def naive_linear_single_underlying(signals: np.ndarray,
                                       labels: np.ndarray,
                                       future_signals: np.ndarray,
                                       shrinkage_list: list,
                                       use_msrr: bool = False,
                                       return_in_sample_pred: bool = False,
                                       compute_smart_weights: bool = False,
                                       oos_returns: np.ndarray = np.array([0])):
        """
        This function computes NaiveLinear_{t+1} =\sum_i S_{i,t} R_{t+1}
        :param signals: in sample features
        :param labels: in sample labels
        :param future_signals: OOS features
        :param shrinkage_list: list of ridge basic_parameters
        :param use_msrr: if use MSRR, then betas are determined by MSRR regression
        :param compute_smart_weights :
        :param return_in_sample_pred :
        :param future_signals :
        :param core_z_values : if not None: # in this case we fit both bstar and residual bstar to the
        \|\hat\beta(z)\| evaluated at core_z_values
        :return:
        """
        # print(f'upps, running signals.shape={signals.shape}')
        signals[np.isnan(signals)] = 0
        naive_linear_returns = future_signals.mean() * oos_returns
        return naive_linear_returns

    @staticmethod
    def get_a_single_slice(signals,
                           returns,
                           i: int,
                           window: int,
                           lag: int,
                           stepp: int,
                           shrinkage_list: list,
                           use_msrr: bool,
                           maximal_fraction_of_nans: float = 0.85,
                           normalize_before_ridge: bool = False,
                           compute_smart_weights=False,
                           predict_one_period=False,
                           dates=None,
                           return_stuff_for_ridge=True,
                           produce_oos_returns=False,
                           test_fixed_kappas=False,
                           fixed_kappas=[],
                           test_changing_kappas=False,
                           constant_signal=None):
        '''
        This function get data slices including returns and signals
        :param signals:
        :param returns:
        :param i:
        :param window:
        :param lag:
        :param stepp:
        :param shrinkage_list:
        :param use_msrr:
        :param maximal_fraction_of_nans:
        :param normalize_before_ridge:
        :param compute_smart_weights:
        :param predict_once: If predict_once is True, then future_signals only contains one observation
        :param test_fixed_kappas: True if testing fixed kappas
        :param fixed_kappas: list of kappas, will refer to this when test_fixed_kappas is True
        :param test_changing_kappas: True if testing changing kappas
        :param constant_signal: None if not testing changing kappas, $b$ (int or float) if testing
        :return:
        '''
        if lag < 1:
            print('hey, lag must be positive')
            return
        # TODO WE SHOULD SOMEHOW PRE-FILTER RETURNS OF STOCKS FOR WHICH THERE IS NO ENOUGH DATA
        if dates is None:
            # in this case we always work directly with numpy arrays
            rets = returns[(i - window - lag + 1):(i - lag + 1)] if len(returns.shape) == 1 else \
                returns[(i - window - lag + 1):(i - lag + 1), :]

            sigs = signals[(i - window - lag + 1):(i - lag + 1), :]
        else:
            rets = returns.loc[(returns.date >= dates[(i - window - lag + 1)])
                               & (returns.date < dates[(i - lag + 1)])]
            sigs = signals.loc[(returns.date >= dates[(i - window - lag + 1)])
                               & (returns.date < dates[(i - lag + 1)])]

        # ******************************
        # we only optimize over stuff for which we have enough data
        # ******************************
        # TODO THIS PRE-FILTERING IS PROBABLY NOT SUITED WELL FOR UN-BALANCED PANEL DATA
        good_indices = (np.isnan(sigs).sum(0) < maximal_fraction_of_nans * rets.shape[0]) * (np.std(sigs, axis=0) > 0)
        if dates is None:
            used_signals = sigs[:, good_indices]
            used_signals[np.isnan(used_signals)] = 0
        else:
            # todo is it ok to do fillna(0) ?
            used_signals = sigs.loc[:, good_indices].fillna(0)

        if dates is None:
            if predict_one_period:
                future_signals = signals[i, good_indices].reshape(1, -1)
                if produce_oos_returns:
                    oos_returns = returns[i:(i + 1)] if len(returns.shape) == 1 \
                        else returns[i:(i + 1), :]
            else:
                future_signals = signals[i:min(i + stepp, signals.shape[0]), good_indices]
                if produce_oos_returns:
                    oos_returns = returns[i:min(i + stepp, signals.shape[0])] if len(returns.shape) == 1 \
                        else returns[i:min(i + stepp, signals.shape[0]), :]
        else:
            # here for simplicity we assume stepp == 1 todo do we need step > 1
            future_signals = signals.loc[signals.date == dates[i]]

        if normalize_before_ridge:
            used_signals_std = np.sqrt(np.sum(used_signals ** 2, axis=0))
            # used_signals_std = np.std(used_signals, axis=0, ddof=1)
            used_signals /= used_signals_std
            future_signals /= used_signals_std
        output = [used_signals, rets, future_signals, shrinkage_list]
        if return_stuff_for_ridge:
            output += [use_msrr, False, compute_smart_weights]

        if test_fixed_kappas:  # argument #8 and #9 in self.ridge_regression_single_underlying()
            output += [test_fixed_kappas]
            output += [fixed_kappas]

        if test_changing_kappas:
            output += [False]  # test_fixed_kappas=False
            output += [[]]
            output += [test_changing_kappas]
            output += [constant_signal]

        if produce_oos_returns:
            output += [oos_returns]
        return output

    def rolling_ridge_universal(self,
                                signals: pd.DataFrame,
                                returns: pd.DataFrame,
                                window: int,
                                diagonal_shrinkage: list,
                                lag: int = 1,
                                use_msrr=False,
                                stepp=30,
                                normalize_before_ridge=True,
                                maximal_fraction_of_nans=0.85,
                                compute_smart_weights=False,
                                predict_one_period=False,
                                panel_regression=False,
                                test_fixed_kappas=False,
                                fixed_kappas=[],
                                test_changing_kappas=False,
                                constant_signal=None) -> tuple:
        """
        This function gets prediction of rolling ridge regression
        :param use_msrr: In the regression, we can either invert the covariance matrix of signals (the OLS approach),
        or invert the covariance matrix of managed returns (S*R); the latter is the MSRR approach, and corresponds to
        use_msrr=True
        :param stepp:
        :param signals:
        :param returns:
        :param window:
        :param diagonal_shrinkage:
        :param lag:
        :return:
        """
        dates = returns.index
        t1 = time.time()
        # first we slice the data into time slices
        slices = [(self.get_a_single_slice(signals=signals.values,
                                           returns=returns.values,
                                           i=i,
                                           window=window,
                                           lag=lag,
                                           stepp=stepp,
                                           shrinkage_list=diagonal_shrinkage,
                                           use_msrr=use_msrr,
                                           maximal_fraction_of_nans=maximal_fraction_of_nans,
                                           normalize_before_ridge=normalize_before_ridge,
                                           compute_smart_weights=compute_smart_weights,
                                           predict_one_period=predict_one_period,
                                           test_fixed_kappas=test_fixed_kappas,
                                           fixed_kappas=fixed_kappas,
                                           test_changing_kappas=test_changing_kappas,
                                           constant_signal=constant_signal))
                  for i in range(window + lag - 1, returns.shape[0], stepp)]

        t2 = time.time()
        print(f'slicing took {t2 - t1}')
        results = [self.ridge_regression_single_underlying(*slice_) for slice_ in slices]

        t3 = time.time()
        print(f'regression took {t3 - t2}')
        # 'beta_names' will be absent when compute_smart_weights = False
        cols = results[0]['beta_names'] if 'beta_names' in results[0].keys() else diagonal_shrinkage
        norms_of_beta = np.array([result['norm_of_beta'] for result in results]).squeeze().mean(0)

        if not panel_regression:
            # now we take all the slices and concatenate them into a pd.DataFrame of predictions
            # each column corresponds to a different way of predicting
            predictions = pd.DataFrame(index=dates, columns=cols).astype(float)
            predictions.loc[dates[window + lag - 1]:] \
                = np.concatenate([result['predictions'] for result in results], axis=0)

        output = {'norms_of_beta': norms_of_beta, 'predictions': predictions}

        if compute_smart_weights:
            # the first few elements of 'beta_names' are just diagonal_shrinkage
            # however, for computing 'ce_from_paper', 'cl_from_paper', we need to choose
            # a bstar estimator
            # on top of that, we also have a portfolio version
            cols1 = results[0]['beta_names'][len(diagonal_shrinkage):]
            cols2 = results[0]['names_for_bstar_estimation']
            for key in ['smart_bstar', 'ce_from_paper', 'cl_from_paper']:
                tmp = pd.DataFrame(index=dates, columns=cols2 if key == 'smart_bstar' else cols1).astype(float)
                if not panel_regression:
                    tmp.loc[dates[window + lag - 1]:] = np.concatenate([result[key] for result in results], axis=0)
                output[key] = tmp
        t4 = time.time()

        print(f'last steps took {t4 - t3}')
        return output

    def naive_linear_strategy(self,
                              signals: pd.DataFrame,
                              returns: pd.DataFrame,
                              window: int,
                              diagonal_shrinkage: list,
                              lag: int = 1,
                              use_msrr=False,
                              stepp=30,
                              normalize_before_ridge=True,
                              maximal_fraction_of_nans=0.85,
                              compute_smart_weights=False,
                              predict_one_period=False,
                              panel_regression=False) -> tuple:
        """
        This function gets prediction of
        :param use_msrr: In the regression, we can either invert the covariance matrix of signals (the OLS approach),
        or invert the covariance matrix of managed returns (S*R); the latter is the MSRR approach, and corresponds to
        use_msrr=True
        :param stepp:
        :param signals:
        :param returns:
        :param window:
        :param diagonal_shrinkage:
        :param lag:
        :return:
        """
        dates = returns.index
        t1 = time.time()
        # first we slice the data into time slices
        slices = [(self.get_a_single_slice(signals=signals.values,
                                           returns=returns.values,
                                           i=i,
                                           window=window,
                                           lag=lag,
                                           stepp=stepp,
                                           shrinkage_list=diagonal_shrinkage,
                                           use_msrr=use_msrr,
                                           maximal_fraction_of_nans=maximal_fraction_of_nans,
                                           normalize_before_ridge=normalize_before_ridge,
                                           compute_smart_weights=compute_smart_weights,
                                           predict_one_period=predict_one_period,
                                           produce_oos_returns=True))  # location 2 is future signals, 7 is OOS returns
                  for i in range(window + lag - 1, returns.shape[0], stepp)]

        t2 = time.time()
        print(f'slicing took {t2 - t1}')
        results = np.array([self.naive_linear_single_underlying(*slice_) for slice_ in slices])
        naive_linear_strategy_returns = pd.DataFrame(results, index=dates[window + lag - 1:],
                                                     columns=['naive_linear_strategy_returns']).astype(float)
        return naive_linear_strategy_returns

    @staticmethod
    def generate_parameters_for_neural_features():
        '''
        Get sets of basic_parameters
        :return:
        '''
        parameters = list()
        for distribution in ['normal', 'f', 'gamma', 'gumbel', 'laplace']:
            for parameter1 in np.arange(1, 10):
                for parameter2 in np.arange(1, 10):
                    for activation in RandomFeatures().permitted_activation_functions:
                        for number_features in [10, 50, 100, 500, 1000, 2000]:
                            for bias_distribution in ['normal', 'f', 'gamma', 'gumbel', 'laplace']:
                                for bias_parameter1 in np.arange(1, 10):
                                    for bias_parameter2 in np.arange(1, 10):
                                        parameters += [distribution,
                                                       [parameter1, parameter2],
                                                       activation,
                                                       number_features,
                                                       bias_distribution,
                                                       [bias_parameter1, bias_parameter2]]
        return parameters

    def generate_random_neuron_features(self,
                                        signals: np.ndarray,
                                        distribution_requirements: dict,
                                        distribution: str,
                                        distribution_parameters: list,
                                        activation: str,
                                        number_features: int,
                                        bias_distribution=None,
                                        bias_distribution_parameters=None,
                                        random_seed=0):
        """
        this function builds random neuron features f(w'S) where w is a vector of random weights and
        f is an activation function
        :param random_seed:
        :param signals:
        :param bias_distribution_parameters:
        :param distribution_requirements:
        :param distribution_parameters:
        :param distribution:
        :param activation:
        :param number_features:
        :param bias_distribution:
        :return:
        """
        self.check_distribution_requirements(distribution, distribution_parameters, distribution_requirements)
        if bias_distribution:
            self.check_distribution_requirements(bias_distribution, bias_distribution_parameters,
                                                 distribution_requirements)

        number_signals = signals.shape[1]
        size = [number_signals, number_features]
        if activation == 'cos_and_sin':
            size = [number_signals, int(number_features / 2)]

        # first we initialize the random seed
        np.random.seed(random_seed)
        # X = np.random.normal(0, a) means X is distributed as Normal(0, a^2).  (a=standard deviation)
        # This is an important property of Gaussian distributions: multiplying by a constant keeps is Gaussian,
        # just scales the standard deviation
        if distribution != 'gaussian_mixture':
            random_vectors = getattr(np.random, distribution)(*distribution_parameters, size)
        else:
            random_vectors = getattr(np.random, 'standard_normal')(size)
            gamma_values = distribution_parameters
            minimal_gamma = gamma_values[0]
            maximal_gamma = gamma_values[1]
            all_gamma_values = np.random.uniform(minimal_gamma, maximal_gamma, [1, size[1]])
            # now we use numpy broadcasting to do elemen-wise multiplication.
            random_vectors = random_vectors * all_gamma_values

        # This is for debug with Matlab only
        debug_with_matlab = False
        if debug_with_matlab:
            random_vectors = pd.read_csv(
                '/Users/kyzhou/Dropbox/MSRR/Code/Empirical/RFF_matlab_test/random_vectors_matlab.csv', header=None)
            random_vectors = np.array(random_vectors).T

        multiplied_signals = np.matmul(random_vectors.T, signals.T)
        if bias_distribution:
            multiplied_signals = self.add_bias(multiplied_signals, bias_distribution, bias_distribution_parameters)
        final_random_features = self.apply_activation_to_multiplied_signals(multiplied_signals, activation)
        return final_random_features

    @staticmethod
    def check_distribution_requirements(distribution,
                                        distribution_parameters,
                                        distribution_requirements):
        if distribution == 'gaussian_mixture':
            return
        if distribution not in distribution_requirements:
            raise Exception(f'{distribution} is not permitted. If you need it, do not be lazy and update the class')
        elif len(distribution_parameters) != distribution_requirements[distribution]:
            raise Exception(f'{distribution} requires {distribution_requirements[distribution]} basic_parameters')

    def generate_random_binning_features(self,
                                         signals: np.ndarray,
                                         distribution_requirements: dict,
                                         distribution: str,
                                         distribution_parameters: list,
                                         number_features: int,
                                         random_rotation=False,
                                         random_seed=0):
        """
           WARNING: THE FEATURES ARE SUPPOSED TO BE NORMALIZED!!!
           ALWAYS PRE-PROCESS THE DATA (USING ROLLING WINDOW) !!!
           signals are assumed to be T \times M
           :param random_seed:
           :param random_rotation:
           :param distribution_parameters:
           :param distribution:
           :param distribution_requirements:
           :param signals:
           :param number_features:
           :return:
           """

        self.check_distribution_requirements(distribution, distribution_parameters, distribution_requirements)

        number_signals = signals.shape[1]
        size = [number_signals, number_features]
        np.random.seed(random_seed)
        if random_rotation:
            rotate = np.random.randn(signals.shape[1], signals.shape[1])
            tmp = np.matmul(rotate, rotate.T)
            _, eigvec = np.linalg.eigh(tmp)
            # now, eigenvectors give a random rotation
            signals_rotated = np.matmul(eigvec.T, signals.T).T
        else:
            signals_rotated = signals.copy()
        delta = getattr(np.random, distribution)(*distribution_parameters, size)
        delta = delta * (np.abs(delta) > (10 ** (- 10))) + (np.abs(delta) < (10 ** (- 10))) * (10 ** (-10))  # clip
        u_ = np.random.uniform(0, 1, [number_signals, number_features]) * delta
        subtracted = signals_rotated.reshape(signals.shape[0], 1, number_signals) \
                     - u_.reshape(1, number_features, number_signals)
        subtracted_and_divided = subtracted / delta.reshape(1, number_features, number_signals)

        binned_signals = np.floor(subtracted_and_divided).reshape([signals.shape[0],
                                                                   signals.shape[1] * number_features])
        return binned_signals


class RandomFeaturesTesting:
    """
    This class manages creation of random features

    #########################################
    The list of possible distributions is:
    #########################################
    beta(a, b[, size])	Draw samples from a Beta distribution.
    binomial(n, p[, size])	Draw samples from a binomial distribution.
    chisquare(df[, size])	Draw samples from a chi-square distribution.
    dirichlet(alpha[, size])	Draw samples from the Dirichlet distribution.
    exponential([scale, size])	Draw samples from an exponential distribution.
    f(dfnum, dfden[, size])	Draw samples from an F distribution.
    gamma(shape[, scale, size])	Draw samples from a Gamma distribution.
    geometric(p[, size])	Draw samples from the geometric distribution.
    gumbel([loc, scale, size])	Draw samples from a Gumbel distribution.
    hypergeometric(ngood, nbad, nsample[, size])	Draw samples from a Hypergeometric distribution.
    laplace([loc, scale, size])	Draw samples from the Laplace or double
    exponential distribution with specified location (or mean) and scale (decay).
    logistic([loc, scale, size])	Draw samples from a logistic distribution.
    lognormal([mean, sigma, size])	Draw samples from a log-normal distribution.
    logseries(p[, size])	Draw samples from a logarithmic series distribution.
    multinomial(n, pvals[, size])	Draw samples from a multinomial distribution.
    multivariate_normal(mean, cov[, size, …)	Draw random samples from a multivariate normal distribution.
    negative_binomial(n, p[, size])	Draw samples from a negative binomial distribution.
    noncentral_chisquare(df, nonc[, size])	Draw samples from a noncentral chi-square distribution.
    noncentral_f(dfnum, dfden, nonc[, size])	Draw samples from the noncentral F distribution.
    normal([loc, scale, size])	Draw random samples from a normal (Gaussian) distribution.
    pareto(a[, size])	Draw samples from a Pareto II or Lomax distribution with specified shape.
    poisson([lam, size])	Draw samples from a Poisson distribution.
    power(a[, size])	Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
    rayleigh([scale, size])	Draw samples from a Rayleigh distribution.
    standard_cauchy([size])	Draw samples from a standard Cauchy distribution with mode = 0.
    standard_exponential([size])	Draw samples from the standard exponential distribution.
    standard_gamma(shape[, size])	Draw samples from a standard Gamma distribution.
    standard_normal([size])	Draw samples from a standard Normal distribution (mean=0, stdev=1).
    standard_t(df[, size])	Draw samples from a standard Student’s t distribution with df degrees of freedom.
    triangular(left, mode, right[, size])	Draw samples from the triangular distribution over the interval [left, right].
    uniform([low, high, size])	Draw samples from a uniform distribution.
    vonmises(mu, kappa[, size])	Draw samples from a von Mises distribution.
    wald(mean, scale[, size])	Draw samples from a Wald, or inverse Gaussian, distribution.
    weibull(a[, size])	Draw samples from a Weibull distribution.
    zipf(a[, size])	Draw samples from a Zipf distribution.

    ####################################################################################################################
    activation functions
    ####################################################################################################################
    cos, sin, exp, arctan (sigmoid style!), tanh,
    ReLu (careful, relu is implemented separately through the multiplication method (x * (x > 0)) which is the fastest
    Elu (x * (x > 0)) + alpha * (exp(x) - 1) * (x < 0)
    and SoftPlus = log(1+exp(x))

    """

    # the next shows the number of basic_parameters defining the distribution
    distribution_requirements = {'beta': 2, 'binomial': 2, 'chisquare': 0,
                                 'dirichlet': 1, 'exponential': 0, 'f': 2,
                                 'gamma': 1, 'geometric': 1, 'gumbel': 2,
                                 'hypergeometric': 2, 'laplace': 2, 'logistic': 2, 'lognormal': 2,
                                 'logseries': 1, 'multinomial': 2, 'multivariate_normal': 2,
                                 'negative_binomial': 2, 'noncentral_chisquare': 2,
                                 'noncentral_f': 3, 'normal': 2, 'pareto': 1, 'poisson': 1,
                                 'power': 1, 'rayleigh': 1, 'standard_cauchy': 0, 'standard_exponential': 0,
                                 'standard_gamma': 1, 'standard_normal': 0, 'standard_t': 1, 'triangular': 3,
                                 'uniform': 2, 'vonmises': 2, 'wald': 2, 'weibull': 1, 'zipf': 1,
                                 'gaussian_mixture': 0}

    permitted_activation_functions = ['cos', 'sin', 'exp', 'arctan', 'tanh', 'ReLu', 'Elu', 'SoftPlus',
                                      'cos_and_sin']

    def __init__(self,
                 raw_signals: pd.DataFrame = None,
                 returns: pd.DataFrame = None,
                 asset_class: str = 'futures',
                 balanced: bool = True,
                 ticker=None,
                 logger=None,
                 horizon=1,
                 normalize_before_ridge=False,
                 produce_and_plot_linear_managed_returns=False,
                 country_code=None,
                 settings=None,
                 clip_extreme_positions=True,
                 test_fixed_kappas=False,
                 fixed_kappas=[],
                 test_changing_kappas=False,
                 constant_signal=None):
        '''
        This function creates an instance of random features
        :param raw_signals:
        :param returns:
        :param asset_class:
        :param balanced:
        :param ticker:
        :param logger:
        '''

        self.rf = RandomFeatures()  # create a single rf to host functions
        self.settings = settings
        self.verbose = settings.verbose
        self.shrinkage_list = settings.shrinkage_list
        self.lag = settings.lag  # this is the lag for computing regression coefficients
        self.use_msrr = settings.use_msrr  # then we run simple ridge
        # the next dictionary defines the step in rolling ridge regression for each rolling window.
        # the longer the window, the larger the step can be because fast reaction to changing conditions is less relevant
        self.step_map = settings.step_map
        self.windows = settings.windows
        self.numbers_of_signals = settings.numbers_of_signals
        self.results_folder = settings.results_folder
        self.plots_folder = settings.linear_plots_folder
        self.normalize_before_ridge = normalize_before_ridge
        self.ticker = ticker
        self.horizon = horizon
        self.test_fixed_kappas = test_fixed_kappas
        self.fixed_kappas = fixed_kappas
        self.test_changing_kappas = test_changing_kappas
        self.constant_signal = constant_signal
        # TODO: Add self.returns_frequency to class BasicRandomFeatureSettingForFutures and
        #  class BasicRandomFeatureSettingForBonds
        self.returns_frequency = settings.returns_frequency
        self.predict_one_period = settings.predict_one_period
        self.country_code = country_code

        sigs = raw_signals.copy()
        rets = returns.copy()
        if balanced:
            sigs.dropna(axis=1, inplace=True, how='all')  # drop columns with all nan
            sigs.dropna(inplace=True, how='all')  # drop rows with all nan
            rets = rets.reindex(sigs.index).dropna()
            sigs = sigs.reindex(rets.index)

        # create an instance object
        self.raw_signals = sigs
        self.returns = rets
        self.random_features = dict()
        self.predictions = dict()
        self.logger = logger
        self.asset_class = asset_class
        self.linear_managed_returns = dict()
        self.equal_weighted_linear_managed_returns = dict()

        if produce_and_plot_linear_managed_returns:
            t1 = time.time()
            self.produce_and_plot_simple_linear_managed_returns(settings=settings,
                                                                clip=clip_extreme_positions)
            t2 = time.time()
            print(f'plotting linear took {t2 - t1}')

    @staticmethod
    def compare_with_zero_param(managed_returns,
                                file,
                                column_for_comparison='fair_zero_parameter_expanding'):
        tstats = [regression_with_tstats(predicted_variable=managed_returns[col],
                                         explanatory_variables=managed_returns[column_for_comparison])['const']
                  for col in managed_returns.columns if ((str(col) != column_for_comparison)
                                                         & (not str(col).startswith('fair')) & (
                                                                     str(col) != 'zero_param'))]
        if file is None:
            return tstats

        managed_returns = managed_returns / managed_returns.std()

        managed_returns.dropna().cumsum().plot()

        # sr = sharpe_ratio(managed_returns, horizon=horizon)
        plt.title(f'median tst={np.median(tstats)}, max tst={np.max(tstats)}')

        plt.savefig(file)
        plt.close('all')
        return tstats

    @staticmethod
    def compare_with_benchmark(managed_returns,
                               benchmark_col,
                               file):
        """
        This function computes tstats wrt benchmark column
        :param managed_returns:
        :param benchmark_col:
        :param file:
        :return:
        """
        tstats = [regression_with_tstats(predicted_variable=managed_returns[col],
                                         explanatory_variables=managed_returns[benchmark_col])['const']
                  for col in managed_returns.columns if col != benchmark_col]
        if file is None:
            return tstats

        managed_returns = managed_returns / managed_returns.std()

        managed_returns.dropna().cumsum().plot()

        # sr = sharpe_ratio(managed_returns, horizon=horizon)
        plt.title(f'median tst={np.median(tstats)}, max tst={np.max(tstats)}')

        plt.savefig(file)
        plt.close('all')

        return tstats

    def produce_and_plot_simple_linear_managed_returns(self,
                                                       settings,
                                                       clip=True):
        """
        Given raw signals, we just run a rolling ridge regression
        on the returns, with different ridge penalizations, and different rolling windows
        Parameters
        ----------
        horizon :
        settings :

        Returns
        -------

        """
        if self.raw_signals.shape[0] < 10:
            return
        if self.plots_folder is not None:
            folder_for_linear_path = os.path.join(self.plots_folder, f'{self.ticker}_{self.horizon}')
        else:
            folder_for_linear_path = os.path.join(self.results_folder, f'{self.ticker}_{self.horizon}')
        if not os.path.exists(folder_for_linear_path):
            os.mkdir(folder_for_linear_path)
        folder_for_linear = os.path.join(folder_for_linear_path, 'linear_model')
        if not os.path.exists(folder_for_linear):
            os.mkdir(folder_for_linear)
        folder_for_equal_weighted_linear = os.path.join(folder_for_linear_path, 'equal_weighted_linear_model')
        if not os.path.exists(folder_for_equal_weighted_linear):
            os.mkdir(folder_for_equal_weighted_linear)
        for window in settings.windows:
            if self.raw_signals.shape[0] < window + 10:
                continue
            predictions = self.rf.rolling_ridge_universal(signals=self.raw_signals,
                                                          returns=self.returns,
                                                          window=window,
                                                          diagonal_shrinkage=settings.shrinkage_list,
                                                          lag=settings.lag,
                                                          use_msrr=self.use_msrr,
                                                          stepp=settings.step_map[window],
                                                          normalize_before_ridge=True,
                                                          maximal_fraction_of_nans=0.85,
                                                          compute_smart_weights=True,
                                                          predict_one_period=self.predict_one_period,
                                                          panel_regression=False
                                                          )['predictions']
            if clip:
                predictions_clip = predictions.abs().expanding().quantile(0.9).shift(settings.lag)  # .shift(2)
                predictions = predictions.clip(lower=-predictions_clip, upper=predictions_clip, axis=1)
            managed_returns = predictions * self.returns.values.reshape(-1, 1)

            equal_weighted_linear_strategy = self.raw_signals * self.returns.values.reshape(-1, 1)
            managed_returns['zero_param'] = equal_weighted_linear_strategy.mean(1)

            # infeasible by using sign(sum_t (R*S)) of the full sample
            full_sample_sign = np.sign(equal_weighted_linear_strategy.mean())
            infeasible_equal_weighted_linear_strategy = equal_weighted_linear_strategy * full_sample_sign
            managed_returns['zero_param_with_infeasible_sign'] = infeasible_equal_weighted_linear_strategy.mean(1)

            fair_zero_parameter_signal_returns = self.raw_signals.mul(self.returns, axis=0)
            fair_zero_parameter_signal_returns_expanding_sign = np.sign(
                fair_zero_parameter_signal_returns.expanding(min_periods=window).mean().shift(
                    settings.lag))  # .shift(2)
            fair_zero_parameter_expanding = (
                        fair_zero_parameter_signal_returns * fair_zero_parameter_signal_returns_expanding_sign).mean(1)
            managed_returns['fair_zero_parameter_expanding'] = fair_zero_parameter_expanding

            fair_zero_parameter_signal_returns_rolling_sign = np.sign(
                fair_zero_parameter_signal_returns.rolling(window, min_periods=window).mean().shift(
                    settings.lag))  # .shift(2)
            fair_zero_parameter_rolling = (
                        fair_zero_parameter_signal_returns * fair_zero_parameter_signal_returns_rolling_sign).mean(1)
            managed_returns['fair_zero_parameter_rolling'] = fair_zero_parameter_rolling

            managed_returns = managed_returns[
                ['zero_param', 'zero_param_with_infeasible_sign', 'fair_zero_parameter_expanding',
                 'fair_zero_parameter_rolling'] + list(predictions.columns)]

            for vol_adjust in [True, False]:
                if vol_adjust:
                    managed_returns_adjusted, vol = vol_adjust_data(managed_returns,
                                                                    periods=settings.window_for_return_volatility,
                                                                    monthly=(settings.lag == 1))
                else:
                    managed_returns_adjusted = managed_returns.copy()

                managed_returns_adjusted = managed_returns_adjusted.dropna()
                self.linear_managed_returns[window, vol_adjust] = managed_returns_adjusted

                file = os.path.join(folder_for_linear,
                                    f'{self.ticker}_performance_linear_window={window}_vol_adjust={vol_adjust}.jpeg')
                RandomFeaturesTesting.compare_with_zero_param(managed_returns_adjusted,
                                                              file)

        # save linear managed returns
        file = os.path.join(folder_for_linear, f'{self.ticker}_linear_managed_returns.npy')
        np.save(file, self.linear_managed_returns, allow_pickle=True)

    def generate_random_neuron_features_for_testing(self,
                                                    index,
                                                    distribution: str,
                                                    distribution_parameters: list,
                                                    activation: str,
                                                    number_features: int,
                                                    bias_distribution=None,
                                                    bias_distribution_parameters=None):
        """
        this function builds random neuron features f(w'S+bias) where w is a vector of random weights and
        f is an activation function, and bias is a random bias
        :param distribution_parameters:
        :param distribution:
        :param activation:
        :param number_features:
        :param bias_distribution:
        :param index: random seed
        :return:
        """
        signals = self.raw_signals.values
        distribution_requirements = self.distribution_requirements
        final_random_features = self.rf.generate_random_neuron_features(
            signals=signals,
            distribution_requirements=distribution_requirements,
            distribution=distribution,
            distribution_parameters=distribution_parameters,
            activation=activation,
            number_features=number_features,
            bias_distribution=bias_distribution,
            bias_distribution_parameters=bias_distribution_parameters,
            random_seed=index)
        # for debug: size = (5000, 1092)

        final_random_features = pd.DataFrame(final_random_features.T, index=self.raw_signals.index)

        self.random_features[(distribution,
                              tuple(distribution_parameters),
                              activation,
                              number_features,
                              bias_distribution,
                              tuple(bias_distribution_parameters),
                              index)] \
            = final_random_features
        return final_random_features

    def generate_random_features_from_list(self,
                                           list_of_specs,
                                           parallel=False):
        """
        given a list of different specifications, generate random features for each of them
        :param list_of_specs:
        :return:
        """

        if not parallel:
            if list_of_specs[0][0] == 'mix':
                list_of_specs = list_of_specs[0][1]
            for spec in list_of_specs:
                # t1 = time.time()
                # spec[0] is just an index to initialize the np.random.seed
                if len(spec[1]) == 4:
                    self.generate_random_binning_features(spec[0], **spec[1])
                elif len(spec[1]) == 6:
                    self.generate_random_neuron_features_for_testing(spec[0], **spec[1])

                t2 = time.time()
                # print(f'producing random features for {spec} took {t2 - t1}')
            if len(list_of_specs) > 1:
                merged_features = pd.concat([self.random_features[key] for key in self.random_features], axis=1)
                perm = np.random.permutation(merged_features.shape[1])
                merged_features = merged_features.iloc[:, perm]
                self.random_features = {'merged_features': merged_features}

    def generate_random_binning_features(self, index, distribution: str, distribution_parameters: list,
                                         number_features: int, random_rotation=False):
        signals = self.raw_signals.values
        distribution_requirements = self.distribution_requirements
        binned_signals = self.rf.generate_random_binning_features(signals,
                                                                  distribution_requirements,
                                                                  distribution,
                                                                  distribution_parameters,
                                                                  number_features,
                                                                  random_rotation,
                                                                  index)

        binned_signals = pd.DataFrame(binned_signals, index=self.raw_signals.index)
        self.random_features[
            ('random binning', distribution, tuple(distribution_parameters), random_rotation, number_features, index)] \
            = binned_signals

    def rolling_ridge(self,
                      spec_for_random_features,
                      window,
                      compute_smart_weights,
                      panel_regression=False):
        """

        :param spec_for_random_features: specification that we will use for random features
        For example, ['normal', [0, 0.1], 1000, False]
        :param window: rolling window for ridge regression
        :param diagonal_shrinkage: list of values for diagonal shrinkage
        :param lag:
        :param use_msrr:
        :param stepp:
        :return:
        """
        signals = self.random_features[spec_for_random_features]
        numbers_of_sigs = self.numbers_of_signals

        # # save signals for CME_BP
        # save_signals = True
        # if save_signals:
        #     signals_save = signals.dropna()
        #     signals_save.to_csv('CME_BO_signals_gamma0.1.csv')
        #     breakpoint()

        # TODO: Do we need to add all signals by + [signals.shape[1]]? This would cost a long time!
        numbers_of_sigs = list(numbers_of_sigs[numbers_of_sigs < signals.shape[1]]) + [signals.shape[1]]

        returns = self.returns
        if window > min(returns.dropna().shape[0], signals.dropna().shape[0]) - 10:
            return
        for number in numbers_of_sigs:
            print('numbers_of_sigs = ' + str(number))
            # this is the key loop investigating the "virtue of complexity"
            # as we expand the signal universe, we want to see Sharpes to monotonically increase
            if spec_for_random_features[2] == 'cos_and_sin':
                selected_signals_index = [*range(0, int(number / 2))] + \
                                         [*range(int(signals.shape[1] / 2),
                                                 int(signals.shape[1] / 2) + int(number / 2))]
            else:
                selected_signals_index = [*range(0, number)]
            t1 = time.time()
            output \
                = self.rf.rolling_ridge_universal(signals=signals.iloc[:, selected_signals_index],
                                                  returns=returns,
                                                  window=window,
                                                  diagonal_shrinkage=self.shrinkage_list,
                                                  lag=self.lag,
                                                  use_msrr=self.use_msrr,
                                                  stepp=self.step_map[window],
                                                  normalize_before_ridge=self.normalize_before_ridge,
                                                  compute_smart_weights=compute_smart_weights,
                                                  predict_one_period=self.predict_one_period,
                                                  panel_regression=panel_regression,
                                                  test_fixed_kappas=self.test_fixed_kappas,
                                                  fixed_kappas=self.fixed_kappas,
                                                  test_changing_kappas=self.test_changing_kappas,
                                                  constant_signal=self.constant_signal)
            self.predictions[spec_for_random_features, window, number] = output

            if self.verbose:
                managed_rets = output['predictions'] * returns.values.reshape(-1, 1)
                sr_by_freq = sharpe_ratio_by_freq(managed_rets,
                                                  self.returns_frequency)
                print(f'for {spec_for_random_features, window, number}, we got Sharpes of {sr_by_freq}')
                self.logger.info(f'for {spec_for_random_features, window, number}, we got {sr_by_freq}')
            t2 = time.time()
            # breakpoint()
            print(f'producing numbers_of_sigs {number} took {t2 - t1}')

    def run_rolling_ridge_for_all_specs(self,
                                        number_processes=1,
                                        compute_smart_weights=False,
                                        panel_regression=False):
        """
        Here, we loop through all possible random features and also all possible windows
        :param lag:
        :param use_msrr:
        :param stepp:
        :param parallel:
        :return:
        """
        if number_processes == 1:
            for window in self.windows:
                print('window ' + str(window))
                for spec_for_random_features in self.random_features:
                    print('spec_for_random_features ' + str(spec_for_random_features))
                    t1 = time.time()
                    self.rolling_ridge(
                        spec_for_random_features,
                        window,
                        compute_smart_weights,
                        panel_regression)
                    t2 = time.time()
                    print(f'rolling ridge for {window} took {t2 - t1}')
                    self.logger.info(f'rolling ridge for {window} took {t2 - t1}')


def compute_random_features_and_run_ridge_regression(random_features: RandomFeaturesTesting,
                                                     parameter,
                                                     ticker: str,
                                                     horizon=1,
                                                     compute_smart_weights=False,
                                                     test_fixed_kappas=False,
                                                     fixed_kappas=[],
                                                     test_changing_kappas=False,
                                                     constant_signal=None):
    """
    This is function is the main function to get the random features performance
    It is very important here that parameter is a tuple;
    parameter[0] is the np.random.seed to be used
    :param ticker:
    :param random_features:
    :param parameter: could be a dictionary or a list of dictionaries (if we want a giant, merged model)
    :param test_fixed_kappas: True if testing fixed kappas
    :param fixed_kappas: list of kappas, will refer to this when test_fixed_kappas is True
    :param test_changing_kappas: True if testing changing kappas
    :param constant_signal: None if not testing changing kappas, $b$ (int or float) if testing

    :return:
    """
    signals = random_features.raw_signals
    returns = random_features.returns
    logger = logging.getLogger(ticker)
    log_file = os.path.join(random_features.results_folder, f'log_file_{ticker}.log')
    if horizon > 1:
        log_file = log_file.replace('.log', f'_{horizon}.log')
    file_handler = logging.FileHandler(log_file, mode="a")
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    tmp = RandomFeaturesTesting(signals.fillna(0),
                                returns,
                                asset_class=random_features.asset_class,
                                ticker=ticker,
                                logger=logger,
                                horizon=horizon,
                                normalize_before_ridge=random_features.normalize_before_ridge,
                                produce_and_plot_linear_managed_returns=False,
                                country_code=random_features.country_code,
                                settings=random_features.settings,
                                test_fixed_kappas=test_fixed_kappas,
                                fixed_kappas=fixed_kappas,
                                test_changing_kappas=test_changing_kappas,
                                constant_signal=constant_signal)
    tmp.generate_random_features_from_list([parameter])
    t1 = time.time()
    tmp.run_rolling_ridge_for_all_specs(compute_smart_weights=compute_smart_weights)
    result = tmp.predictions
    t2 = time.time()
    if parameter[0] == 'mix':
        result['mixture_model_list'] = parameter
        model0 = parameter[1][0]
        parameter = f'mixture_{model0}.npy'  # parameter is 'mixture_'+first model
    print(f'full round for {parameter}, {ticker} took {t2 - t1}')
    fold = os.path.join(random_features.results_folder, f'{ticker}')
    if horizon > 1:
        fold += f'_{horizon}'
    if not os.path.exists(fold):
        os.mkdir(fold)
    file = os.path.join(fold, f'{parameter}_results.npy')
    np.save(file, result, allow_pickle=True)
    return result


if __name__ == '__main__':
    # seed = 1
    # np.random.seed(seed)
    # number_random_features = 10
    # small_subset_size = 3
    # step_for_voc_curve = 2
    # shrinkage_list = [1, 2]
    # ranking = 'cdf'
    # raw_signals = np.random.randn(5, number_random_features)
    # labels = np.random.randn(5, 1)
    # date_ids = 0 * labels
    #
    # output, betas_true, random_features = RandomFeatures.ridge_regression_with_giant_number_of_random_features(
    #     raw_signals,
    #     labels,
    #     shrinkage_list=shrinkage_list,
    #     stock_ids=labels.flatten(),
    #     date_ids=date_ids,
    #     number_random_features=number_random_features,
    #     small_subset_size=small_subset_size,
    #     seed=seed,
    #     ranking=ranking,
    #     future_raw_signals=raw_signals,
    #     future_date_ids=date_ids,
    #     test_mode=True,
    #     produce_betas=True,
    #     produce_voc_curve=True)
    #
    # # now we use ridge_regression_single_underlying to run ridge regression
    # res_old_func = RandomFeatures.ridge_regression_single_underlying(
    #     signals=random_features / np.sqrt(number_random_features),
    #     labels=labels,
    #     future_signals=random_features,
    #     shrinkage_list=shrinkage_list,
    #     return_beta=True)
    # print(f"The beta from ridge_regression_single_underlying is \n {res_old_func['betas'].T}")

    # my test
    seed = 1
    np.random.seed(seed)

    shrinkage_list = [0.000001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 5000]
    ranking = None
    activation = 'cos'
    raw_dim = 3
    T = 1000
    raw_signals = np.random.randn(T, raw_dim)
    in_sample_signals = raw_signals[:int(T / 2), :]

    out_of_sample_signals = raw_signals[int(T / 2):, :]
    beta = np.random.randn(1, in_sample_signals.shape[1])
    labels = raw_signals @ beta.T + np.random.randn(raw_signals.shape[0], 1) * 0.001
    # y = X * beta + eps
    in_sample_labels = labels[:int(T / 2)]
    out_of_sample_labels = labels[int(T / 2):]
    norm_p_ = True

    date_ids = in_sample_labels * 0
    small_subset_size = 500
    voc_grid = [100, 500, 1000]
    number_random_features = max(voc_grid)
    ranking = 'rank'
    ## get the new predictions
    # output, betas_true, random_features, future_random_features_all \
    res \
        = RandomFeatures.ridge_regression_with_giant_number_of_random_features(
        raw_signals=in_sample_signals,
        labels=in_sample_labels,
        shrinkage_list=shrinkage_list,
        stock_ids=in_sample_labels.flatten(),
        date_ids=date_ids,
        number_random_features=number_random_features,
        small_subset_size=small_subset_size,
        seed=seed,
        ranking=ranking,
        future_raw_signals=out_of_sample_signals,
        future_date_ids=out_of_sample_labels * 0,
        test_mode=False,
        normalize_p=norm_p_,
        run_linear_model=False,
        produce_betas=True,
        produce_voc_curve=True,
        voc_grid=voc_grid,
        do_ranking_before_full_panel=False)

    P = 500
    betas = res['betas'][P]
    t_voc = np.array(voc_grid)
    t_voc = t_voc[t_voc <= P].tolist()

    pred = RandomFeatures.get_predictions_given_beta_with_giant_number_of_random_features(
        shrinkage_list=shrinkage_list,
        betas_pre_trained=betas,
        number_random_features=P,
        small_subset_size=small_subset_size,
        seed=seed,
        ranking=ranking,
        voc_grid=t_voc,
        future_raw_signals=out_of_sample_signals,
        future_date_ids=out_of_sample_labels * 0,
        normalize_p=norm_p_)

    print(res['future_predictions'][P] - pred)
