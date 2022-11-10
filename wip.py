import numpy as np
import pandas as pd
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from helpers.random_features import RandomFeatures
import matplotlib.pyplot as plt


def great(x: int) -> int:
    """

    :param x: ineteger
    :return: another integer, x^2
    """
    return x ** 2


def simulate_data(seed: int,
                  sample_size: int,
                  number_features_: int,
                  beta_and_psi_link_: float,
                  noise_size_: float,
                  activation_: str = 'linear',
                  number_neurons_: int = 1
                  ):
    """
    this function simulates potentially highly non-linear data on which we will be testing
    our RMT stuff

    :param number_neurons_: we generate data y = \sum_{i=1}^K activation(features * beta + noise)
    each summand is a neuron. K = number_neurons #Mohammad: How are we summing this stuff? so we use different random
    activation functions of the same class?
    :param activation_: type of non-linearity  #Mohammad: Do we need to specify parameters for the functions we use?
    :param sample_size: sample size
    :param number_features_: number_features #Mohammad: Low dimensional features? That can become high dimensional
    through our activation functions
    :param beta_and_psi_link_: how eigenvalues of Sigma_beta and Psi are related #Mohammad: This the crucial
    parameter for our feature importance application
    :param noise_size_: size of noise
    :return: labels and features. Labels are noisy and potentially non-linear functions of features
    """
    np.random.seed(seed)
    psi_eigenvalues = np.abs(np.random.uniform(0.01, 1, [1, number_features_]))
    features = np.random.randn(sample_size, number_features_) * (psi_eigenvalues ** 0.5)

    beta_eigenvalues = psi_eigenvalues ** beta_and_psi_link_  # we should also experiment with non-monotonic links
    labels_ = np.zeros([sample_size, 1])
    beta_dict = dict()
    for neuron in range(number_neurons_):
        betas = np.random.randn(number_features_, 1) * (beta_eigenvalues ** 0.5).reshape(-1, 1)
        noise = np.random.randn(sample_size, 1) * noise_size_

        labels_ \
            += RandomFeaturesGenerator.apply_activation_to_multiplied_signals(
            multiplied_signals=features @ betas + noise,
            activation=activation_)
        beta_dict[neuron] = betas
    return labels_, features, beta_dict, psi_eigenvalues


def main(seed: int,
         full_sample_size: int,
         number_features: int,
         beta_and_psi_link: float,
         noise_size: float,
         activation: str,
         number_neurons: int,
         use_random_features: bool = True) -> dict:
    """
    Main simulation function
    :param seed:
    :param sample_size:
    :param number_features_:
    :param beta_and_psi_link_:
    :param noise_size_:
    :param activation_:
    :param number_neurons_:
    :return:
    """

    # TODO MOHAMMAD: When number_neurons = 1
    #  and noise = 0 and activation = linear and sample_size is large relative to number_features,
    #  then you should recover the true betas

    return regression_results


def run_experiment(seed: int,
                   sample_size: int,
                   number_features_: int,
                   beta_and_psi_link_: float,
                   noise_size_: float,
                   shrinkage_list: list,
                   activation_: str = 'linear',
                   number_neurons_: int = 1,
                   use_random_features: bool = False,

                   ):
    """

    :param shrinkage_list:
    :param seed:
    :param sample_size:
    :param number_features_:
    :param beta_and_psi_link_:
    :param noise_size_:
    :param activation_:
    :param number_neurons_:
    :param use_random_features:
    :return: Plot for comparison
    """

    labels, features, beta_dict = simulate_data(seed=seed,
                                                sample_size=sample_size,
                                                number_features_=number_features_,
                                                beta_and_psi_link_=beta_and_psi_link_,
                                                noise_size_=noise_size_,
                                                activation_=activation_,
                                                number_neurons_=number_neurons_)
    gamma = 1.
    number_random_features = 10000

    specification = {'distribution': 'normal',
                     'distribution_parameters': [0, gamma],
                     'activation': activation_,
                     'number_features': number_random_features,
                     'bias_distribution': None,
                     'bias_distribution_parameters': [0, gamma]}
    if use_random_features:
        random_features = RandomFeaturesGenerator.generate_random_neuron_features(
            features,
            seed + 10,
            **specification
        )
    else:
        random_features = features

    in_sample_period = int(sample_size / 2)

    regression_results = RandomFeatures.ridge_regression_single_underlying(
        signals=features[:in_sample_period, :],
        labels=labels[:in_sample_period],
        future_signals=features[in_sample_period:, :],
        shrinkage_list=shrinkage_list,
        use_msrr=False,
        return_in_sample_pred=True,
        compute_smart_weights=False,
        test_fixed_kappas=False,
        fixed_kappas=[],
        test_changing_kappas=False,
        constant_signal=None,
        print_time=False,
        return_beta=True,
        keep_only_big_beta=False,
        core_z_values=None,
        clip_bstar=10000)  # understand what this clip does

    plt.title(['Beta for different shrinkage and c=' + str(number_features_ / sample_size)])
    legend_list = []
    for i in range(len(shrinkage_list)):
        legend_list.append(['z = ' + str(round(shrinkage_list[i], 2))])
        plt.scatter(regression_results['betas'][i], beta_dict[0])

    plt.legend(legend_list)
    plt.xlabel('True Beta')
    plt.ylabel('Estimated Beta')
    plt.show()


if __name__ == '__main__':

    # run_experiment(seed=0,
    #                sample_size=100,
    #                number_features_=10000,
    #                beta_and_psi_link_=2,
    #                noise_size_=0,
    #                activation_='linear',
    #                number_neurons_=1,
    #                shrinkage_list=np.exp(np.arange(-10, 10, 5)).tolist())
    #
    # run_experiment(seed=0,
    #                sample_size=10000,
    #                number_features_=100,
    #                beta_and_psi_link_=2,
    #                noise_size_=0,
    #                activation_='linear',
    #                number_neurons_=1,
    #                shrinkage_list=np.exp(np.arange(-10, 10, 5)).tolist())

    # gamma = 1.
    # number_random_features = 10000
    # use_random_features: bool = False
    #
    # specification = {'distribution': 'normal',
    #                  'distribution_parameters': [0, gamma],
    #                  'activation': activation_,
    #                  'number_features': number_random_features,
    #                  'bias_distribution': None,
    #                  'bias_distribution_parameters': [0, gamma]}
    # if use_random_features:
    #     random_features = RandomFeaturesGenerator.generate_random_neuron_features(
    #         features,
    #         seed + 10,
    #         **specification
    #     )
    # else:
    #     random_features = features
    #
    # in_sample_period = int(sample_size / 2)
    #
    #
    # regression_results = RandomFeatures.ridge_regression_single_underlying(
    #     signals=features[:in_sample_period, :],
    #     labels=labels[:in_sample_period],
    #     future_signals=features[in_sample_period:, :],
    #     shrinkage_list=shrinkage_list,
    #     use_msrr=False,
    #     return_in_sample_pred=True,
    #     compute_smart_weights=False,
    #     test_fixed_kappas=False,
    #     fixed_kappas=[],
    #     test_changing_kappas=False,
    #     constant_signal=None,
    #     print_time=False,
    #     return_beta=True,
    #     keep_only_big_beta=False,
    #     core_z_values=None,
    #     clip_bstar=10000)  # understand what this clip does
    #
    # plt.title(['Beta for different shrinkage and c=' + str(number_features_/sample_size)])
    # legend_list = []
    # for i in range(len(shrinkage_list)):
    #     legend_list.append(['z = ' + str(round(shrinkage_list[i], 2))])
    #     plt.scatter(regression_results['betas'][i], beta_dict[0])
    #
    # plt.legend(legend_list)
    # plt.xlabel('True Beta')
    # plt.ylabel('Estimated Beta')
    # plt.show()

    # Implement leave two out estimators
    # For any matrix A independent of t_1 and t_2
    # E[R_{t_2+1} S_{t_1}'A S_{t_2} R_{t_1+1}]\ =\  \beta'\Psi A \Psi \beta\

    seed = 0
    sample_size = 100
    number_features_ = 10000
    beta_and_psi_link_ = 2
    noise_size_ = 0
    activation_ = 'linear'
    number_neurons_ = 1
    shrinkage_list = np.exp(np.arange(-10, 10, 5)).tolist()

    labels, features, beta_dict, psi_eigenvalues = simulate_data(seed=seed,
                                                sample_size=sample_size,
                                                number_features_=number_features_,
                                                beta_and_psi_link_=beta_and_psi_link_,
                                                noise_size_=noise_size_,
                                                activation_=activation_,
                                                number_neurons_=number_neurons_)
    A = np.eye(number_features_)
    estimator = 0
    num = 0
    for i in range(sample_size):
        for j in range(sample_size):
            if j != i:
                num += 1
                estimator += labels[i] * features[i] @ A @ features[j] * labels[j]

    estimator = estimator/num
    true_value = np.dot(beta_dict[0]**2, psi_eigenvalues **2)
