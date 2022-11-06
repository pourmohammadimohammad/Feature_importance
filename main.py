import numpy as np
import pandas as pd
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from helpers.random_features import RandomFeatures


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
    each summand is a neuron. K = number_neurons
    :param activation_: type of non-linearity
    :param sample_size: sample size
    :param number_features_: number_features
    :param beta_and_psi_link_: how eigenvalues of Sigma_beta and Psi are related
    :param noise_size_: size of noise
    :return: labels and features. Labels are noisy functions of features
    """
    np.random.seed(seed)
    psi_eigenvalues = np.abs(np.random.uniform(0.01, 1, [1, number_features_]))
    features = np.random.randn(sample_size, number_features_) * (psi_eigenvalues ** 0.5)

    beta_eigenvalues = psi_eigenvalues ** beta_and_psi_link_  # we should also experiment with non-monotonic links
    labels_ = np.zeros([sample_size, 1])
    for neuron in range(number_neurons_):
        betas = np.random.randn(number_features_, 1) * (beta_eigenvalues ** 0.5).reshape(-1, 1)
        noise = np.random.randn(sample_size, 1) * noise_size_

        labels_ \
            += RandomFeaturesGenerator.apply_activation_to_multiplied_signals(
            multiplied_signals=features @ betas + noise,
            activation=activation_)
    return labels_, features


def main(seed: int,
         full_sample_size: int,
         number_features: int,
         beta_and_psi_link: float,
         noise_size: float,
         activation: str,
         number_neurons: int) -> dict:
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
    labels, features = simulate_data(seed=seed,
                                     sample_size=full_sample_size,
                                     number_features_=number_features,
                                     beta_and_psi_link_=beta_and_psi_link,
                                     noise_size_=noise_size,
                                     activation_=activation,
                                     number_neurons_=number_neurons)

    gamma = 1.
    number_random_features = 10000
    specification = {'distribution': 'normal',
                     'distribution_parameters': [0, gamma],
                     'activation': activation,
                     'number_features': number_random_features,
                     'bias_distribution': None,
                     'bias_distribution_parameters': [0, gamma]}

    random_features = RandomFeaturesGenerator.generate_random_neuron_features(
        features,
        seed + 10,
        **specification
    )

    in_sample_period = int(full_sample_size / 2)

    shrinkage_list = np.exp(np.arange(-10, 20, 1)).tolist()

    regression_results = RandomFeatures.ridge_regression_single_underlying(
        signals=random_features[:in_sample_period, :],
        labels=labels[:in_sample_period],
        future_signals=random_features[in_sample_period:, :],
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
        clip_bstar=10000)

    return regression_results


if __name__ == '__main__':
    # export PYTHONPATH="${PYTHONPATH}:/Users/malamud/Dropbox/MY_STUFF/TRADING/virtueofcomplexityeverywhere"
    full_sample_size = 2000
    number_features = 100
    beta_and_psi_link = 0.
    noise_size = 0.

    """
    activation = linear and number_neurons = 1 corresponds to an exact linear model. 
    """
    activation = 'linear'
    number_neurons = 1

    seed = 0
