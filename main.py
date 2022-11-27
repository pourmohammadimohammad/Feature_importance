import numpy as np
import pandas as pd
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from helpers.random_features import RandomFeatures
import matplotlib.pyplot as plt




def run_experiment(seed: int,
                   sample_size: int,
                   number_features_: int,
                   beta_and_psi_link_: float,
                   noise_size_: float,
                   shrinkage_list: list,
                   activation_: str = 'linear',
                   number_neurons_: int = 1,
                   use_random_features: bool = False):
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

    labels, features, beta_dict, psi_eigenvalues = simulate_data(seed=seed,
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
    shrinkage_list = np.linspace(0.1, 10, 100)


    seed = 0
    run_experiment(seed=seed,
                   sample_size=full_sample_size,
                   shrinkage_list=shrinkage_list,
                   number_features_=number_features,
                   number_neurons_=number_neurons,
                   activation_=activation,
                   noise_size=noise_size,
                   beta_and_psi_link=beta_and_psi_link)
