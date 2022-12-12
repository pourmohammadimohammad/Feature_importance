import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from parameters import *
import pickle
import sys
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator


class RandomData:
    def __init__(self, par: Params):
        self.par = par
        self.raw_dir = Constant.BASE_DIR + 'data/raw/'
        self.p_dir = Constant.BASE_DIR + 'data/pickle/'
        os.makedirs(self.p_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)

    def gen_random_data(self, create=False):
        if create:
            labels, features = RandomData.simulate_data(seed=self.par.simulated_data.seed,
                                                        sample_size=self.par.simulated_data.t,
                                                        number_features_=int(self.par.simulated_data.t*self.par.simulated_data.c),
                                                        beta_and_psi_link_=self.par.simulated_data.beta_and_psi_link,
                                                        noise_size_=self.par.simulated_data.noise_size,
                                                        activation_=self.par.simulated_data.activation,
                                                        number_neurons_=self.par.simulated_data.number_neurons,
                                                        simple_beta=self.par.simulated_data.simple_beta,
                                                        alpha=self.par.simulated_data.alpha,
                                                        b_star=self.par.simulated_data.b_star)

            np.save(arr=labels, file=self.p_dir + f'labels_{self.par.simulated_data.get_name()}')
            np.save(arr=features, file=self.p_dir + f'features_{self.par.simulated_data.get_name()}')
            print('Data Processed!')

        else:
            labels = np.load(file=self.p_dir + f'labels_{self.par.simulated_data.get_name()}.npy')
            features = np.load(file=self.p_dir + f'features_{self.par.simulated_data.get_name()}.npy')
            print('Data Loaded!')

        return labels, features

    @staticmethod
    def simulate_data(seed: int,
                      sample_size: int,
                      number_features_: int,
                      beta_and_psi_link_: float,
                      noise_size_: float,
                      activation_: str = 'linear',
                      number_neurons_: int = 1,
                      simple_beta: bool = False,
                      alpha: float = 1,
                      b_star: float = 0.01,
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

        beta_eigenvalues = psi_eigenvalues ** beta_and_psi_link_ / np.sqrt(number_features_)

        :param noise_size_: size of noise
        :return: labels and features. Labels are noisy and potentially non-linear functions of features
        """
        np.random.seed(100)
        psi_eigenvalues = np.abs(np.random.uniform(0.01, 1, [1, number_features_])) ** alpha
        psi_l = psi_eigenvalues ** beta_and_psi_link_

        # we should also experiment with non-monotonic links
        if simple_beta:
            beta_eigenvalues = b_star * np.ones([1, number_features_]) / number_features_
        else:
            beta_eigenvalues = b_star * psi_l / np.sum(psi_l)

        beta_dict = dict()
        for neuron in range(number_neurons_):
            betas = np.random.randn(number_features_, 1) * (beta_eigenvalues ** 0.5).reshape(-1, 1)
            beta_dict[neuron] = betas

        np.random.seed(seed)
        features = np.random.randn(sample_size, number_features_) * (psi_eigenvalues ** 0.5)

        labels_ = np.zeros([sample_size, 1])
        for neuron in range(number_neurons_):
            noise = np.random.randn(sample_size, 1) * noise_size_
            labels_ \
                += RandomFeaturesGenerator.apply_activation_to_multiplied_signals(
                multiplied_signals=features @ betas + noise,
                activation=activation_)

        return labels_, features
