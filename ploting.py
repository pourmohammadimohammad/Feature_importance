import numpy
import numpy as np
import pandas as pd
from leaveout import *
from main import *
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from helpers.random_features import RandomFeatures
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers.marcenko_pastur import MarcenkoPastur
from parameters import *
from wip import *

def plot_sub_plots(estimators, shrinkage_list, ax_legend, c, plot_name):
    times = list(estimators.keys())
    num_seeds = len(estimators[times[0]])
    for name in estimators[times[0]][0].keys():
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
        for i in range(len(times)):
            a_0 = i % 2
            a_1 = int((i - a_0) / 2) % 2
            [ax[a_0, a_1].plot(shrinkage_list, estimators[times[i]][j][name])
             for j in range(num_seeds)]
            ax[a_0, a_1].set_title(f'T = {times[i]}')

        ax[0, 0].legend(ax_legend, loc='upper right')
        fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
        fig.suptitle(name.upper() + ' ' + plot_name + f" \n  c = {c}", fontsize=12)
        plt.show()


def plot_sub_plots_mean(estimators, shrinkage_list, ax_legend, c, train_frac):
    times = list(estimators.keys())
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
    for i in range(len(times)):
        a_0 = i % 2
        a_1 = int((i - a_0) / 2) % 2
        [ax[a_0, a_1].plot(shrinkage_list, estimators[times[i]][j])
         for j in range(len(estimators[times[i]]))]

        ax[a_0, a_1].set_title(f'T = {times[i]}')

    ax[0, 0].legend(ax_legend, loc='upper right')
    fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
    fig.suptitle(' Mean' + f" \n  c = {c} \n train fraction = {train_frac} ", fontsize=12)
    plt.show()


def plot_sub_plots_ins_vs_oos(estimators, estimators_oos, shrinkage_list, ax_legend, c, train_frac, plot_name):
    times = list(estimators.keys())
    num_seeds = len(estimators[times[0]])
    for name in estimators[times[0]][0].keys():
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
        for i in range(len(times)):
            a_0 = i % 2
            a_1 = int((i - a_0) / 2) % 2
            [ax[a_0, a_1].plot(shrinkage_list, estimators[times[i]][j][name])
             for j in range(num_seeds)]

            [ax[a_0, a_1].plot(shrinkage_list, estimators_oos[times[i]][j][name])
             for j in range(num_seeds)]
            ax[a_0, a_1].set_title(f'T = {times[i]}')

        ax[0, 0].legend(ax_legend, loc='upper right')
        fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
        fig.suptitle(name.upper() + ' ' + plot_name + f" \n  c = {c} \n train fraction = {train_frac} ", fontsize=12)
        plt.show()


def plot_unconditionally(estimators_oos, estimators_ins, true_values, shrinkage_list, c, times,
                         name='mean'):
    avg_perf_OOS = {}
    avg_perf_INS = {}
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
    for i in range(len(times)):

        avg_perf_OOS[times[i]] = np.zeros(len(estimators_oos[times[i]][0][name]))
        avg_perf_INS[times[i]] = np.zeros(len(estimators_ins[times[i]][0][name]))
        for j in range(len(seeds)):
            avg_perf_OOS[times[i]] = avg_perf_OOS[times[i]] + np.array(estimators_oos[times[i]][j][name]) / len(seeds)
            avg_perf_INS[times[i]] = avg_perf_INS[times[i]] + np.array(estimators_ins[times[i]][j][name]) / len(seeds)

        a_0 = i % 2
        a_1 = int((i - a_0) / 2) % 2

        ax[a_0, a_1].plot(shrinkage_list, true_values[times[i]])
        ax[a_0, a_1].plot(shrinkage_list, avg_perf_OOS[times[i]])
        ax[a_0, a_1].plot(shrinkage_list, avg_perf_INS[times[i]])
        ax[a_0, a_1].set_title(f'T = {times[i]}')

    ax[0, 0].legend([' INS ', ' OOS ', 'Theoretical '], loc='upper right')
    fig.suptitle(f'c = {c}')
    fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
    plt.show()


def expanding_window_experment(t: int,
                               c: float,
                               train_frac: float,
                               beta_and_psi_link: float,
                               shrinkage_list: np.ndarray,
                               seeds: list):
    estimators_oos = []
    estimated_values_saved = []
    [estimated_values_saved.append(run_loo(t=t,
                                           c=c,
                                           train_frac=train_frac,
                                           beta_and_psi_link=beta_and_psi_link,
                                           shrinkage_list=shrinkage_list,
                                           seed=s,
                                           growing_oos=True)) for s in seeds]
    [estimators_oos.append(estimated_values_saved[j][1]) for j in range(len(seeds))]
    # true_values = estimated_values_saved[0][2].theoretical_mean()
    ax_legend = []
    [ax_legend.append(f'Seed {s}') for s in seeds]
    name = 'mean'
    times = estimated_values_saved[0][2].times
    num_seeds = len(estimators_oos)
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
    for i in range(len(times)):
        a_0 = i % 2
        a_1 = int((i - a_0) / 2) % 2
        [ax[a_0, a_1].plot(shrinkage_list, estimators_oos[j][name][i])
         for j in range(num_seeds)]

        ax[a_0, a_1].set_title(f'T_1 = {times[i]}')

    ax[0, 0].legend(ax_legend, loc='upper right')
    fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
    fig.suptitle(name.upper() + ' Out of Sample', fontsize=12)
    plt.show()

    return estimators_oos


def ins_vs_oos_experiment(times: list,
                          complexity: list,
                          train_frac: float,
                          beta_and_psi_link: float,
                          shrinkage_list: np.ndarray):
    ax_legend = []
    [ax_legend.extend([f'Complexity = {c} INS', f'Complexity = {c} OOS']) for c in complexity]

    estimators = {}
    for i in range(len(times)):
        estimators[times[i]] = []
        [estimators[times[i]].append(run_loo(t=times[i],
                                             c=c,
                                             train_frac=train_frac,
                                             beta_and_psi_link=beta_and_psi_link,
                                             shrinkage_list=shrinkage_list)[0:2]) for c in complexity]

    for name in estimators[times[0]][0][0].keys():
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
        for i in range(len(times)):
            a_0 = i % 2
            a_1 = int((i - a_0) / 2) % 2
            [ax[a_0, a_1].plot(shrinkage_list, estimators[times[i]][j][0][name])
             for j in range(len(complexity))]

            [ax[a_0, a_1].plot(shrinkage_list, estimators[times[i]][j][1][name])
             for j in range(len(complexity))]

            ax[a_0, a_1].set_title(f'T = {times[i]}')

        ax[0, 0].legend(ax_legend, loc='upper right')
        fig.text(0.5, 0.04, 'Shrinkage Size', ha='center', fontsize=12)
        fig.suptitle(name.upper() + f" \n  beta-psi link= {beta_and_psi_link}", fontsize=12)
        plt.show()
        return estimators


def complete_comparison_experiment(t: int,
                                   c: float,
                                   train_frac: float,
                                   beta_and_psi_link: float,
                                   shrinkage_list: np.ndarray):
    mean_dict = {}
    ins, oos, loo = run_loo(t=t, c=c,
                            train_frac=train_frac,
                            beta_and_psi_link=beta_and_psi_link,
                            shrinkage_list=shrinkage_list,
                            simple_beta=True)

    mean_dict['INS'] = ins['mean']
    mean_dict['OOS'] = oos['mean']

    mean_dict['Theoretical'] = loo.theoretical_mean()

    loo.True_value_eq_176(DataUsed.INS)
    mean_dict['eq 176 INS'] = loo.true_value_sigma_beta_eq_176
    mean_dict['eq 176 limit INS'] = loo.true_value_beta_eq_176

    loo.True_value_eq_176(DataUsed.OOS)
    mean_dict['eq 176 OOS'] = loo.true_value_sigma_beta_eq_176
    mean_dict['eq 176 limit OOS'] = loo.true_value_beta_eq_176

    return


def simulate_across_seeds(t: int,
                          c: float,
                          train_frac: float,
                          beta_and_psi_link: float,
                          shrinkage_list: np.ndarray,
                          seeds: list):
    estimators_oos = []
    estimators_ins = []
    estimated_values_saved = []
    [estimated_values_saved.append(run_loo(t=t, c=c, train_frac=train_frac,
                                           beta_and_psi_link=beta_and_psi_link,
                                           shrinkage_list=shrinkage_list,
                                           seed=s,
                                           simple_beta=True)) for s in seeds]
    [estimators_ins.append(estimated_values_saved[j][0]) for j in range(len(seeds))]
    [estimators_oos.append(estimated_values_saved[j][1]) for j in range(len(seeds))]
    true_values = estimated_values_saved[0][2].theoretical_mean()

    return estimators_ins, estimators_oos, true_values


def across_seeds_experiment(times: list,
                            c: float,
                            train_frac: float,
                            beta_and_psi_link: float,
                            shrinkage_list: np.ndarray,
                            seeds: list):
    estimators_oos = {}
    estimators_ins = {}
    true_values = {}

    for t in times:
        estimators_oos[t], estimators_ins[t], true_values[t] = simulate_across_seeds(t,
                                                                                     c,
                                                                                     train_frac,
                                                                                     beta_and_psi_link,
                                                                                     shrinkage_list,
                                                                                     seeds)
    seeds_legend =[]
    [seeds_legend.append(f'seed {s}') for s in seeds]

    plot_unconditionally(estimators_oos=estimators_oos,
                         estimators_ins=estimators_ins,
                         true_values=true_values,
                         shrinkage_list=shrinkage_list,
                         c=c)

    plot_sub_plots(estimators=estimators_oos,
                   shrinkage_list=shrinkage_list,
                   ax_legend=seeds_legend,
                   c=c,
                   plot_name='Out of Sample')

    plot_sub_plots(estimators=estimators_ins,
                   shrinkage_list=shrinkage_list,
                   ax_legend=seeds_legend,
                   c=c,
                   plot_name='In Sample')

