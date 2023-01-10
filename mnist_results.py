import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from parameters import *
from tqdm import tqdm
from gen_random_data import RandomData
import sys
from leave_out import *


def voc_plots(experiments, title):
    models = experiments[title]
    c_list = [m.c for m in models]
    ax_legend = ['Optimal', 'Inf Opt']
    optimal_mse = [m.oos_optimal_mse for m in models]
    inf_mse = [m.infeasible_oos_optimal_mse for m in models]
    plt.scatter(c_list, optimal_mse)
    plt.scatter(c_list, inf_mse)
    plt.legend(ax_legend)
    plt.xlabel('c')
    plt.ylabel('MSE')
    plt.title(title)
    plt.savefig('plots/MSE' + title.replace(" ", "") + '.jpg')
    plt.close()

    optimal_sharpe = [m.oos_optimal_sharpe for m in models]
    inf_sharpe = [m.infeasible_oos_optimal_sharpe for m in models]
    plt.scatter(c_list, optimal_sharpe)
    plt.scatter(c_list, inf_sharpe)
    plt.legend(ax_legend)
    plt.xlabel('c')
    plt.ylabel('Sharpe')
    plt.title(title)
    plt.savefig('plots/Sharpe' + title.replace(" ", "") + '.jpg')
    plt.close()

    print('Done')


def optimal_vs_oos_simple_plot(experiments, exp_type,g):
    # needs to be changed to acomodate new efficient par class
    models = experiments[exp_type]
    c_list = [np.round(model.par.data.c, 2) for model in models]
    s = sorted(c_list)
    my_order = [s.index(x) for x in c_list]
    models = [models[i] for i in my_order]
    ax_legend = []
    [ax_legend.append(f'c = {np.round(model.par.data.c , 2)} OOS') for model in
     models]
    [ax_legend.append(f'c = {np.round(model.par.data.c , 2)} INS') for model
     in models]
    [ax_legend.append(f'c = {np.round(model.par.data.c , 2)} Optimal') for model
     in models]
    [ax_legend.append(f'c = {np.round(model.par.data.c , 2)} Inf Opt') for model
     in models]
    shrinkage_list = models[0].par.plo.shrinkage_list
    ones = np.ones(models[0].par.plo.shrinkage_list.shape)
    colors = []
    [colors.append(f'C{i}') for i in range(len(models))]

    [plt.plot(shrinkage_list, models[i].oos_perf_est['mse'], color=colors[i])
     for i in range(len(models))]
    [plt.plot(shrinkage_list, models[i].ins_perf_est['mse'], color=colors[i], linestyle='dotted')
     for i in range(len(models))]


    [plt.plot(shrinkage_list, ones * models[i].oos_optimal_mse
              , color=colors[i], linestyle='dashed')
     for i in range(len(models))]
    [plt.plot(shrinkage_list, ones * models[i].infeasible_oos_optimal_mse
              , color=colors[i], linestyle='dashdot')
     for i in range(len(models))]

    plt.title(f'g = {round(g,2)}')
    plt.legend(ax_legend, loc='upper right')
    plt.xlabel('z')
    plt.ylabel('MSE')
    plt.xscale('log')
    if len(models) == 1:
        plt.savefig('plots/' + exp_type + '.jpg')
    else:
        plt.savefig('plots/' + exp_type + '.jpg')

    plt.show()


def ins_vs_oos_weights_plot(experiments, exp_type):
    models = experiments[exp_type]  # fix this redundent line
    c_list = [np.round(model.par.simulated_data.c / model.par.plo.train_frac, 2) for model in models]
    s = sorted(c_list)
    my_order = [s.index(x) for x in c_list]
    models = [models[i] for i in my_order]
    ax_legend = []
    [ax_legend.append(f'c = {np.round(model.par.simulated_data.c / model.par.plo.train_frac, 2)} OOS') for model in
     models]
    [ax_legend.append(f'c = {np.round(model.par.simulated_data.c / model.par.plo.train_frac, 2)} ins') for model
     in models]
    colors = []
    [colors.append(f'C{i}') for i in range(len(models))]
    shrinkage_list = models[0].par.plo.shrinkage_list

    [plt.plot(shrinkage_list, models[i].w_sharpe_oos, color=colors[i]) for i in range(len(models))]

    [plt.plot(shrinkage_list, models[i].w_sharpe_ins, color=colors[i], linestyle='dashed') for i in range(len(models))]

    plt.title(exp_type)
    plt.legend(ax_legend, loc='upper right')
    plt.xlabel('z')
    plt.ylabel('weights sharpe')
    if len(models) == 1:
        plt.savefig('plots/weights_sharpe' + exp_type.replace(" ", "") + str(models[0].par.simulated_data.c) + '.jpg')
    else:
        plt.savefig('plots/weights_sharpe' + exp_type.replace(" ", "") + '.jpg')

    plt.show()

    [plt.plot(shrinkage_list, models[i].w_mse_oos, color=colors[i]) for i in range(len(models))]

    [plt.plot(shrinkage_list, models[i].w_mse_ins, color=colors[i], linestyle='dashed') for i in range(len(models))]

    plt.title(exp_type)
    plt.legend(ax_legend, loc='upper right')
    plt.xlabel('z')
    plt.ylabel('weights mse')
    if len(models) == 1:
        plt.savefig('plots/weights_mse' + exp_type.replace(" ", "") + str(models[0].par.simulated_data.c) + '.jpg')
    else:
        plt.savefig('plots/weights_mse' + exp_type.replace(" ", "") + '.jpg')
    plt.show()


def mean_comparison(loo: LeaveOut):
    c = np.round(loo.par.simulated_data.c / loo.par.plo.train_frac, 2)
    ax_legend = [f'INS {c}', f'OOS {c}', f'True {c}']
    plt.plot(loo.par.plo.shrinkage_list, loo.ins_perf_est['mean'])
    plt.plot(loo.par.plo.shrinkage_list, loo.oos_perf_est['mean'])
    plt.plot(loo.par.plo.shrinkage_list, loo.mean_true)
    plt.legend(ax_legend)
    plt.xlabel('z')
    plt.ylabel('mean')
    plt.title(loo.par.experiment_title)
    plt.savefig('plots/m' + loo.par.experiment_title.replace(" ", "") + str(loo.par.simulated_data.c) + '.jpg')
    plt.close()


def ins_vs_oos_m_plot(experiments, exp_type, z_list):
    models = experiments[exp_type]  # fix this redundent line
    c_list = [np.round(model.par.simulated_data.c / model.par.plo.train_frac, 2) for model in models]
    s = sorted(c_list)
    my_order = [s.index(x) for x in c_list]
    models = [models[i] for i in my_order]
    ax_legend = []
    [ax_legend.append(f'c = {np.round(model.par.simulated_data.c / model.par.plo.train_frac, 2)} OOS') for model in
     models]
    [ax_legend.append(f'c = {np.round(model.par.simulated_data.c / model.par.plo.train_frac, 2)} ins') for model
     in models]
    colors = []
    [colors.append(f'C{i}') for i in range(len(models))]
    shrinkage_list = models[0].par.plo.shrinkage_list

    k = len(z_list)
    fig, ax = plt.subplots(1, k, sharex=True, figsize=(8 * k, 8))

    for j in range(k):
        [ax[j].plot(shrinkage_list, models[i].oos_m_sharpe[z_list[j], :],
                    color=colors[i]) for i in range(len(models))]
        [ax[j].plot(shrinkage_list, models[i].ins_m_sharpe[z_list[j], :],
                    color=colors[i], linestyle='dashed') for i in range(len(models))]
        ax[j].set_title(f'M_z{z_list[j]}')

    ax[0].legend(ax_legend, loc='upper right')
    fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
    fig.suptitle(exp_type, fontsize=12)

    if len(models) == 1:
        plt.savefig('plots/m' + exp_type.replace(" ", "") + str(models[0].par.simulated_data.c) + '.jpg')
    else:
        plt.savefig('plots/m' + exp_type.replace(" ", "") + '.jpg')
    plt.close()


def load_stuff(folder_to_check, f):
    par = Params()
    par.load(f'res/{folder_to_check}/{f}/')
    loo = LeaveOut()
    loo.load(f'res/{folder_to_check}/{f}/')
    loo.par = par
    return loo


def plot_stuff(f):
    if 'leave_out.p' in os.listdir(f'res/{folder_to_check}/{f}/'):
        loo = load_stuff(folder_to_check, f)
        loo.par.update_mnist_basic_title()
        experiments = {loo.par.experiment_title: [loo]}
        optimal_vs_oos_simple_plot(experiments, loo.par.experiment_title, loo.par.plo.g)
        # ins_vs_oos_weights_plot(experiments, loo.par.name)
        # ins_vs_oos_m_plot(experiments, loo.par.experiment_title, z_list)


if __name__ == "__main__":

    folder_to_check = 'mnist_first_test'
    big_data = True
    os.makedirs('plots', exist_ok=True)
    gl = os.listdir(f'res/{folder_to_check}/')
    max_iter = len(gl)
    counter = 0
    z_list = [0, 5, 20, 50, 75]
    print('start')
    if big_data:
        try:

            grid_id = int(sys.argv[1])
        except:
            print('Debug mode on local machine')
            grid_id = -1

        if grid_id < 0:
            while counter < max_iter:
                f = gl[counter]
                plot_stuff(f)
                counter += 1
        else:
            f = gl[grid_id]
            plot_stuff(f)


    else:

        experiments = {}
        for f in os.listdir(f'res/{folder_to_check}/'):
            if 'leave_out.p' in os.listdir(f'res/{folder_to_check}/{f}/'):
                loo = load_stuff(folder_to_check, f)
                loo.par.update_model_name()
                per = Performance()
                per.copy_performance(loo)
                if loo.par.experiment_title in experiments.keys():
                    experiments[loo.par.experiment_title].append(per)
                else:
                    experiments[loo.par.experiment_title] = [per]

        [voc_plots(experiments, exp_type) for exp_type in experiments]
