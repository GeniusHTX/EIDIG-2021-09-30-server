"""
This python file evaluates the discriminatory degree of models.
"""

import os

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
from tensorflow import keras
import sys, os

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_census_income
import generation_utilities


def ids_percentage(sample_round, num_gen, num_attribs, protected_attribs, constraint, model):
    """
    Compute the percentage of individual discriminatory instances with 95% confidence
    :param sample_round: the total times of sample
    :param num_gen: the total number of the generated samples
    :param num_attribs: the number of attributes of the sample
    :param protected_attribs: the indices of the protected attributes
    """
    print("我正在执行 ids_percentage.")
    statistics = np.empty(shape=(0,))
    for i in range(sample_round):
        print("The sample_round is {}.And I am {}".format(sample_round, i))
        gen_id = generation_utilities.purely_random(num_attribs, protected_attribs, constraint, model, num_gen)
        percentage = len(gen_id) / num_gen
        statistics = np.append(statistics, [percentage], axis=0)
    avg = np.average(statistics)
    std_dev = np.std(statistics)
    interval = 1.960 * std_dev / np.sqrt(sample_round)
    print('The percentage of individual discriminatory instances with .95 confidence:', avg, '±', interval)


# load models
adult_model = keras.models.load_model("../models_htx/original_models/adult_model.h5")
# german_model = keras.models.load_model("../models/original_models/german_model.h5")
# bank_model = keras.models.load_model("../models/original_models/bank_model.h5")

# adult_ADF_retrained_model = keras.models.load_model("../models/retrained_models/adult_ADF_retrained_model.h5")
adult_EIDIG_5_retrained_model = keras.models.load_model(
    "../models_htx/retrained_models/adult_EIDIG_5_retrained_model_no_majority_vote_0.02.h5")


# adult_EIDIG_INF_retrained_model = keras.models.load_model(
#     "../models/retrained_models/adult_EIDIG_INF_retrained_model.h5")


def measure_discrimination(sample_round, num_gen):
    """
    Measure the discrimination degree of models on each benchmark
    :param sample_round: the total times of sample
    :param num_gen: the total number of the generated samples
    """

    print(
        'Percentage of discriminatory instances for original model, model retrained with ADF, model retrained with '
        'EIDIG-5, and model retrained with EIDIG-IND, respectively:\n')

    for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0, 6]), ('C-a&g', [0, 7]),
                                         ('C-r&g', [6, 7])]:
        print(benchmark, ':')
        ids_percentage(sample_round, num_gen, len(pre_census_income.X[0]), protected_attribs,
                       pre_census_income.constraint, adult_model)
        # ids_percentage(sample_round, num_gen, len(pre_census_income.X[0]), protected_attribs,
        #                pre_census_income.constraint, adult_ADF_retrained_model)
        ids_percentage(sample_round, num_gen, len(pre_census_income.X[0]), protected_attribs,
                       pre_census_income.constraint, adult_EIDIG_5_retrained_model)
        # ids_percentage(sample_round, num_gen, len(pre_census_income.X[0]), protected_attribs,
        #                pre_census_income.constraint, adult_EIDIG_INF_retrained_model)


def measure_discrimination_specified_model(sample_round, num_gen, model):
    print('Percentage of discriminatory instances for model:\n')

    for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7])]:
        print(benchmark, ':')
        print("我正在进入函数 ids_percentage.")
        ids_percentage(sample_round, num_gen, len(pre_census_income.X[0]),
                       protected_attribs, pre_census_income.constraint, model)


def main(argv=None):
    sample_round = 100
    num_gen = 10000

    # 注意这里model的参数
    adult_EIDIG_5_retrained_model_no_majority_vote_2 = keras.models.load_model(
        "../models_htx/retrained_models/adult_EIDIG_5_retrained_model_no_majority_vote_0.02.h5")
    print("我正在进入函数：measure_discrimination_specified_model")
    measure_discrimination_specified_model(sample_round, num_gen, adult_EIDIG_5_retrained_model_no_majority_vote_2)


# 分为两部，第一步衡量的是 retrained_model，注意应该是利用 formal数据训练出来的
if __name__ == '__main__':
    main()

# reproduce the results reported by our paper
# measure_discrimination(100, 10000)

# just for test
# measure_discrimination(10, 100)
