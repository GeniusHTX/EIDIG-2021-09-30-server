# -*- coding:utf-8 -*-
# @Time  : 2021/9/15 15:21
# @Author: Weisong Sun
# @File  : experiments.py

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import generation_utilities
import time
import pickle
import joblib
from _htx import EIDIG_htx
from preprocessing import pre_census_income
from _htx import pre_census_income_htx


def generate_dis_instance(num_experiment_round, benchmark, X, protected_attribs, constraint,
                          model, g_num=1000, l_num=1000, decay=0.5, c_num=4, max_iter=10,
                          s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    num_ids = np.array([0] * 3)
    time_cost = np.array([0] * 3)

    result_directory = '../logging_data_htx/generated_dis_instances/'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    for j in range(num_experiment_round):
        round_now = j + 1
        print("The num_experiment_round is {}.And I am {}".format(num_experiment_round, j))
        print('--- ROUND', round_now, '---')
        if g_num >= len(X):
            seeds = X.copy()
        else:
            clustered_data = generation_utilities.clustering(X, c_num)
            seeds = np.empty(shape=(0, len(X[0])))
            for i in range(g_num):
                # print("The g_num is {}.And I am {}".format(g_num,i))
                # 这个部分没有问题，进去之后，完全可以出来
                new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i % c_num, fashion=fashion)
                seeds = np.append(seeds, [new_seed], axis=0)

        # 问题在于下面这个函数。似乎进去了。就出不来了。
        ids_EIDIG_5, gen_EIDIG_5, total_iter_EIDIG_5, R = EIDIG_htx.individual_discrimination_generation(X, seeds,
                                                                                                         protected_attribs,
                                                                                                         constraint,
                                                                                                         model,
                                                                                                         decay, l_num,
                                                                                                         5,
                                                                                                         max_iter, s_g,
                                                                                                         s_l,
                                                                                                         epsilon_l)

        # np.save(result_directory + benchmark + '_ids_EIDIG_5_' + str(round_now) + '.npy', ids_EIDIG_5)
        # t2 = time.time()
        # print('EIDIG-5:', 'In', total_iter_EIDIG_5, 'search iterations', len(gen_EIDIG_5),
        #       'non-duplicate instances are explored', len(ids_EIDIG_5),
        #       'of which are discriminatory. Time cost:', t2 - t1, 's.')
        # num_ids[0] += len(ids_EIDIG_5)
        # time_cost[0] += t2 - t1

        filename = result_directory + benchmark + '_ids_EIDIG_5_' + str(round_now) + '.pkl'
        print("我正在导出 pkl 文件。")
        filehandler = open(filename, 'wb')
        pickle.dump(R, filehandler)
        filehandler.close()

        print('\n')

    avg_num_ids = num_ids / num_experiment_round
    avg_speed = num_ids / time_cost
    print('Results of complete comparison on', benchmark,
          'with g_num set to {} and l_num set to {}'.format(g_num, l_num),
          ',averaged on', num_experiment_round, 'rounds:')
    print('EIDIG-5', ':', avg_num_ids[0], 'individual discriminatory instances are generated at a speed of',
          avg_speed[0], 'per second.')


def produce_label(dataset_name):
    ensemble_clf = joblib.load('../models/ensemble_models/' + dataset_name + '_ensemble.pkl')
    protected_attribs = pre_census_income_htx.protected_attribs

    ids_C_a_EIDIG_5 = '../logging_data_htx/generated_dis_instances/C-a_ids_EIDIG_5_1.pkl'

    file = open(ids_C_a_EIDIG_5, 'rb')
    R = pickle.load(file)
    X = pre_census_income_htx.get_all_data()
    # print(X.shape)
    for r in R:
        seed = r.seed[0]
        # print(seed.shape)
        # produce seed's label
        # X_train = pre_census_income.X_train() 不能用这个，因为 X_train 每次都 random 分出来都
        for x in X:
            x_without_label = x[:-1]
            if (x_without_label == seed).all():
                seed_label = x[-1]
        print("seed: " + str(seed) + " label: " + str(seed_label))

        # produce the global discriminatory instances' label
        g_dis_ins = r.g_dis_ins
        num_g_dis_ins = len(g_dis_ins)
        if num_g_dis_ins > 0:
            g_dis_ins_label_vote = ensemble_clf.predict(np.delete(g_dis_ins, protected_attribs, axis=1))
            print("g_dis: " + str(g_dis_ins[0]) + " label: " + str(g_dis_ins_label_vote[0]))

        # produce the local discriminatory instances' label
        l_dis_ins = r.l_dis_ins
        num_l_dis_ins = len(l_dis_ins)
        if num_l_dis_ins > 0:
            # print("the number of local discriminatory instances: " + str(len(l_dis_ins)))
            l_dis_ins_label_vote = ensemble_clf.predict(np.delete(l_dis_ins, protected_attribs, axis=1))
            for i in range(num_l_dis_ins):
                print("l_dis: " + str(l_dis_ins[i]) + " label: " + str(l_dis_ins_label_vote[i]))
        print("---------------------------------------------------------------")
    # end for R

    file.close()


def main(argv=None):
    for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7])]:
        print('\n', benchmark, ':\n')
        ROUND = 1
        # load models
        adult_model = keras.models.load_model("../models_htx/original_models/adult_model.h5")

        # 这个部分目前有大问题
        generate_dis_instance(ROUND, benchmark, pre_census_income.X_train,
                              protected_attribs, pre_census_income.constraint,
                              adult_model)
    dataset_name = "adult"
    # 这个部分的 produce label 是不能要的，因为 这个函数的 produce
    # label 是通过 分类器投票得到的，因此我们不需要它。
    # produce_label(dataset_name)


if __name__ == '__main__':
    main()
