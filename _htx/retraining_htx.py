"""
This python file retrains the original models with augmented training set.
"""

import os
import pickle
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from _htx.EIDIG_htx import DisInstanceResult

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing
from training import train_census_income
from training import train_german_credit
from training import train_bank_marketing

from _htx import pre_census_income_htx


def remove_informal_dis_ins(dataset_name, num_attribs, benchmark, round_now):
    ensemble_clf = joblib.load('../models/ensemble_models/' + dataset_name + '_ensemble.pkl')
    protected_attribs = pre_census_income_htx.protected_attribs

    ids_C_a_EIDIG_5_pkl = '../logging_data_htx/generated_dis_instances_back_up/C-a_ids_EIDIG_5_1.pkl'
    ids_C_g_EIDIG_5_pkl = '../logging_data_htx/generated_dis_instances_back_up/C-g_ids_EIDIG_5_1.pkl'
    ids_C_r_EIDIG_5_pkl = '../logging_data_htx/generated_dis_instances_back_up/C-r_ids_EIDIG_5_1.pkl'

    ids_benchmark_EIDIG_5 = ''
    if benchmark == 'C-a':
        ids_benchmark_EIDIG_5 = ids_C_a_EIDIG_5_pkl
    if benchmark == 'C-g':
        ids_benchmark_EIDIG_5 = ids_C_g_EIDIG_5_pkl
    if benchmark == 'C-r':
        ids_benchmark_EIDIG_5 = ids_C_r_EIDIG_5_pkl

    file = open(ids_benchmark_EIDIG_5, 'rb')
    R = pickle.load(file)
    X = pre_census_income_htx.get_all_data()
    N_R = []
    for r in R:
        seed = r.seed[0]
        # produce seed's label
        for x in X:
            x_without_label = x[:-1]
            if (x_without_label == seed).all():
                seed_label = x[-1]
                print("seed label: " + str(seed_label))

        flag_1 = False  # 用于记录 global 是否与 seed 相同
        flag_2 = False  # 用于记录 global or local discriminatory instances 的 labels 是否与 seed 一致
        # produce the global discriminatory instances' label
        g_dis_ins = r.g_dis_ins
        num_g_dis_ins = len(g_dis_ins)
        if num_g_dis_ins > 0:
            if (g_dis_ins[0] == seed).all():
                # 排除 global = seed
                flag_1 = True
            else:
                g_dis_ins_label_vote = ensemble_clf.predict(np.delete(g_dis_ins, protected_attribs, axis=1))
                if g_dis_ins_label_vote[0] != seed_label:
                    flag_2 = True
        if flag_1 or flag_2:
            continue

        new_l_dis_ins = np.empty(shape=(0, num_attribs))  # 用于存储与 seed label 一致的 local discriminatory instances
        # produce the local discriminatory instances' label
        l_dis_ins = r.l_dis_ins
        num_l_dis_ins = len(l_dis_ins)
        if num_l_dis_ins > 0:
            l_dis_ins_label_vote = ensemble_clf.predict(np.delete(l_dis_ins, protected_attribs, axis=1))
            for i in range(num_l_dis_ins):
                if l_dis_ins_label_vote[i] != seed_label:
                    continue
                new_l_dis_ins = np.append(new_l_dis_ins, [l_dis_ins[i]], axis=0)

        new_r = DisInstanceResult(num_attribs, r.seed.copy(), r.g_dis_ins.copy())
        new_r.set_l_dis_ins(new_l_dis_ins)
        N_R.append(new_r)
    # end for R

    file.close()

    result_directory = '../logging_data_htx/generated_dis_instances_back_up/'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    filename = result_directory + benchmark + '_ids_EIDIG_5_' + str(round_now) + '_formal.pkl'
    filehandler = open(filename, 'wb')
    pickle.dump(N_R, filehandler)
    filehandler.close()


def patch_remove_informal_dis_ins(dataset_name):
    for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7])]:
        print('\n', benchmark, ':\n')
        ROUND = 1
        num_attribs = 12
        remove_informal_dis_ins(dataset_name, num_attribs, benchmark, ROUND)


def retraining_without_majority_voting(dataset_name, approach_name, percent):
    print("我已经进入 retraining_without_majority_voting 函数")
    if dataset_name == 'adult':
        X_train = pre_census_income.X_train_all
        y_train = pre_census_income.y_train_all
        X_test = pre_census_income.X_test
        y_test = pre_census_income.y_test
        model = train_census_income.model
    elif dataset_name == 'german':
        X_train = pre_german_credit.X_train
        y_train = pre_german_credit.y_train
        X_test = pre_german_credit.X_test
        y_test = pre_german_credit.y_test
        model = train_german_credit.model
    elif dataset_name == 'bank':
        X_train = pre_bank_marketing.X_train_all
        y_train = pre_bank_marketing.y_train_all
        X_test = pre_bank_marketing.X_test
        y_test = pre_bank_marketing.y_test
        model = train_bank_marketing.model

    # census income
    dataset_name = 'adult'
    num_attribs = 0
    if dataset_name == 'adult':
        num_attribs = 12
    if dataset_name == 'german':
        num_attribs = 25
    if dataset_name == 'bank':
        num_attribs = 16

    X = pre_census_income_htx.get_all_data()

    all_dis_ins = np.empty(shape=(0, num_attribs))  # 存 all instances
    all_dis_ins_label = np.empty(shape=(0,))  # 存 all instances' labels
    print(all_dis_ins_label.shape)

    for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7])]:
        print('\n', benchmark, ':\n')
        ids_benchmark_EIDIG_5_pkl = '../logging_data_htx/generated_dis_instances_back_up/' + \
                                    benchmark + "_ids_EIDIG_5_1.pkl"
        file = open(ids_benchmark_EIDIG_5_pkl, 'rb')
        R = pickle.load(file)
        for r in R:
            # step1: 找 seed 的 label
            seed = r.seed[0]
            seed_label = 0
            for x in X:
                x_without_label = x[:-1]
                if (x_without_label == seed).all():
                    seed_label = x[-1]

            # step2: set global and local discriminatory instances's label to that same as seed_label
            dis_ins_label = seed_label

            if len(r.g_dis_ins) > 0:
                if (r.g_dis_ins[0] == seed).all():
                    # 排除 global = seed
                    continue

                all_dis_ins = np.append(all_dis_ins, r.g_dis_ins, axis=0)
                for i in range(len(r.g_dis_ins)):
                    all_dis_ins_label = np.append(all_dis_ins_label, [dis_ins_label], axis=0)

            if len(r.l_dis_ins) > 0:
                all_dis_ins = np.append(all_dis_ins, r.l_dis_ins, axis=0)
                for j in range(len(r.l_dis_ins)):
                    all_dis_ins_label = np.append(all_dis_ins_label, [dis_ins_label], axis=0)
        file.close()
    # end for
    num_of_ids = len(all_dis_ins_label)
    print("the number of all formal discriminatory instances: " + str(num_of_ids))
    num_percent = num_of_ids * percent
    print("the number of discriminatory instances used to retraining:" + str(num_percent))
    num_aug = int(num_percent)

    ids_aug = np.empty(shape=(0, len(X_train[0])))
    ids_aug_label = np.empty(shape=(0,))
    for _ in range(num_aug):
        rand_index = np.random.randint(len(all_dis_ins))
        ids_aug = np.append(ids_aug, [all_dis_ins[rand_index]], axis=0)
        ids_aug_label = np.append(ids_aug_label, [all_dis_ins_label[rand_index]], axis=0)

    X_train = np.append(X_train, ids_aug, axis=0)
    y_train = np.append(y_train, ids_aug_label, axis=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    model.evaluate(X_test, y_test)

    model_name = dataset_name + '_' + approach_name + '_retrained_model_no_majority_vote_' + str(percent) + '.h5'
    model_path = '../models_htx/retrained_models/' + model_name
    model.save(model_path)


def retraining(dataset_name, approach_name, ids):
    # randomly sample 5% of individual discriminatory instances generated for data augmentation
    # then retrain the original models

    ensemble_clf = joblib.load('../models/ensemble_models/' + dataset_name + '_ensemble.pkl')
    if dataset_name == 'adult':
        protected_attribs = pre_census_income.protected_attribs
        X_train = pre_census_income.X_train_all
        y_train = pre_census_income.y_train_all
        X_test = pre_census_income.X_test
        y_test = pre_census_income.y_test
        model = train_census_income.model
    elif dataset_name == 'german':
        protected_attribs = pre_german_credit.protected_attribs
        X_train = pre_german_credit.X_train
        y_train = pre_german_credit.y_train
        X_test = pre_german_credit.X_test
        y_test = pre_german_credit.y_test
        model = train_german_credit.model
    elif dataset_name == 'bank':
        protected_attribs = pre_bank_marketing.protected_attribs
        X_train = pre_bank_marketing.X_train_all
        y_train = pre_bank_marketing.y_train_all
        X_test = pre_bank_marketing.X_test
        y_test = pre_bank_marketing.y_test
        model = train_bank_marketing.model
    ids_aug = np.empty(shape=(0, len(X_train[0])))
    num_of_ids = len(ids)
    print("the number of all discriminatory instances: " + str(num_of_ids))
    num_percent_5 = num_of_ids * 0.05
    num_aug = int(num_percent_5)
    print("the number of augment samples from discriminatory instances (5%): " + str(num_aug))
    for _ in range(num_aug):
        rand_index = np.random.randint(len(ids))
        ids_aug = np.append(ids_aug, [ids[rand_index]], axis=0)
    label_vote = ensemble_clf.predict(np.delete(ids_aug, protected_attribs, axis=1))
    X_train = np.append(X_train, ids_aug, axis=0)
    y_train = np.append(y_train, label_vote, axis=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    model.evaluate(X_test, y_test)
    model.save('../models_htx/retrained_models/' + dataset_name + '_' + approach_name + '_retrained_model.h5')


def retraining_modify(dataset_name, approach_name, ids):
    # randomly sample 5% of individual discriminatory instances generated for data augmentation
    # then retrain the original models

    ensemble_clf = joblib.load('../models/ensemble_models/' + dataset_name + '_ensemble.pkl')
    if dataset_name == 'adult':
        protected_attribs = pre_census_income.protected_attribs
        X_train = pre_census_income.X_train_all
        y_train = pre_census_income.y_train_all
        X_test = pre_census_income.X_test
        y_test = pre_census_income.y_test
        model = train_census_income.model
    elif dataset_name == 'german':
        protected_attribs = pre_german_credit.protected_attribs
        X_train = pre_german_credit.X_train
        y_train = pre_german_credit.y_train
        X_test = pre_german_credit.X_test
        y_test = pre_german_credit.y_test
        model = train_german_credit.model
    elif dataset_name == 'bank':
        protected_attribs = pre_bank_marketing.protected_attribs
        X_train = pre_bank_marketing.X_train_all
        y_train = pre_bank_marketing.y_train_all
        X_test = pre_bank_marketing.X_test
        y_test = pre_bank_marketing.y_test
        model = train_bank_marketing.model

    ids_aug = np.empty(shape=(0, len(X_train[0])))

    '''
    修改前
    num_aug = int(len(ids) * 0.05)  # 注意这里的 5% 不是三个属性歧视实例各取 %5，而是先把三个属性对应的歧视实例何在一起，然后 random 取 5%
    '''

    # 修改后
    num_of_ids = len(ids)
    print("the number of all formal discriminatory instances: " + str(num_of_ids))
    num_percent_5 = num_of_ids * 0.05
    print("the number of 5% formal discriminatory instances:" + str(num_percent_5))
    num_aug = max(num_percent_5, 18102)
    print("the number of augment samples from discriminatory instances (5%): " + str(num_aug))  # 18102

    for _ in range(num_aug):
        rand_index = np.random.randint(len(ids))
        ids_aug = np.append(ids_aug, [ids[rand_index]], axis=0)
    label_vote = ensemble_clf.predict(np.delete(ids_aug, protected_attribs, axis=1))  # train 的时候就删掉了

    X_train = np.append(X_train, ids_aug, axis=0)
    y_train = np.append(y_train, label_vote, axis=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_val, y_val),
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    model.evaluate(X_test, y_test)
    model.save('../models_htx/retrained_models/' + dataset_name + '_' +
               approach_name + '_retrained_model.h5')


def retraining_with_formal_dis_ins():
    # census income
    dataset_name = 'adult'
    num_attribs = 0
    if dataset_name == 'adult':
        num_attribs = 12
    if dataset_name == 'german':
        num_attribs = 25
    if dataset_name == 'bank':
        num_attribs = 16

    approach_name = 'EIDIG_5'

    all_dis_ins = np.empty(shape=(0, num_attribs))
    ids_benchmark_EIDIG_5_formal_pkl = ''
    for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7])]:
        print('\n', benchmark, ':\n')
        ids_benchmark_EIDIG_5_formal_pkl = '../logging_data_htx/generated_dis_instances_back_up/' + \
                                           benchmark + "_ids_EIDIG_5_1_formal.pkl"
        file = open(ids_benchmark_EIDIG_5_formal_pkl, 'rb')
        R = pickle.load(file)
        for r in R:
            if len(r.g_dis_ins) > 0:
                all_dis_ins = np.append(all_dis_ins, r.g_dis_ins, axis=0)
            if len(r.l_dis_ins) > 0:
                all_dis_ins = np.append(all_dis_ins, r.l_dis_ins, axis=0)
        file.close()
    # retrain the original models
    print("进入 retraining 函数，正在准备重新训练模型")
    retraining(dataset_name, approach_name, all_dis_ins)


def main(argv=None):
    dataset_name = 'adult'
    approach_name = 'EIDIG_5'
    patch_remove_informal_dis_ins(dataset_name)
    retraining_with_formal_dis_ins()  # 有点疑问的是这个retraining_with_formal和下面的区别？
    percent = 0.02  # 0.02 == 2%, 0.05 == 5%, 0.1 == 10%
    # retraining_without_majority_voting(dataset_name, approach_name, percent)

    # reproduction
    # ids_C_a_EIDIG_5 = np.load('../logging_data/generated_dis_instances_back_up/C-a_ids_EIDIG_5.npy')
    # ids_C_r_EIDIG_5 = np.load('../logging_data/generated_dis_instances_back_up/C-r_ids_EIDIG_5.npy')
    # ids_C_g_EIDIG_5 = np.load('../logging_data/generated_dis_instances_back_up/C-g_ids_EIDIG_5.npy')
    # C_ids_EIDIG_5 = np.concatenate((ids_C_a_EIDIG_5, ids_C_r_EIDIG_5, ids_C_g_EIDIG_5),axis=0)
    # retraining(dataset_name, approach_name, C_ids_EIDIG_5)


if __name__ == '__main__':
    main()
