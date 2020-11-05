# -*-coding: utf8-*-

import sys
from ._operations import Ops  # функции активации и их производные
from ._ops_constants import RELU, SIGMOID, TAN, TRESHOLD_FUNC_HALF, LEAKY_RELU
from ._indexes_to_ser import IndexesToSer  # индексы для сериализации
from ._util import Util

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 2-слойная сеть связей


class TwoLayFcn:

    def __init__(self, ops=Ops(), idxs=IndexesToSer(), util=Util(), in_1=0, out_1=0, out_2=0, with_biasses=True, load_f_name=''):

        self._ops = ops  # также установим функции активации и их производные
        self._idxs = idxs  # индексы для сериализации
        self._util = util

        if load_f_name != '':  # файл сериализации нам  задан, загружаем
            self.load_ser_d_file(load_f_name)

        else:  # файл сериализации нам не задан, строим сеть, после этого блока будет работать с to_file()

            self.build_fcn_pars(in_1, out_1, out_2,
                                with_biasses)

    def load_ser_d_file(self, load_f_name):
        net = None
        with open(load_f_name, 'rb') as f:
            net = pickle.load(f)
        self.W1 = net[self._idxs.W1_k]
        self.W2 = net[self._idxs.W2_k]
        self.act_func_1 = net[self._idxs.act_func_1_k]
        self.act_func_2 = net[self._idxs.act_func_2_k]
        self.in_1 = net[self._idxs.in_1_k]
        self.out_2 = net[self._idxs.out_2_k]
        self.is_with_biasses = net[self._idxs.is_with_biasses_k]
        if self.is_with_biasses:
            self.B1 = net[self._idxs.B1_k]
            self.B2 = net[self._idxs.B2_k]
        else:
            self.B1 = 0
            self.B2 = 0

        self._ops.alpha_leaky_relu = net[self._idxs.alpha_leaky_relu_k]
        self._ops.alpha_sigmoid = net[self._idxs.alpha_sigmoid_k]
        self._ops.alpha_tan = net[self._idxs.alpha_tan_k]
        self._ops.beta_tan = net[self._idxs.beta_tan_k]

    def build_fcn_pars(self, in_1, out_1, out_2, with_biasses):
        np.random.seed(1)
        self.W1 = np.random.normal(0, 1, (out_1, in_1))
        self.W2 = np.random.normal(0, 1, (out_2, out_1))
        self.is_with_biasses = with_biasses

        self.in_1 = in_1
        self.out_2 = out_2
        self.is_with_biasses = with_biasses

        self.alpha_leaky_relu = 1.7159
        self.alpha_sigmoid = 2
        self.alpha_tan = 1.7159
        self.beta_tan = 2/3

        if with_biasses:
            self.B1 = np.random.random((out_1, 1))
            self.B2 = np.random.random((out_2, 1))
        else:
            self.B1 = 0
            self.B2 = 0

    def set_act_funcs(self, act_func1=SIGMOID, act_func2=SIGMOID):
        self.act_func_1 = act_func1
        self.act_func_2 = act_func2

    def to_file(self, f_name):
        """
        Сериализуем в файл
        """
        ser_arr = [None] * 13

        ser_arr[self._idxs.W1_k] = self.W1
        ser_arr[self._idxs.B1_k] = self.B1
        ser_arr[self._idxs.W2_k] = self.W2
        ser_arr[self._idxs.B2_k] = self.B2
        ser_arr[self._idxs.act_func_1_k] = self.act_func_1
        ser_arr[self._idxs.act_func_2_k] = self.act_func_2
        ser_arr[self._idxs.in_1_k] = self.in_1
        ser_arr[self._idxs.out_2_k] = self.out_2
        ser_arr[self._idxs.is_with_biasses_k] = self.is_with_biasses

        ser_arr[self._idxs.alpha_leaky_relu_k] = self._ops.alpha_leaky_relu
        ser_arr[self._idxs.alpha_sigmoid_k] = self._ops.alpha_sigmoid
        ser_arr[self._idxs.alpha_tan_k] = self._ops.alpha_tan
        ser_arr[self._idxs.beta_tan_k] = self._ops.beta_tan

        with open(f_name, 'wb') as f:
            pickle.dump(ser_arr, f)
        print('Weights saved')

    def set_X_Y(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)

    def set_act_funcs_pars(self, alpha_leaky_relu=1, alpha_sigmoid=1, alpha_tan=1, beta_tan=1):
        """
         Если захочим поменять хотя бы 1 параметр активационной функции в отличии от конструктора [именовынный аргумент/аргумент по умолчанию]
        """
        self._ops.alpha_leaky_relu = alpha_leaky_relu
        self._ops.alpha_sigmoid = alpha_sigmoid
        self._ops.alpha_tan = alpha_tan
        self._ops.beta_tan = beta_tan

    def forward(self, X, predict=False):
        X = np.array(X)
        inputs = X.reshape(X.shape[0], 1)

        matr_prod_hid = self.W1.dot(inputs) + self.B1
        hid = self._ops.operations(self.act_func_1, matr_prod_hid)

        matr_prod_out = self.W2.dot(hid) + self.B2
        out_cn = self._ops.operations(self.act_func_2, matr_prod_out)

        if predict:
            return out_cn
        return (hid, out_cn)
    def evaluate(self, inputs):
        m = self.X.shape[0]
        for single_array_ind in range(m):
            inputs = self.X[single_array_ind]

            output_2_layer = self.forward(inputs, predict=True)

            equal_flag = 0

            for row in range(self.out_2):
                elem_net = output_2_layer[row]
                elem_train_out = self.Y[single_array_ind][row]
                if elem_net > 0.5:
                    elem_net = 1
                else:
                    elem_net = 0
                print("elem:", elem_net)
                print("elem Y:", elem_train_out)
                print('----')
                if elem_net == elem_train_out:
                    equal_flag = 1
                else:
                    equal_flag = 0
                    break
            if equal_flag == 1:
                print('-vecs are equal-')
            else:
                print('-vecs are not equal-')
