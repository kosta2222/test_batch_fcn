# -*-coding: utf8-*-

import sys
from _operations import Ops  # функции активации и их производные
from _ops_constants import RELU, SIGMOID, TAN, TRESHOLD_FUNC_HALF, LEAKY_RELU
from _indexes_to_ser import IndexesToSer  # индексы для сериализации
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 2-слойнаЯ сеть свЯзей


class Two_lay_fcn:

    def __init__(self, ops=Ops(), idxs=IndexesToSer(), in_1=0, out_1=0, out_2=0, act_func_1=SIGMOID, act_func_2=SIGMOID, with_biasses=True, load_f_name=''):

        self._ops = ops  # также установим функции активации и их производные
        self._idxs = idxs  # индексы для сериализации

        if load_f_name != '':  # файл сериализации нам  задан, загружаем
            self.load_ser_d_file(load_f_name)

        else:  # файл сериализации нам не задан, строим сеть, после этого блока будет работать to_file()

            self.build_fcn_pars(in_1, out_1, out_2,
                                act_func_1, act_func_2, with_biasses)

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

        self._opsalpha_leaky_relu = net[self._idxs.alpha_leaky_relu_k]
        self.alpha_sigmoid = net[self._idxs.alpha_sigmoid_k]
        self.alpha_tan = net[self._idxs.alpha_tan_k]
        self.beta_tan = net[self._idxs.beta_tan_k]

    def build_fcn_pars(self, in_1, out_1, out_2, act_func_1, act_func_2, with_biasses):
        np.random.seed(1)
        self.W1 = np.random.normal(0, 1, (out_1, in_1))
        self.W2 = np.random.normal(0, 1, (out_2, out_1))
        self.is_with_biasses = with_biasses

        self.act_func_1 = act_func_1
        self.act_func_2 = act_func_2

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

    def plot_gr(self, _file: str, errors: list, epochs: list) -> None:
        fig: plt.Figure = None
        ax: plt.Axes = None
        fig, ax = plt.subplots()
        ax.plot(epochs, errors,
                label="learning",
                )
        plt.xlabel('Эпоха обучения')
        plt.ylabel('loss')
        ax.legend()
        plt.savefig(_file)
        print("Graphic saved")
        plt.show()

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

    def fit(self, learning_rate=0.1, reg_param=0, max_iter=5000, batch_size=1):
        hid_err = 0
        out_cn_err = 0

        hid_bias_err = 0
        out_bias_err = 0

        cost = np.zeros((max_iter, 1))

        samp_count = 1
        for i in range(max_iter):

            hid_err = 0
            out_cn_err = 0

            hid_bias_err = 0
            out_bias_err = 0

            gl_err = 0

            m = self.X.shape[0]
            samp_p = 0
            for _ in range(m):
                sys.stdout.write(
                    "\rIteration: {} and {} ".format(i + 1, samp_p + 1))

                if samp_count % (batch_size+1) != 0:  # Накапливаем градиент
                    # Forward Prop.
                    input_vec = self.X[samp_p].reshape(
                        self.X[samp_p].shape[0], 1)
                    hid, out_cn = self.forward(input_vec)

                    # Back prop.
                    error_metric = out_cn - \
                        self.Y[samp_p].reshape(self.Y[samp_p].shape[0], 1)
                    out_cn_half_err = np.multiply(out_cn - self.Y[samp_p].reshape(self.Y[samp_p].shape[0], 1),
                                                  self._ops.operations(self.act_func_2 + 1, out_cn))

                    out_cn_err +=\
                        out_cn_half_err.dot(hid.T)

                    hid_half_err = np.multiply((self.W2.T.dot(out_cn_half_err)),
                                               self._ops.operations(self.act_func_1 + 1, hid))
                    hid_err += hid_half_err.dot(input_vec.T)

                    hid_bias_err += hid_half_err * 1
                    out_bias_err += out_cn_half_err * 1
                    gl_err += np.sum(np.square(error_metric))
                else:  # Применяем его

                    self.W1 = self.W1 - learning_rate * \
                        (hid_err / m) + ((reg_param / m) * self.W1)

                    self.W2 = self.W2 - learning_rate * \
                        (out_cn_err / m) + ((reg_param / m) * self.W2)

                    self.B1 = self.B1 - learning_rate * (hid_bias_err / m)
                    self.B2 = self.B2 - learning_rate * (out_bias_err / m)
                    samp_p += 1
                    samp_count += 1
                    continue

                samp_p += 1
                samp_count += 1

            gl_err = gl_err / 2
            cost[i] = gl_err
            sys.stdout.write("error {0}".format(gl_err))
            sys.stdout.flush()  # Updating the teself.Xt.

        self.plot_gr('gr.png', cost, range(max_iter))
