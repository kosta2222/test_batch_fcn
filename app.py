# -*-coding: utf8-*-

import sys
from work_with_arr import calc_vec_as_one_zero, calc_as_hash
import math
import numpy as np
import matplotlib.pyplot as plt
from work_with_symbls import create_matrices_x_y_from_symb_code


import pickle

# 2-слойнаЯ сеть свЯзей
TRESHOLD_FUNC_HALF = 0
TRESHOLD_FUNC_HALF_DERIV = 1
SIGMOID = 2
SIGMOID_DERIV = 3
RELU = 4
RELU_DERIV = 5
TAN = 6
TAN_DERIV = 7
INIT_W_MY = 8
INIT_W_RANDOM = 9
LEAKY_RELU = 10
LEAKY_RELU_DERIV = 11


class Two_lay_fcn:

    def __init__(self, in_1=0, out_1=0, out_2=0, act_func_1=SIGMOID, act_func_2=SIGMOID, with_biasses=True, load_f_name=''):
        # индексы для сериализации в массиве
        self.W1_k = 0
        self.W2_k = 1
        self.B1_k = 2
        self.B2_k = 3
        self.act_func_1_k = 4
        self.act_func_2_k = 5
        self.in_1_k = 6
        self.out_2_k = 7
        self.is_with_biasses_k = 8
        self.alpha_leaky_relu_k = 9
        self.alpha_sigmoid_k = 10
        self.alpha_tan_k = 11
        self.beta_tan_k = 12

        if load_f_name != '':  # файл сериализации нам  задан, загружаем
            with open(load_f_name, 'rb') as f:
                net = pickle.load(f)
            self.W1 = net[self.W1_k]
            self.W2 = net[self.W2_k]
            self.act_func_1 = net[self.act_func_1_k]
            self.act_func_2 = net[self.act_func_2_k]
            self.in_1 = net[self.in_1_k]
            self.out_2 = net[self.out_2_k]
            self.is_with_biasses = net[self.is_with_biasses_k]
            if with_biasses:
                self.B1 = net[self.B1_k]
                self.B2 = net[self.B2_k]
            else:
                self.B1 = 0
                self.B2 = 0

            self.alpha_leaky_relu = net[self.alpha_leaky_relu_k]
            self.alpha_sigmoid = net[self.alpha_sigmoid_k]
            self.alpha_tan = net[self.alpha_tan_k]
            self.beta_tan = net[self.beta_tan_k]

        else:  # файл сериализации нам не задан, строим сеть, после этого блока будет работать to_file()
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

        ser_arr[self.W1_k] = self.W1
        ser_arr[self.B1_k] = self.B1
        ser_arr[self.W2_k] = self.W2
        ser_arr[self.B2_k] = self.B2
        ser_arr[self.act_func_1_k] = self.act_func_1
        ser_arr[self.act_func_2_k] = self.act_func_2
        ser_arr[self.in_1_k] = self.in_1
        ser_arr[self.out_2_k] = self.out_2
        ser_arr[self.is_with_biasses_k] = self.is_with_biasses

        ser_arr[self.alpha_leaky_relu_k] = self.alpha_leaky_relu
        ser_arr[self.alpha_sigmoid_k] = self.alpha_sigmoid
        ser_arr[self.alpha_tan_k] = self.alpha_tan
        ser_arr[self.beta_tan_k] = self.beta_tan

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
        self.alpha_leaky_relu = alpha_leaky_relu
        self.alpha_sigmoid = alpha_sigmoid
        self.alpha_tan = alpha_tan
        self.beta_tan = beta_tan

    def operations(self, op, x):
        y = np.zeros(x.shape[0])
        height = x.shape[0]
        if op == RELU:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = 0
                else:
                    y[row] = x[row][0]

            return np.array([y]).T

        elif op == RELU_DERIV:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = 0
            else:
                y[row] = 1
            return np.array([y]).T
        elif op == TRESHOLD_FUNC_HALF:
            for row in range(height):
                if (x[row][0] > 0.5):
                    y[row] = 1
                else:
                    y[row] = 0
            return np.array([y]).T
        elif op == TRESHOLD_FUNC_HALF_DERIV:
            return 1
        elif op == LEAKY_RELU:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = self.alpha_leaky_relu * x[row][0]
                else:
                    y[row] = x[row][0]
            return np.array([y]).T

        elif op == LEAKY_RELU_DERIV:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = self.alpha_leaky_relu
                else:
                    y[row] = 1
            return np.array([y]).T
        elif op == SIGMOID:
            y = 1 / (1 + np.exp(-self.alpha_sigmoid * x))
            return y
        elif op == SIGMOID_DERIV:
            return self.alpha_sigmoid * x * (1 - x)
        elif op == TAN:
            y = self.alpha_tan * np.tanh(self.beta_tan * x)
            return y
        elif op == TAN_DERIV:
            return self.beta_tan * self.alpha_tan * 4 / ((np.exp(self.beta_tan * x) + np.exp(-self.beta_tan * x))**2)
        else:
            print("Op or function does not support ", op)

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

    def evaluate_as_logic(self):
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

        print("========")

    def forward(self, X, predict=False):
        X = np.array(X)
        inputs = X.reshape(X.shape[0], 1)

        matr_prod_hid = self.W1.dot(inputs) + self.B1
        hid = self.operations(self.act_func_1, matr_prod_hid)

        matr_prod_out = self.W2.dot(hid) + self.B2
        out_cn = self.operations(self.act_func_2, matr_prod_out)

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
                                                  self.operations(self.act_func_2 + 1, out_cn))

                    out_cn_err +=\
                        out_cn_half_err.dot(hid.T)

                    hid_half_err = np.multiply((self.W2.T.dot(out_cn_half_err)),
                                               self.operations(self.act_func_1 + 1, hid))
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

        

    def predict_spec(self, contr_co_s: tuple, s, devider=1):
        if contr_co_s == 'logic':
            print('mode', contr_co_s)
        elif contr_co_s == 'math':
            print('mode', contr_co_s)
        else:
            raise Exception("Theare no such keyword %s" % contr_co_s)
        x, _, exs = create_matrices_x_y_from_symb_code(
            s, self.in_1, self.out_2, devider)
        b_c_el = 0
        height_x = x.shape[0]
        for i in range(height_x):
            print('%d) ' % (i+1), end=' ')
            print(exs[i], end=' ')
            ans = self.forward(x[i], predict=True)
            len_ans = ans.shape[0]
            for elem in range(len_ans):
                ans_1 = ans[elem][0]

                if contr_co_s == 'logic':
                    if elem == 0:
                        b_c_el = math.ceil(ans_1*10-0.5)
                    elif elem == 1:
                        if ans_1 < 0.5:
                            ans_1 = 0
                        else:
                            ans_1 = 1
                        print('b_c_el', b_c_el, 'ans_1', ans_1)
                elif contr_co_s == 'math':
                    if elem == 0:
                        b_c_el = math.ceil(ans_1*10-0.5)
                    elif elem == 1:
                        ans_1 = math.ceil(ans_1*10-0.5)
                        print('b_c_el', b_c_el, 'ans_1', ans_1)

            print('***')


def main():
    s = """1+0=1;
       1+1=2;
       1+2=3;
       1+3=4;
       1+4=5;
       """

    X, Y, _ = create_matrices_x_y_from_symb_code(s, 5, 2, devider=10)
    net = Two_lay_fcn(5, 7, 2, act_func_1=SIGMOID,
                      act_func_2=SIGMOID, with_biasses=True)
    net.set_X_Y(X, Y)
    net.set_act_funcs_pars(alpha_sigmoid=3.5)
    net.fit(max_iter=7000, reg_param=0, batch_size=3)
    net.to_file('wei.my')  # Сохранили обучение на s
    
    # s1 = """
    #  1|1=1;
    #  1|0=1;
    #  0|1=1;
    #  0|0=0;"""
    """
    net_1 = Two_lay_fcn(load_f_name='wei_1.my')  # Дообучаем ее на s1

    X1, Y1, _ = create_matrices_x_y_from_symb_code(s1, 5, 2, devider=1)
    net_1.set_X_Y(X1, Y1)
    net_1.fit(max_iter=1000, reg_param=0, batch_size=3)
    net_1.to_file('wei.my')  # Сохранили обучение на s1
    """

def test():
    s = """1+0=1;
       1+1=2;
       1+2=3;
       1+3=4;
       1+4=5;
       2+2=4;
       7+2=9;
       2+0=2;
       2*3=6;
       """

    net = Two_lay_fcn(load_f_name='wei.my')
    net.predict_spec('math', s, devider=10)
    # s1 = """
    #  1|1=1;
    #  1|0=1;
    #  0|1=1;
    #  0|0=0"""
    # X, Y, _ = create_matrices_x_y_from_symb_code(s1, 5, 2, devider=10)
    # net.predict_spec('logic', s1, devider=1)
    


main()
# test()
