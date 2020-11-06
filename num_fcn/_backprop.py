from ._two_lay_fcn import TwoLayFcn
import sys as _sys
import numpy as np


class Backprop:
    def __init__(self, fcn=TwoLayFcn()):
        self._fcn = fcn
        self.samp_p = 0
        self.samp_count = 1

    def calc_out_err(self, out_cn):
        out_cn_half_err = np.multiply(out_cn - self._fcn.Y[self.samp_p].reshape(self._fcn.Y[self.samp_p].shape[0], 1),
                                      self._fcn._ops.operations(self._fcn.act_func_2 + 1, out_cn))
        return out_cn_half_err

    def calc_hid_err(self, hid, out_cn_half_err):
        hid_half_err = np.multiply((self._fcn.W2.T.dot(out_cn_half_err)),
                                   self._fcn._ops.operations(self._fcn.act_func_1 + 1, hid))
        return hid_half_err

    def update_matrix(self, w, errors, reg_param, learning_rate, m):
        w = w - learning_rate * \
            (errors / m) + ((reg_param / m) * w)
        return w

    def fit(self, learning_rate=0.1, reg_param=0, max_iter=5000, batch_size=1):
        hid_err = 0
        out_cn_err = 0

        hid_bias_err = 0
        out_bias_err = 0

        cost = np.zeros((max_iter, 1))

        
        for i in range(max_iter):

            hid_err = 0
            out_cn_err = 0

            hid_bias_err = 0
            out_bias_err = 0

            gl_err = 0

            m = self._fcn.X.shape[0]
            # is_samples = True
            self.samp_p=0
            for _ in range(m) :
                # self.samp_p=0
                _sys.stdout.write(
                    "\rIteration: {} and {} ".format(i + 1, self.samp_p + 1))

                if self.samp_count % (batch_size+1) != 0:  # Накапливаем градиент
                    input_vec = self._fcn.X[self.samp_p].reshape(
                        self._fcn.X[self.samp_p].shape[0], 1)
                    hid, out_cn = self._fcn.forward(input_vec)

                    error_metric = out_cn - \
                        self._fcn.Y[self.samp_p].reshape(
                            self._fcn.Y[self.samp_p].shape[0], 1)
                    out_cn_half_err = self.calc_out_err(out_cn)
                    out_cn_err +=\
                        out_cn_half_err.dot(hid.T)

                    hid_half_err = self.calc_hid_err(hid, out_cn_half_err)

                    hid_err += hid_half_err.dot(input_vec.T)

                    hid_bias_err += hid_half_err * 1
                    out_bias_err += out_cn_half_err * 1
                   
                    gl_err += self._fcn._util.mse(error_metric)
                else:  # Применяем его
                    self._fcn.W1 = self.update_matrix(
                        self._fcn.W1, hid_err, reg_param, learning_rate, m)

                    self._fcn.W2 = self.update_matrix(
                        self._fcn.W2, out_cn_err, reg_param, learning_rate, m)

                    self._fcn.B1 = self._fcn.B1 - \
                        learning_rate * (hid_bias_err / m)
                    self._fcn.B2 = self._fcn.B2 - \
                        learning_rate * (out_bias_err / m)
                   
                self.samp_p += 1
                self.samp_count += 1

            gl_err = gl_err / 2
            cost[i] = gl_err
            _sys.stdout.write("error {0}".format(gl_err))
            _sys.stdout.flush()  # Updating the teself.Xt.

        self._fcn._util.plot_gr('gr.png', cost, range(max_iter))
