from ._two_lay_fcn import TwoLayFcn
from ._backprop import Backprop
from .work_with_symbls import create_matrices_x_y_from_symb_code
from ._ops_constants import SIGMOID
import math


class LearnLogicAsSymbs(TwoLayFcn):
    def set_X_Y(self, str_X=None):
        self.str_X = str_X
        self.X, self.Y, _ = create_matrices_x_y_from_symb_code(
            str_X, self.in_1, self.out_2, devider=1)

    def evaluate(self):
        devider = 1
        x, _, exs = create_matrices_x_y_from_symb_code(
            self.str_X, self.in_1, self.out_2, devider)
        b_c_el = 0
        height_x = x.shape[0]
        for i in range(height_x):
            print('%d) ' % (i+1), end=' ')
            print(exs[i], end=' ')
            ans = self.forward(x[i], predict=True)
            for elem in range(self.out_2):
                ans_1 = ans[elem][0]
                if elem == 0:
                    print('b_c_el', ans_1)
                    b_c_el = math.ceil(ans_1*10-0.5)
                elif elem == 1:
                    print('ans',ans_1)
                    if ans_1 < 0.5:
                        ans_1 = 0
                    else:
                        ans_1 = 1
                    print('b_c_el', b_c_el, 'ans', ans_1)

            print('***')

    def predict(self, inputt):
        b_c_el=0
        ans_1=0
        devider=1
        x, _, _ = create_matrices_x_y_from_symb_code(
            inputt, self.in_1, self.out_2, devider)
        print('x', x)    
        ans= self.forward(x[0], predict=True)
        print('ans', ans)
        for elem in range(self.out_2):
            ans_1 = ans[elem]
            if elem == 0:
                b_c_el = math.ceil(ans_1*10-0.5)
                print('b_c_el', b_c_el)
            elif elem == 1:
                if ans_1 < 0.5:
                    ans_1 = 0
                else:
                    ans_1 = 1
                return (b_c_el, ans_1)

    def __str__(self):
        s = "in1: {0}  out2: {1}".format(self.in_1, self.out_2)
        return s
