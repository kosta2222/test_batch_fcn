from ._two_lay_fcn import TwoLayFcn
from .work_with_symbls import create_matrices_x_y_from_symb_code
from ._ops_constants import SIGMOID
import math


class LearnMathAsSymbs(TwoLayFcn):
    def set_X_Y(self, s):
        self.X, self.Y, _ = create_matrices_x_y_from_symb_code(
            s, self.in_1, self.out_2, devider=10)

    def evaluate(self, contr_co_s, s, devider=10):
        if contr_co_s == 'math':
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
                if contr_co_s == 'math':
                    if elem == 0:
                        b_c_el = math.ceil(ans_1*10-0.5)
                    elif elem == 1:
                        ans_1 = math.ceil(ans_1*10-0.5)
                        # return (b_c_el, ans_1)
                        print(b_c_el, ans_1)    
                         
            print('***')

