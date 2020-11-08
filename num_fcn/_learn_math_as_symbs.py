from ._two_lay_fcn import TwoLayFcn
from .work_with_symbls import create_matrices_x_y_from_symb_code
from ._ops_constants import SIGMOID
import math


class LearnMathAsSymbs(TwoLayFcn):
    def set_X_Y(self, str_X=None):
        self.str_X = str_X
        self.X, self.Y, _ = create_matrices_x_y_from_symb_code(
            str_X, self.in_1, self.out_2, devider=10)

    def evaluate(self):
        devider = 10
        x, _, exs = create_matrices_x_y_from_symb_code(
            self.str_X, self.in_1, self.out_2, devider)
        b_c_el = 0
        height_x = x.shape[0]
        for i in range(height_x):
            print('%d) ' % (i+1), end=' ')
            print(exs[i], end=' ')
            ans = self.forward(x[i], predict=True)
            len_ans = ans.shape[0]
            for elem in range(len_ans):
                ans_1 = ans[elem][0]
                if elem == 0:
                    b_c_el = math.ceil(ans_1*10-0.5)
                elif elem == 1:
                    ans_1 = math.ceil(ans_1*10-0.5)
                print(b_c_el, ans_1)

            print('***')

    def predict(self, inputt):
        # ans_1=0
        # b_c_el=0
        x, _, exs = create_matrices_x_y_from_symb_code(
            inputt, self.in_1, self.out_2, 10)
        _, ans = self.forward(x, predict=True)
        len_ans = ans.shape[0]
        for elem in range(len_ans):
            ans_1 = ans[elem][0]
            if elem == 0:
                b_c_el = math.ceil(ans_1*10-0.5)
            elif elem == 1:
                ans_1 = math.ceil(ans_1*10-0.5)
        return (b_c_el, ans_1)    
