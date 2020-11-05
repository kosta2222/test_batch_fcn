from ._two_lay_fcn import Two_lay_fcn
from .work_with_symbls import create_matrices_x_y_from_symb_code
from ._ops_constants import SIGMOID
import math


class LearnMathAsSymbs(Two_lay_fcn):
    def set_X_Y(self, s):
        self.X, self.Y, _ = create_matrices_x_y_from_symb_code(
            s, self.in_1, self.out_2, devider=10)

    def predict_spec(self, contr_co_s, s, devider=10):
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
                if contr_co_s == 'math':
                    if elem == 0:
                        b_c_el = math.ceil(ans_1*10-0.5)
                    elif elem == 1:
                        ans_1 = math.ceil(ans_1*10-0.5)
                        return (b_c_el, ans_1)
                        print(b_c_el, ans_1)    
                         
            print('***')


def learn():
    s1 ="""
     1+0=1;
     1+1=2;
     1+2=3;
     1+3=4;
     1+4=5;
     1+5=6;
     6+1=7;
     7+1=8;
     8+1=9;
       """

    net_1 = LearnMathAsSymbs(in_1=5, out_1=7, out_2=2)  
    net_1.set_act_funcs_pars(alpha_sigmoid=2)
    net_1.set_X_Y(s1)
    net_1.fit(max_iter=8000, reg_param=0, batch_size=2, learning_rate=0.1)
    net_1.to_file('wei_math.my')  # Сохранили обучение на s1
    net_1.predict_spec("math",s1)

# learn()    
