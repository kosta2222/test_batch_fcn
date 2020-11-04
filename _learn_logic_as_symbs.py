from  _two_lay_fcn import Two_lay_fcn
from work_with_symbls import create_matrices_x_y_from_symb_code
from _ops_constants import SIGMOID


class LearnLogicAsSymbs(Two_lay_fcn):
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


    def predict_spec(self, contr_co_s: tuple, s, devider=1):
        if contr_co_s == 'logic':
            print('mode', contr_co_s)
        # elif contr_co_s == 'math':
        #     print('mode', contr_co_s)
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
    # s = """1+0=1;
    #    1+1=2;
    #    1+2=3;
    #    1+3=4;
    #    1+4=5;
    #    """

    # X, Y, _ = create_matrices_x_y_from_symb_code(s, 5, 2, devider=10)
    # net = LearnLogicAsSymbs(5, 7, 2, act_func_1=SIGMOID,
    #                         act_func_2=SIGMOID, with_biasses=True)
    # net.set_X_Y(X, Y)
    # net.set_act_funcs_pars(alpha_sigmoid=3.5)
    # net.fit(max_iter=7000, reg_param=0, batch_size=3)
    # net.to_file('wei.my')  # Сохранили обучение на s

    s1 = """
     1|1=1;
     1|0=1;
     0|1=1;
     0|0=0;"""
    
    net_1 = LearnLogicAsSymbs(in_1=5, out_1=7, out_2=2)  # Дообучаем ее на s1
    net_1.set_act_funcs_pars(alpha_sigmoid=3.5)

    X1, Y1, _ = create_matrices_x_y_from_symb_code(s1, 5, 2, devider=1)
    net_1.set_X_Y(X1, Y1)
    net_1.fit(max_iter=1000, reg_param=0, batch_size=3)
    net_1.to_file('wei.my')  # Сохранили обучение на s1
    


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
