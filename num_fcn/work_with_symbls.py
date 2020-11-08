import numpy as np

ICONST = 0.1
ADD = 0.2
SUBST = 0.3
MULT = 0.4
DEVIDE = 0.5
AND = 0.6
OR = 0.7
XOR = 0.8


def create_matrices_x_y_from_symb_code(math_text: str, net_in, net_out, devider) -> (tuple, tuple):
    splitted_text_by_simicln = math_text.split(';')
    print('spl text', splitted_text_by_simicln)
    len_splted = len(splitted_text_by_simicln)
    x = np.zeros((len_splted, net_in))
    y = np.zeros((len_splted, net_out))
    for el in range(len_splted):
        splitted_text_by_simicln[el] = splitted_text_by_simicln[el].strip()
    for row in range(len_splted):
        len_str = len(splitted_text_by_simicln[row])
        str_p = 0
        b_c_p = -1
        b_c_p_y = -1
        for _ in range(len_str):
            if str_p > len(splitted_text_by_simicln[row])-1:
                break
            symb = splitted_text_by_simicln[row][str_p]
            if symb.isdigit():
                b_c_p += 1
                x[row][b_c_p] = ICONST
                b_c_p += 1
                x[row][b_c_p] = int(symb)/devider
            elif symb == '+':
                b_c_p += 1
                x[row][b_c_p] = ADD

            elif symb == '-':
                b_c_p += 1
                x[row][b_c_p] = SUBST
            elif symb == '*':
                b_c_p += 1
                x[row][b_c_p] = MULT
            elif symb == '/':
                b_c_p += 1
                x[row][b_c_p] = DEVIDE
            elif symb == '&':
                b_c_p += 1
                x[row][b_c_p] = AND
            elif symb == '|':
                b_c_p += 1
                x[row][b_c_p] = OR
            elif symb == 'x':
                b_c_p += 1
                x[row][b_c_p] = XOR
            elif symb == '=':
                b_c_p_y += 1
                y[row][b_c_p_y] = ICONST
                b_c_p_y += 1
                y[row][b_c_p_y] = int(
                    splitted_text_by_simicln[row][str_p+1])/devider
                break

            str_p += 1

    return x, y, splitted_text_by_simicln

