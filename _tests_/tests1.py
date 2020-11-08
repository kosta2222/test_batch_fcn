from num_fcn.work_with_symbls import create_matrices_x_y_from_symb_code

s = "1|0=1;1|1=1;0|1=1;0|0=0"
x, y, exs = create_matrices_x_y_from_symb_code(s, 5, 2, 1)
assert len(x)==4
assert len(y)==4
assert len(exs)==4
assert any(x[0]) == any([0.1, 1., 0.7, 0.1, 0.])
assert any(y[0])==any([0.1, 1.])
assert any(exs) == any(['1|0=1', '1|1=1', '0|1=1', '0|0=0'])