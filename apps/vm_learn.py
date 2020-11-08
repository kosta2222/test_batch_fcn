# from num_fcn._backprop import Backprop  # Еще не изучил mini-batch обучение
from num_fcn._backprop_frixoe import Backprop # Здесь full-batch обучение

from num_fcn._net_factory import NetFactory
from num_fcn._ops_constants  import SIGMOID, LEAKY_RELU, TAN, RELU
from num_fcn._configure_cmd_ap import ConfigureCmdApp


x=[[1, 0], [0, 1], [1, 1], [0, 0]]
y=[[1],[1],[1],[0]]

s="1|1=1;1|0=1;0|1=1;0|0=0;"

net_1 = NetFactory.get_net("sym_log") # фабрика различных обрабатывающих сетей
net_1.set_act_funcs()
net_1.build_fcn_pars(5, 7, 2, True)
net_1.set_act_funcs_pars()
net_1.set_X_Y(s)
back_o=Backprop(fcn=net_1)
conf=ConfigureCmdApp(learn=back_o)
conf.loop()              