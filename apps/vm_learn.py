from num_fcn._backprop import Backprop
# from num_fcn._learn_logic_as_symbs import LearnLogicAsSymbs
from num_fcn._net_factory import NetFactory
from num_fcn._ops_constants  import SIGMOID, LEAKY_RELU, TAN, RELU
from num_fcn._configure_cmd_ap import ConfigureCmdApp


x=[[1, 0], [0, 1], [1, 1], [0, 0]]
y=[[1],[1],[1],[0]]
    
net_1 = NetFactory.get_net("net")  # обучаем ее на s1
net_1.set_act_funcs()
net_1.build_fcn_pars(2, 7, 1, True)
net_1.set_act_funcs_pars(alpha_tan=1.73, beta_tan=2/3)
net_1.set_X_Y(x, y)
back_o=Backprop(fcn=net_1)
conf=ConfigureCmdApp(learn=back_o)
conf.loop()              