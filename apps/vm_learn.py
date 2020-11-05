from num_fcn._backprop import Backprop
from num_fcn._learn_logic_as_symbs import LearnLogicAsSymbs
from num_fcn._ops_constants  import SIGMOID, LEAKY_RELU, TAN, RELU
from num_fcn._configure_cmd_ap import ConfigureCmdApp


s1 = """
     1|1=1;
     1|0=1;
     0|1=1;
     0|0=0;"""
    
net_1 = LearnLogicAsSymbs(in_1=5, out_1=7, out_2=2)  # обучаем ее на s1
net_1.set_act_funcs()
net_1.set_X_Y(s1)
back_o=Backprop(fcn=net_1)
conf=ConfigureCmdApp(learn=back_o, sx=s1)
conf.loop()              