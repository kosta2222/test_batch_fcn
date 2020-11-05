from num_fcn._backprop import Backprop
from num_fcn._learn_logic_as_symbs import LearnLogicAsSymbs
from num_fcn._ops_constants  import SIGMOID, LEAKY_RELU, TAN, RELU


"""
Настройка гиперпараметров через консоль
"""


class ConfigureCmdApp:
    def __init__(self, learn=Backprop(), sx=None):
        """
        arg:
        learn обьект минимизации ошибки он сам принимает сеть прямого распространения, предполагаем сеть будет прринимать символьную информацию
        sx символьная строка обучения  
        """
        self.sx=sx
        self._learn = learn
        self.__max_iter = 1000
        self.__learning_rate=0.1
        self.__alpha_sigmoid = 1
        self.__batch_size = 2
        self.__reg_param = 0
        self.__act_func1 = SIGMOID
        self.__act_func2= SIGMOID
        self.funcs=['', 'SIGMOID', '', 'RELU', '', 'TAN', 'LEAKY_RELU']

    def loop(self):
        print('set-max-iter <Val> | set-act-func-1 <Str> | set-act-func-2 <Str> | set-alpha-sigmoid <Val> | r - сразу дефаултные:\
max-iter=1000 alpha-sigmoid=1 batch-size=2 reg-param=0 обе активационные функции SIGMOID | stop')
        shell_is_running = True
        while shell_is_running:
            s = input('->')
            if s == 'set-max-iter':
                val = input('Val->')
                self.__max_iter = int(val)
            elif s=='r':
                self._learn._fcn.set_act_funcs(act_func1=self.__act_func1, act_func2=self.__act_func2)
                self._learn.fit(learning_rate=self.__learning_rate, reg_param=self.__reg_param, max_iter=self.__max_iter,
                batch_size=self.__batch_size)
                self._learn._fcn.evaluate('logic', self.sx )
                s=input('?->Продолжать заново обучение с другими параметрами? y/n :')
                if s=='y':
                    continue
                elif s=='n':
                    self._learn._fcn.to_file('wei.my')
                    break

            elif s=='set-alpha-sigmoid':
                s=input('Val->')
                self.__alpha_sigmoid=float(s)     
            elif s=='stop':
                break
            elif s=='set-act-func-1':
                print(self.funcs) 
                val = input('Str->')
                try:
                  self.__act_func1 = self.funcs.index(val)
                except ValueError:
                    print('No such function')
            elif s=='set-act-func-2':
                print(self.funcs) 
                val = input('Str->')
                try:
                  self.__act_func2 = self.funcs.index(val)
                except ValueError:
                    print('No such function')
            else:
                print("Unrecognized cmd")



