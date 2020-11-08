from ._two_lay_fcn import TwoLayFcn
from ._learn_logic_as_symbs import LearnLogicAsSymbs
from ._learn_math_as_symbs import LearnMathAsSymbs
class NetFactory:
 o=None
 @classmethod
 def get_net(cls, key):
    """
    Возвращает сеть нацеленную на обработку определеной информации
    arg: key сторока - просто числовая сеть "net", символьная математика "sym_math", символьная логика "sym_log"
    return обьект
    """ 
    if key=='net':
      cls.o=TwoLayFcn()
    elif key=='sym_math':
      cls.o=LearnMathAsSymbs()
    elif key=='sym_log':
      cls.o=LearnLogicAsSymbs()    

    return cls.o      