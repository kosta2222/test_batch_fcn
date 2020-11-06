from ._two_lay_fcn import TwoLayFcn
class NetFactory:
 o=None
 @classmethod
 def get_net(self, key):
    """
    Возвращает сеть нацеленную на обработку определеной информации
    arg: key сторока - просто числовая сеть "net", символьная математика "sym_math", символьная логика "sym_log"
    return обьект
    """ 
    if key=='net':
      o=TwoLayFcn()

    return o      