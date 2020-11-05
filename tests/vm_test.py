from num_fcn._learn_logic_as_symbs import LearnLogicAsSymbs
from num_fcn._learn_math_as_symbs import LearnMathAsSymbs
import os

s1=os.path.join(os.path.dirname(__file__),"..", 'wei.my')
s2=os.path.join(os.path.dirname(__file__), "..", 'wei_math.my')
class VmTest:
    def __init__(self, learn_logic_fcn=LearnLogicAsSymbs(load_f_name=s1),
     learn_math_fcn=LearnMathAsSymbs(load_f_name=s2)):
        self._learn_logic_fcn = learn_logic_fcn
        self._learn_math_fcn = learn_math_fcn

    def loop(self):
        ICONST = 1
        print('exit - выйти, mode_logic | mode_math')
        shell_running = True
        while shell_running:
            s = input('->')
            if s == 'exit':
                break
            if s == 'mode_logic':
                while shell_running:
                    s1 = input('L->')
                    if s1 == 'exit':
                        break
                    b_c, ans=self._learn_logic_fcn.predict_spec("logic", s1)
                    if b_c == ICONST:
                        print("answer %d" % ans)
                    else:
                        print("uncnown answer")
            elif s == 'mode_math':
                while shell_running:
                    s2 = input('M->')
                    if s2 == 'exit':
                        break
                    b_c, ans=self._learn_math_fcn.predict_spec("math", s2)
                    if b_c == ICONST:
                        print("answer %d" % ans)
                    else:
                        print("uncnown answer")


def main():
    v = VmTest()
    v.loop()


main()
