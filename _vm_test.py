from _learn_logic_as_symbs import LearnLogicAsSymbs

class VmTest:
    def __init__(self, learn_logic_fcn = LearnLogicAsSymbs(load_f_name='wei.my')):
       self._learn_logic_fcn=learn_logic_fcn
    def loop(self):
        ICONST = 1
        print('exit - выйти')
        shell_running=True
        while shell_running:
            s=input('->')
            if s=='exit':
                break
            b_c, ans = self._learn_logic_fcn.predict_spec("logic", s)
            if b_c == ICONST:
              print("answer %d"%ans)
            else:
               print("uncnown answer")



def main():
    v= VmTest()
    v.loop()

main()    
