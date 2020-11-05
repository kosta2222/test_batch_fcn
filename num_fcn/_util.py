import numpy as np
import matplotlib.pyplot as plt


class Util:
    def plot_gr(self, _file: str, errors: list, epochs: list) -> None:
        fig: plt.Figure = None
        ax: plt.Axes = None
        fig, ax = plt.subplots()
        ax.plot(epochs, errors,
                label="learning",
                )
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        ax.legend()
        plt.savefig(_file)
        print("Graphic saved")
        plt.show()

    def mse(self, y):
        return np.sum(np.square(y))
