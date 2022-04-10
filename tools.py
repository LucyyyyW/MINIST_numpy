import matplotlib.pyplot as plt
import numpy as np

def draw_loss(loss,index,path,stat):
    plt.plot(loss,index)
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.title('%s'%stat)
    plt.savefig(path)
    return

def save_model(params, path):
    np.save(path, params)

def load_model(path):
    params = np.load(path).item()
    key = list(params.keys())
    return params[key[-1]]