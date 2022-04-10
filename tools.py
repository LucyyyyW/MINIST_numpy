import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def draw_loss(loss,index,path,stat):
    plt.plot(loss,index)
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.title('%s'%stat)
    plt.savefig(path)
    plt.clf()
    return

def save_model(params, path):
    np.save(path, params)

def load_model(path):
    params = np.load(path, allow_pickle=True).item()
    key = list(params.keys())
    return params['last']

def draw_params(weight, path):
    fig = plt.figure(figsize=(12,8))
    ax = Axes3D(fig)
    a,b = weight.shape
    x = np.arange(0,b,1)
    y = np.arange(0,a,1)
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,weight)
    plt.title('layer1_weight')


