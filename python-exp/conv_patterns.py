import numpy as np
import matplotlib.pyplot as plt
import pickle

layers = None
file = "layers-stoch-32-64-989percent.pkl"
#with open(file) as h:
with open("layers-adam-001-32-64.pkl") as h:
    layers = pickle.load(h) 

conv_layer = layers[0]
    
plt.gray()
plt.axis('off')
num_patterns = conv_layer.Wxz.shape[0]
sqrt_n = int(np.sqrt(num_patterns))+1
for i in range(num_patterns):
    w = conv_layer.Wxz[i]
    s = str(sqrt_n)+str(sqrt_n)+str(i)
    plt.subplot(sqrt_n, sqrt_n, i+1)
    plt.imshow(w.reshape(w.shape[1], w.shape[2]))
    #plt.imshow(w.reshape(w.shape[1], w.shape[2]), interpolation="nearest")
plt.show()


