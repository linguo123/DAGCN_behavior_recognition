import numpy as np
import torch
import  matplotlib.pyplot as plt
import seaborn as sns
#sns.set_theme()

path = './A_l10.pt'
a_pt =  torch.load(path)
npimg =  a_pt.cpu().detach().numpy()
img = np.transpose(npimg[1],(1,2,0))




for i in range(32):
    #
    #img[:, :, i]
    sns.heatmap(img[:, :, i])
    #plt.imshow(img[:, :, i])
    A_ = './A_l'+str(i)+'.jpg'
    plt.savefig(A_)
    plt.show()




print(npimg.shape)
# A_image =  torch.mak


