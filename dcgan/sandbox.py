import numpy as np
import matplotlib.pyplot as plt

def torch_to_numpy(img, batch=False):
    
    if batch:
        return img.detach().cpu().numpy().transpose(0, 2, 3, 1)
    else:
        return img.detach().cpu().numpy().transpose(1, 2, 0)
    
def torch_imshow_batch(img_batch, row, col):
    img_batch = img_batch * 0.5 + 0.5;
    img_batch = torch_to_numpy(img_batch, batch=True)
    
    img_size = img_batch.shape[1:]
    img_to_show = np.empty([row, col, *img_size])
    
    
    for r in range(row):
        for c in range(col):
            img_to_show[r, c] = img_batch[r*col + c]
     
    img_to_show = np.concatenate(img_to_show,axis=1)
    img_to_show = np.concatenate(img_to_show,axis=1)
    
    plt.imshow(img_to_show)
    plt.show()
    
    
torch_imshow_batch(x_gen, 8, 4)
