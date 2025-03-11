import os
import torch
import numpy as np
from PIL import Image
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Image.MAX_IMAGE_PIXELS = 10000000000


def SavePredictMask(
    val_loader,
    val_id,
    cmu_model,
    img_format='jpg',
    HE_path='./data/HE/',
    mask_path='./data/mask_png/0/',
):
    
    current_date = date.today().strftime('%Y%m%d')
    with torch.no_grad():
        for img, mask, _ in val_loader:
            img = img.to('cuda')
            output = cmu_model(img)
            output = torch.sigmoid(output)
            out_all = output[0,0,:,:].cpu().numpy()
            output = torch.where(output > 0.5, output, torch.tensor(0.0).cuda())

    img_path = os.path.join(HE_path,val_id+'.'+img_format)
    he_img = Image.open(img_path)
    width, height = he_img.size
    if max(width, height) > 3000:
        he_img = he_img.resize((width//20, height//20))
    he_img = np.array(he_img)
    
    output = output[0,0,:,:].cpu().numpy()
    truemask = mpimg.imread(os.path.join(mask_path,val_id+'.png'))
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(he_img)
    axs[0].set_title(val_id + ' HE image')
    axs[1].imshow(output, cmap='gray')
    axs[1].set_title(val_id + ' Predict')
    axs[2].imshow(truemask, cmap='gray')
    axs[2].set_title(val_id + ' Ground Truth')

    plt.tight_layout()
    plt.savefig('checkpoint/%s/%s_PredictMask.png' % (current_date,val_id))
    plt.show()
    
    return out_all, output