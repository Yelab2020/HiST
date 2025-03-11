import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import pearsonr, spearmanr


Image.MAX_IMAGE_PIXELS = 10000000000


def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)
    return acc


def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
        # TP : True Positive
        # FN : False Negative
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
        # TN : True Negative
        # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return SP

def get_precision(SR,GT,threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT== torch.max(GT)
        # TP : True Positive
        # FP : False Positive
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    return PC

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    
    output_ = torch.tensor(output_)
    target_=torch.tensor(target_)
    SE = get_sensitivity(output_,target_,threshold=0.5)
    PC = get_precision(output_,target_,threshold=0.5)
    SP= get_specificity(output_,target_,threshold=0.5)
    ACC=get_accuracy(output_,target_,threshold=0.5)
    F1 = 2*SE*PC/(SE+PC + 1e-6)
    return iou, dice , SE, PC, F1,SP,ACC


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
        

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        



def calculate_correlations(gene_list, val_id, val_loader, model, out_dir):
    """Calculate Correlations"""
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
        
    val_output_list = []
    val_real_list = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for img, mask, _ in val_loader:
            img = img.to(device)
            output = model(img)
            val_output_list.append(output)
            val_real_list.append(mask.to(device))

    pearson_cors, pearson_pvalues = [], []
    spearman_cors, spearman_pvalues = [], []
    for tensor1, tensor2 in zip(val_output_list, val_real_list):
        for n in range(val_output_list[0].shape[1]):
            tensor1_flat = tensor1[:, n, :, :].view(-1).cpu()
            tensor2_flat = tensor2[:, n, :, :].view(-1).cpu()
            pearson_cor, pearson_pvalue = pearsonr(tensor1_flat, tensor2_flat)
            spearman_cor, spearman_pvalue = spearmanr(tensor1_flat, tensor2_flat)
            pearson_cors.append(pearson_cor)
            pearson_pvalues.append(pearson_pvalue)
            spearman_cors.append(spearman_cor)
            spearman_pvalues.append(spearman_pvalue)
    
    correlation_results = {
        'gene': gene_list,
        'Pearson_Correlation': pearson_cors,
        'Pearson_Pvalue': pearson_pvalues,
        'Spearman_Correlation': spearman_cors,
        'Spearman_Pvalue': spearman_pvalues
    }
    df_correlations = pd.DataFrame(correlation_results)
    file = os.path.join(out_dir, val_id + '_correlations.csv')
    df_correlations.to_csv(file, index=False)
    
    return df_correlations




def SavePredictMask(
    val_loader,val_id,cmu_model,
    HE_path : str = './CRC/data/HE/',
    HE_ext : str = '.jpg',
    mask_path : str = './data/mask_png/',
    mask_ext : str = '.png',
    mask_class : int = 0,
    out_dir : str = 'checkpoint_tumor_loo',
    verbose : bool = True
):
    
    with torch.no_grad():
        for img, mask, _ in val_loader:
            img = img.to('cuda')
            output = cmu_model(img)
            output = torch.sigmoid(output)
            out_all = output[0,0,:,:].cpu().numpy()
            output = torch.where(output > 0.5, output, torch.tensor(0.0).cuda())

    img_path = os.path.join(HE_path, val_id + HE_ext)
    he_img = Image.open(img_path)
    width, height = he_img.size
    he_img = he_img.resize((width//20, height//20))
    he_img = np.array(he_img)
    
    output = output[0,0,:,:].cpu().numpy()
    truemask = mpimg.imread(os.path.join(mask_path, str(mask_class), val_id + mask_ext))
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(he_img)
    axs[0].set_title(val_id + ' HE Image')
    axs[1].imshow(output, cmap='gray')
    axs[1].set_title(val_id + ' Predict')
    axs[2].imshow(truemask, cmap='gray')
    axs[2].set_title(val_id + ' Ground Truth')

    plt.tight_layout()
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, val_id + '_PredictMask.png'))
    if verbose:
        plt.show()
    
    return out_all, output