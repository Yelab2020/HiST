import os
import numpy as np
from PIL import Image
import rpy2.robjects as robjects

# os.environ['R_HOME'] = '/work/usr/software/miniconda3/envs/hiST/lib/R'

def MaskRds2png(rds_path, type_num:int, out_path):
    sample_rds = os.path.basename(rds_path)
    sample_id = sample_rds.split('.')[0]
    readRDS = robjects.r['readRDS']
    mask_rds = readRDS(rds_path)
    mask_array = np.array(mask_rds)
    for i in range(type_num):
        channel_i_array = mask_array[:,:,i]*255
        mask = Image.fromarray(channel_i_array.astype(np.uint8), mode='L')
        type_path = os.path.join(out_path,'%d'%i)
        if not os.path.exists(type_path):
            os.makedirs(type_path)
        mask.save(os.path.join(type_path,(sample_id+'.png')))
        
        
# for sample in rds_paths:
#     Mask_rds2png(sample, type_num=3, out_path = './data/mask_png/')