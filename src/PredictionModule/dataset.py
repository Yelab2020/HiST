import os
import cv2
import torch
import numpy as np
import rpy2.robjects as robjects


def rds2geneMatirx(rds_path):
    sample_rds = rds_path.split('/')[-1]
    sample_id = sample_rds.split('.')[0]
    readRDS = robjects.r['readRDS']
    rds = readRDS(rds_path)
    array = np.array(rds)
    return array

class GeneDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, tensor_lists, rds_path, num_genes):
        self.img_ids = img_ids
        self.tensor_lists = tensor_lists
        self.gene_dir = rds_path
        self.num_genes = num_genes

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img = self.tensor_lists[idx]
        img = img.view(80,64,768)
        img = img.permute(2,0,1)
        
        img_id = self.img_ids[idx]
        
        gene_matrix = rds2geneMatirx(os.path.join(self.gene_dir,self.img_ids[idx]+'.rds.gz'))
        gene_matrix = gene_matrix.astype('float32')
        gene_matrix = gene_matrix.transpose(2,0,1)
        
        return img, gene_matrix, {'img_id': img_id}
    
    
class TumorDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, tensor_lists, mask_dir, mask_ext, num_classes):
        self.img_ids = img_ids
        self.tensor_lists = tensor_lists
        self.mask_dir = mask_dir
        self.mask_ext = mask_ext
        self.num_classes = num_classes

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img = self.tensor_lists[idx]
        img = img.view(80,64,768)
        img = img.permute(2,0,1)
        
        img_id = self.img_ids[idx]
        
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)
        
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}