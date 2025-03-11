import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CoxDataset(Dataset):
    def __init__(self,
                 clinical_df: pd.DataFrame,
                 features, # list[list]
                 HE_dim: int = 768,
                 method: str = 'all'
                 ):
        '''
        Args:
            clinical_df: clinical dataframe
            he_features: list of features
            method: 'all' or 'gene' or 'mask' or 'he' or 'gene+mask' or 'gene+he' or 'mask+he'
        '''
        super().__init__()
        self.features = features
        self.HE_dim = HE_dim
        self.clinical_df = clinical_df
        self.event = clinical_df['OS']
        self.time = clinical_df['OS.time']
        self.method = method
        
        valid_methods = ['all', 'gene', 'mask', 'he', 'gene+mask', 'gene+he', 'mask+he']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, but got {self.method}")
        
    def __len__(self):
        return len(self.event)

    def __getitem__(self, idx):
        features = self.features
        method = self.method
        clinical = self.clinical_df.iloc[idx,3:]
        clinical = np.array(clinical.astype(int))
        event = self.event[idx]
        time = self.time[idx]
        
        if method == 'all':
            gene_matrix = torch.tensor(features[0][idx])
            mask_matrix = torch.tensor(features[1][idx])
            he_feature = features[2][idx].view(80,64,self.HE_dim).permute(2,0,1)
            matrix = torch.cat([gene_matrix, mask_matrix, he_feature], dim=0)
        elif method == 'gene':
            gene_matrix = torch.tensor(features[0][idx])
            matrix = gene_matrix
        elif method == 'mask':
            mask_matrix = torch.tensor(features[0][idx])
            matrix = mask_matrix
        elif method == 'he':
            he_feature = features[0][idx].view(80,64,self.HE_dim).permute(2,0,1)
            matrix = he_feature
        elif method == 'gene+mask':
            gene_matrix = torch.tensor(features[0][idx])
            mask_matrix = torch.tensor(features[1][idx])
            matrix = torch.cat([gene_matrix, mask_matrix], dim=0)
        elif method == 'gene+he':
            gene_matrix = torch.tensor(features[0][idx])
            he_feature = features[1][idx].view(80,64,self.HE_dim).permute(2,0,1)
            matrix = torch.cat([gene_matrix, he_feature], dim=0)
        elif method == 'mask+he':
            mask_matrix = torch.tensor(features[0][idx])
            he_feature = features[1][idx].view(80,64,self.HE_dim).permute(2,0,1)
            matrix = torch.cat([mask_matrix, he_feature], dim=0)
        
        return matrix, clinical, event, time
    
    
    
    
class LabelDataset(Dataset):
    def __init__(self,
                 features, # list[list]
                 labels: list,
                 HE_dim: int = 768,
                 method: str = 'all'
                 ):
        '''
        Args:
            features: list of features
            labels: list of labels
            method: 'all' or 'gene' or 'mask' or 'he' or 'gene+mask' or 'gene+he' or 'mask+he'
        '''
        super().__init__()
        self.features = features
        self.HE_dim = HE_dim
        self.labels = labels
        self.method = method

        valid_methods = ['all', 'gene', 'mask', 'he', 'gene+mask', 'gene+he', 'mask+he']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, but got {self.method}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features
        method = self.method
        label = self.labels[idx]

        if method == 'all':
            gene_matrix = torch.tensor(features[0][idx])
            mask_matrix = torch.tensor(features[1][idx])
            he_feature = features[2][idx].view(80,64,self.HE_dim).permute(2,0,1)
            matrix = torch.cat([gene_matrix, mask_matrix, he_feature], dim=0)
        elif method == 'gene':
            gene_matrix = torch.tensor(features[0][idx])
            matrix = gene_matrix
        elif method == 'mask':
            mask_matrix = torch.tensor(features[0][idx])
            matrix = mask_matrix
        elif method == 'he':
            he_feature = features[0][idx].view(80,64,self.HE_dim).permute(2,0,1)
            matrix = he_feature
        elif method == 'gene+mask':
            gene_matrix = torch.tensor(features[0][idx])
            mask_matrix = torch.tensor(features[1][idx])
            matrix = torch.cat([gene_matrix, mask_matrix], dim=0)
        elif method == 'gene+he':
            gene_matrix = torch.tensor(features[0][idx])
            he_feature = features[1][idx].view(80,64,self.HE_dim).permute(2,0,1)
            matrix = torch.cat([gene_matrix, he_feature], dim=0)
        elif method == 'mask+he':
            mask_matrix = torch.tensor(features[0][idx])
            he_feature = features[1][idx].view(80,64,self.HE_dim).permute(2,0,1)
            matrix = torch.cat([mask_matrix, he_feature], dim=0)
        
        return matrix, label