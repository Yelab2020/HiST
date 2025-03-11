import os
import anndata
import numpy as np
import scanpy as sc
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from collections import OrderedDict

def convert_to_positive(arr):
    min_val_abs = np.abs(np.min(arr))
    converted_arr = arr + min_val_abs
    return converted_arr

def Convert2Anndata(
    sample_id:str,
    gene_matrix:np.ndarray,
    gene_list:list,
    tissue_positions_path:str,
    scale_factor_path:str,
    HE_path:str,
    img_format = '.jpg'
):
    rows = []
    for row in range(gene_matrix.shape[1]):
        for col in range(gene_matrix.shape[2]):
            gene_value = gene_matrix[0,row,col]
            col_adj = col * 2 if row % 2 == 0 else col * 2 + 1
            rows.append({'row': row, 'col': col_adj})
    df = pd.DataFrame.from_records(rows)
    data = pd.DataFrame(gene_matrix.reshape((len(gene_list),-1)).transpose(),columns = gene_list)
    df = pd.concat([df,data],axis = 1)
    coord = pd.read_csv(os.path.join(tissue_positions_path,sample_id+'.csv'))
    merged_df = coord.merge(df,on=['row','col'],how='left')
    scalefactor_df = pd.read_csv(os.path.join(scale_factor_path,sample_id+'.csv'))
    scalefactor_hires = scalefactor_df['hires'][0]
    scalefactor_lowres = scalefactor_df['lowres'][0]
    HE_img = Image.open(os.path.join(HE_path,sample_id+img_format))
    if max(HE_img.size[0],HE_img.size[1]) > 3000:
        height_adj = int(HE_img.size[0] * scalefactor_hires)
        width_adj = int(HE_img.size[1] * scalefactor_hires)
        HE_img_hires = HE_img.resize((height_adj,width_adj))
        height_adj = int(HE_img.size[0] * scalefactor_lowres)
        width_adj = int(HE_img.size[1] * scalefactor_lowres)
        HE_img_lowres = HE_img.resize((height_adj,width_adj))
        HE_img_lowres = np.array(HE_img_lowres,dtype = np.float32)/255
    else:
        HE_img_hires = HE_img
        height_adj = int(HE_img.size[0] / scalefactor_hires * scalefactor_lowres)
        width_adj = int(HE_img.size[1] / scalefactor_hires * scalefactor_lowres)
        HE_img_lowres = HE_img.resize((height_adj,width_adj))
        HE_img_lowres = np.array(HE_img_lowres,dtype = np.float32)/255
    HE_img_hires = np.array(HE_img_hires,dtype = np.float32)/255

    X = np.array(merged_df.iloc[:,6:],dtype = np.float32)
    if not np.all(X >= 0):
        X = convert_to_positive(X)
    sparse_X = csr_matrix(X)
    obs = merged_df.iloc[:,0:6].set_index('barcode',drop = True)
    obs.columns = ['in_tissue','array_row','array_col','imagerow','imagecol']
    var = pd.DataFrame({'feature_types' : ['Gene Expression'] * len(gene_list)},index = gene_list)
    obsm = {'spatial':np.array(merged_df.iloc[:,[5,4]])}
    uns = OrderedDict()
    scalefactor_df.columns = ['spot_diameter_fullres','fiducial_diameter_fullres','tissue_hires_scalef','tissue_lowres_scalef']
    scalefactor_dict = scalefactor_df.to_dict(orient = 'records')[0]
    image_dict = {
        'hires':HE_img_hires,
        'lowres':HE_img_lowres
    }
    sample_dict = {
        'images':image_dict,
        'scalefactors':scalefactor_dict
    }
    uns['spatial'] = {sample_id: sample_dict}

    adata = anndata.AnnData(X = sparse_X,
                           obs = obs,
                           obsm = obsm,
                           var = var,
                           uns = uns)
    
    return adata


def plot_cluster(
    adata_original,
    resolution = 0.4,
    figsize=(8, 8),
    n_top_genes=100,
    spot_size=1000,
    save_changes = False
):
    adata = adata_original.copy()
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes) #
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(
        adata,
        key_added="clusters", directed=False, n_iterations=2,
        resolution = resolution
    )
    plt.rcParams["figure.figsize"] = figsize
#     sc.pl.umap(adata, color=["clusters"], wspace=0.4)
    sc.pl.spatial(adata, img_key="hires", color="clusters", size=spot_size)
    
    if save_changes:
        return adata
    
    
def mask2col(adata,mask_array):
    adata_copy = adata.copy()
    is_tumor = []
    obs = adata_copy.obs[['array_row','array_col']]
    
    rows = np.repeat(np.arange(80), 64)
    cols = np.tile(np.arange(64), 80)
    tumor_infos = mask_array.flatten()
    mask_df = pd.DataFrame({'row': rows,
                            'col': cols,
                            'is_tumor': tumor_infos})
    
    new_col = []
    for r in obs.itertuples(index=True):
        row = r.array_row
        col = r.array_col
        col_adj = col / 2 if row % 2 == 0 else (col - 1)/2
        new_col.append(col_adj)
    obs.loc[:,'array_col'] = new_col
    
    merged_obs = obs.merge(mask_df,left_on = ['array_row','array_col'], right_on = ['row','col'], how = "left")
    is_tumor_col = merged_obs['is_tumor'].copy()
    indice1 = is_tumor_col>0.5
    indice2 = is_tumor_col<=0.5
    is_tumor_col.loc[indice1] = "Tumor"
    is_tumor_col.loc[indice2] = "Normal"
    
    return list(is_tumor_col)