import os
import scipy
import torch
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import PIL.Image as Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from CMUNet.model import CMUNet
from util.seed import seed_torch
import matplotlib.colors as mcolors
from CMUNet.dataset import GeneDataset, TumorDataset
from palettable.colorbrewer.diverging import RdYlBu_10_r

Image.MAX_IMAGE_PIXELS = 10000000000

def NormalizeColor(gene_values):
    q1 = np.percentile(gene_values,25)
    q3 = np.percentile(gene_values,75)
    iqr = q3 - q1
    fen_low = q1-1.5*iqr
    fen_high = q3+1.5*iqr
    minima = max(fen_low, min(gene_values))
    maxima = min(fen_high, max(gene_values))
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=RdYlBu_10_r.mpl_colormap)
    mapped_rgb = mapper.to_rgba(gene_values, alpha=None, bytes=True, norm=True)
    mapped_rgb = mapped_rgb / 255.0
    return mapped_rgb, mapper, minima, maxima


def _norm(original_list):
    min_value = min(original_list)
    max_value = max(original_list)
    scaled_list = [(x - min_value) / (max_value - min_value) for x in original_list]
    return scaled_list


def NormalizeColor_together(gene_values1,gene_values2):
    gene_values1 = _norm(gene_values1)
    gene_values2 = _norm(gene_values2)
    gene_values = list(gene_values1)+list(gene_values2)
    q1 = np.percentile(gene_values,25)
    q3 = np.percentile(gene_values,75)
    iqr = q3 - q1
    fen_low = q1-1.5*iqr
    fen_high = q3+1.5*iqr
    minima = max(fen_low, min(gene_values))
    maxima = min(fen_high, max(gene_values))
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=RdYlBu_10_r.mpl_colormap)
    mapped_rgb1 = mapper.to_rgba(gene_values1, alpha=None, bytes=True, norm=True)
    mapped_rgb1 = mapped_rgb1 / 255.0
    mapped_rgb2 = mapper.to_rgba(gene_values2, alpha=None, bytes=True, norm=True)
    mapped_rgb2 = mapped_rgb2 / 255.0
    return mapped_rgb1, mapped_rgb2, mapper, minima, maxima


def ValidateSampleGene(
    sample_id:str,
    sample_list:list,
    gene_list:list,
    all_sample_features:list,
    model_path:str,
    rds_path:str='./data/geneMatrix3/normed',
    seed:int=42
):
    seed_torch(seed)
    val_idx = [sample_list.index(sample_id)]
    val_dataset = GeneDataset(
        img_ids=[sample_list[i] for i in val_idx],
        tensor_lists=[all_sample_features[i] for i in val_idx],
        rds_path=rds_path,
        num_genes=len(gene_list))
    
    val_loader_nobatch = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False)
    
    val_output_list = []
    val_real_list = []

    cmu_model = CMUNet(img_ch=768, output_ch=len(gene_list), l=7, k=7)
    cmu_model = cmu_model.cuda()
    
    cmu_model.load_state_dict(torch.load(model_path))
    cmu_model.eval()
    
    with torch.no_grad():
        for img, gene_matrix, _ in val_loader_nobatch:
            img = img.to('cuda')
            output = cmu_model(img)
            val_output_list.append(output)
            val_real_list.append(gene_matrix.to('cuda'))
        
    predict_gene_matrix = val_output_list[0].squeeze(0).cpu().numpy()

    real_gene_matrix = val_real_list[0].squeeze(0).cpu().numpy()
    
    return predict_gene_matrix[:,1:-1,:],real_gene_matrix[:,1:-1,:]


def ValidateSampleTumor(
    sample_id:str,
    sample_list:list,
    all_sample_features:list,
    model_path:str,
    mask_dir:str,
    mask_ext:str='.png',
    num_classes:int=1,
    sigmoid:bool=False,
    seed:int=42
):
    seed_torch(seed)
    val_idx = [sample_list.index(sample_id)]
    val_dataset = TumorDataset(
        img_ids=[sample_list[i] for i in val_idx],
        tensor_lists=[all_sample_features[i] for i in val_idx],
        mask_dir=mask_dir,
        mask_ext=mask_ext,
        num_classes=num_classes
    )
    
    val_loader_nobatch = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False)
    
    val_output_list = []
    val_real_list = []

    cmu_model = CMUNet(img_ch=768, output_ch=num_classes, l=7, k=7)
    cmu_model = cmu_model.cuda()
    
    cmu_model.load_state_dict(torch.load(model_path))
    cmu_model.eval()
    
    with torch.no_grad():
        for img, mask_matrix, _ in val_loader_nobatch:
            img = img.to('cuda')
            output = cmu_model(img)
            if sigmoid:
                output = torch.sigmoid(output)
                # torch.where(output > 0.5, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
            val_output_list.append(output)
            val_real_list.append(mask_matrix.to('cuda'))
        
    predict_mask_matrix = val_output_list[0].squeeze(0).cpu().numpy()

    real_mask_matrix = val_real_list[0].squeeze(0).cpu().numpy()
    
    return predict_mask_matrix[:,1:-1,:],real_mask_matrix[:,1:-1,:]


def VisualizeTumorPreprocess(
    mask_array:np.ndarray,
    sample_id:str,
    tissue_positions_path:str = './data/tissue_positions_list/',
    scale_factor_path:str = './data/scale_factor/',
    HE_path:str = './data/HE/',
    img_format:str = '.jpg',
    scale = True
):
    #adj coordinates
    rows = []
    for row in range(mask_array.shape[0]):
        for col in range(mask_array.shape[1]):
            is_tumor = int(mask_array[row, col] > 0)
            col_adj = col * 2 if row % 2 == 0 else col * 2 + 1
            rows.append({'row': row, 'col': col_adj, 'is_tumor': is_tumor})
    df = pd.DataFrame.from_records(rows)
    plot_df = df[df['is_tumor'] == 1]
    #scale img
    coord = pd.read_csv(os.path.join(tissue_positions_path,sample_id+'.csv'))
    merged_df = coord.merge(plot_df,on=['row','col'],how='left').fillna(0)
    merged_df['is_tumor'] = merged_df['is_tumor'].astype(int)
    scalefactor_df = pd.read_csv(os.path.join(scale_factor_path,sample_id+'.csv'))
    scalefactor_hires = scalefactor_df['hires'][0]
    HE_img = Image.open(os.path.join(HE_path,sample_id+img_format))
    if scale:
        height_adj = int(HE_img.size[0] * scalefactor_hires)
        width_adj = int(HE_img.size[1] * scalefactor_hires)
        HE_img = HE_img.resize((height_adj,width_adj))
    
    return HE_img,merged_df,scalefactor_hires



def VisualizeGenePreprocess(
    gene_matrix:np.ndarray,
    sample_id:str,
    gene:str,
    gene_list:list,
    tissue_positions_path:str = './data/tissue_positions_list/',
    scale_factor_path:str = './data/scale_factor/',
    HE_path:str = './data/HE/',
    img_format:str = '.jpg',
    scale = True
):
    #adj coordinates
    rows = []
    for row in range(gene_matrix.shape[1]):
        for col in range(gene_matrix.shape[2]):
            gene_value = gene_matrix[gene_list.index(gene),row,col]
            col_adj = col * 2 if row % 2 == 0 else col * 2 + 1
            rows.append({'row': row, 'col': col_adj, 'gene_values': gene_value})
    df = pd.DataFrame.from_records(rows)
    #scale img
    coord = pd.read_csv(os.path.join(tissue_positions_path,sample_id+'.csv'))
    merged_df = coord.merge(df,on=['row','col'],how='left')
    scalefactor_df = pd.read_csv(os.path.join(scale_factor_path,sample_id+'.csv'))
    scalefactor_hires = scalefactor_df['hires'][0]
    HE_img = Image.open(os.path.join(HE_path,sample_id+img_format))
    if scale:
        height_adj = int(HE_img.size[0] * scalefactor_hires)
        width_adj = int(HE_img.size[1] * scalefactor_hires)
        HE_img = HE_img.resize((height_adj,width_adj))
    
    return HE_img,merged_df,scalefactor_hires


def load_mask(mask_path):
    mask = Image.open(mask_path)
    mask = np.array(mask)
    return mask[1:-1, :]

def plot_tumor_image(
    ax,
    HE_img,
    merged_df,
    scalefactor_hires,
    plot_title, 
    alpha, 
    cmap, 
    title
):
    ax.imshow(HE_img)
    ax.scatter(merged_df['imagecol'] * scalefactor_hires,
               merged_df['imagerow'] * scalefactor_hires,
               c=merged_df['is_tumor'], marker='o', alpha= alpha, s=2, cmap = cmap)
    if title:
        ax.set_title(plot_title)
    ax.axis('off')


def plot_gene_image(
    ax,
    HE_img,
    merged_df,
    scalefactor_hires,
    plot_title,
    alpha,
    title
):
    mapped_rgb, mapper, minima, maxima = NormalizeColor(merged_df['gene_values'])
    ax.imshow(HE_img)
    ax.scatter(merged_df['imagecol'] * scalefactor_hires,
               merged_df['imagerow'] * scalefactor_hires,
               c=mapped_rgb, marker='o', alpha= alpha, s=2)
    if title:
        ax.set_title(plot_title)
    ax.axis('off')
    cbar = plt.colorbar(mapper, ax=ax)
    cbar.set_label('Expression Level')
    cbar.set_ticks([minima,maxima])
    cbar.set_ticklabels(['Low', 'High'])
    
    

def plot_gene_image_DEG(
    ax,
    HE_img,
    merged_df,
    scalefactor_hires,
    plot_title,
    alpha,
    title,
    mapped_rgb, mapper, minima, maxima
):
    ax.imshow(HE_img)
    ax.scatter(merged_df['imagecol'] * scalefactor_hires,
               merged_df['imagerow'] * scalefactor_hires,
               c=mapped_rgb, marker='o', alpha= alpha, s=2)
    if title:
        ax.set_title(plot_title)
    ax.axis('off')
    cbar = plt.colorbar(mapper, ax=ax)
    cbar.set_label('Expression Level')
    cbar.set_ticks([minima,maxima])
    cbar.set_ticklabels(['Low', 'High'])

def VisualizeGeneST_DEG(
        sample_id1:str,
        sample_id2:str,
        sample_list:list,
        gene:str,
        gene_list:list,
        all_sample_features:list,
        model_path:str,
        rds_path:str,
        tissue_positions_path:str = './data/tissue_positions_list/',
        scale_factor_path:str = './data/scale_factor/',
        HE_path:str = './data/HE/',
        img_format:str = '.jpg',
        scale = True,
        alpha=1,
        title=True,
        out_dir=None
):
    predict_gene_matrix1,real_gene_matrix1 = ValidateSampleGene(sample_id1,sample_list,gene_list,
                                                          all_sample_features,model_path,rds_path)
    predict_gene_matrix2,real_gene_matrix2 = ValidateSampleGene(sample_id2,sample_list,gene_list,
                                                          all_sample_features,model_path,rds_path)

    HE_img1, merged_df1, scalefactor_hires1 = VisualizeGenePreprocess(predict_gene_matrix1, sample_id1, gene, gene_list,
                                                                      tissue_positions_path, scale_factor_path, HE_path, img_format, scale)
    HE_img2, merged_df2, scalefactor_hires2 = VisualizeGenePreprocess(predict_gene_matrix2, sample_id2, gene, gene_list,
                                                                      tissue_positions_path, scale_factor_path, HE_path, img_format, scale)
    HE_img1, merged_df3, scalefactor_hires1 = VisualizeGenePreprocess(real_gene_matrix1, sample_id1, gene, gene_list,
                                                                      tissue_positions_path, scale_factor_path, HE_path, img_format, scale)
    HE_img2, merged_df4, scalefactor_hires2 = VisualizeGenePreprocess(real_gene_matrix2, sample_id2, gene, gene_list,
                                                                      tissue_positions_path, scale_factor_path, HE_path, img_format, scale)

    mapped_rgb1, mapped_rgb2, mapper1, minima1, maxima1 = NormalizeColor_together(merged_df1['gene_values'],merged_df2['gene_values'])
    mapped_rgb3, mapped_rgb4, mapper2, minima2, maxima2 = NormalizeColor_together(merged_df3['gene_values'],merged_df4['gene_values'])
    
    fig, axs = plt.subplots(2, 2,figsize=(10,10))
    plot_gene_image_DEG(axs[0,0], HE_img1, merged_df1, scalefactor_hires1, f'{sample_id1} Predicted: {gene}', alpha, title,
                       mapped_rgb1, mapper1, minima1, maxima1)
    plot_gene_image_DEG(axs[0,1], HE_img2, merged_df2, scalefactor_hires2, f'{sample_id2} Predicted: {gene}', alpha, title,
                       mapped_rgb2, mapper1, minima1, maxima1)
    plot_gene_image_DEG(axs[1,0], HE_img1, merged_df3, scalefactor_hires1, f'{sample_id1} Ground Truth: {gene}', alpha, title,
                       mapped_rgb3, mapper2, minima2, maxima2)
    plot_gene_image_DEG(axs[1,1], HE_img2, merged_df4, scalefactor_hires2, f'{sample_id2} Ground Truth: {gene}', alpha, title,
                       mapped_rgb4, mapper2, minima2, maxima2)
    plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir,exist_ok=True)
        plt.savefig(f'{out_dir}/{sample_id1}_{sample_id2}_{gene}.pdf')
    plt.show()


def VisualizeGeneST(
        sample_id:str,
        sample_list:list,
        gene:str,
        gene_list:list,
        all_sample_features:list,
        model_path:str,
        rds_path:str,
        tissue_positions_path:str = './data/tissue_positions_list/',
        scale_factor_path:str = './data/scale_factor/',
        HE_path:str = './data/HE/',
        img_format:str = '.jpg',
        scale = True,
        predict=False,
        compare=False,
        alpha=1,
        title=True,
        out_dir=None
):
    assert sample_id and sample_list and gene
    predict_gene_matrix,real_gene_matrix = ValidateSampleGene(sample_id,sample_list,gene_list,
                                                          all_sample_features,model_path,rds_path)
    
    if not compare:
        if predict:
            gene_matrix = predict_gene_matrix
        else:
            gene_matrix = real_gene_matrix
        
        HE_img, merged_df, scalefactor_hires = VisualizeGenePreprocess(gene_matrix, sample_id, gene, gene_list,
                                                                       tissue_positions_path, scale_factor_path, HE_path, img_format, scale)
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_gene_image(ax, HE_img, merged_df, scalefactor_hires, f'{sample_id} {f"Predicted {gene}" if predict else f"Ground Truth {gene}"}', alpha, title)
        plt.show()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f'{f"{sample_id}_{gene}_Predicted.pdf" if predict else f"{sample_id}_{gene}_GroundTruth.pdf"}'))
        
    else:  
        HE_img1, merged_df1, scalefactor_hires1 = VisualizeGenePreprocess(predict_gene_matrix, sample_id, gene, gene_list,
                                                                          tissue_positions_path, scale_factor_path, HE_path, img_format, scale)
        HE_img2, merged_df2, scalefactor_hires2 = VisualizeGenePreprocess(real_gene_matrix, sample_id, gene, gene_list,
                                                                          tissue_positions_path, scale_factor_path, HE_path, img_format, scale)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        plot_gene_image(ax1, HE_img1, merged_df1, scalefactor_hires1, f'{sample_id} Predicted', alpha, title)
        plot_gene_image(ax2, HE_img2, merged_df2, scalefactor_hires2, f'{sample_id} Ground Truth', alpha, title)
        plt.tight_layout()
        plt.show()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f'{sample_id}_{gene}_FeaturePlot.pdf'))
        
        
def VisualizeTumorST(
        sample_id:str,
        sample_list:list,
        predict_masks:list,
        tissue_positions_path:str = './data/tissue_positions_list/',
        scale_factor_path:str = './data/scale_factor/',
        HE_path:str = './data/HE/',
        mask_path:str = './data/mask_png/0/',
        img_format:str = '.jpg',
        scale = True,
        predict=False,
        compare=False,
        alpha=1,
        cmap=plt.cm.get_cmap('Set3'),
        title=True,
        out_dir=None
):
    if not compare:
        if predict:
            assert sample_list and predict_masks
            idx = sample_list.index(sample_id)
            mask_array = predict_masks[idx]
        else:
            mask_array = load_mask(os.path.join(mask_path, f'{sample_id}.png'))
        
        HE_img, merged_df, scalefactor_hires = VisualizeTumorPreprocess(mask_array, sample_id,
                                                                        tissue_positions_path, scale_factor_path, HE_path, img_format, scale)
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_tumor_image(ax, HE_img, merged_df, scalefactor_hires, f'{sample_id} {"Predicted" if predict else "Ground Truth"}', alpha, cmap, title)
        plt.show()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f'{f"{sample_id}_PredictedTumor.pdf" if predict else f"{sample_id}_GroundTruthTumor.pdf"}'))
    else:
        assert sample_list and predict_masks
        idx = sample_list.index(sample_id)
        predict_mask_array = predict_masks[idx][1:-1, :]
        real_mask_array = load_mask(os.path.join(mask_path, f'{sample_id}.png'))
        
        HE_img1, merged_df1, scalefactor_hires1 = VisualizeTumorPreprocess(predict_mask_array, sample_id,
                                                                           tissue_positions_path, scale_factor_path, HE_path, img_format, scale)
        HE_img2, merged_df2, scalefactor_hires2 = VisualizeTumorPreprocess(real_mask_array, sample_id,
                                                                           tissue_positions_path, scale_factor_path, HE_path, img_format, scale)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        plot_tumor_image(ax1, HE_img1, merged_df1, scalefactor_hires1, f'{sample_id} Predicted', alpha, cmap, title)
        plot_tumor_image(ax2, HE_img2, merged_df2, scalefactor_hires2, f'{sample_id} Ground Truth', alpha, cmap, title)
        plt.tight_layout()
        plt.show()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f'{sample_id}_TumorRegion.pdf'))
        
        
        
def PredictGeneOnhiST(
        sample_id: str,
        gene_list: list,
        sample_list: list,
        all_sample_features: list,
        model_path: str,
        seed: int = 42
):
    seed_torch(seed)
    idx = list(sample_list).index(sample_id)
    img = all_sample_features[idx]
    img = img.view(80,64,768)
    img = img.permute(2,0,1)
    img = img.unsqueeze(0)

    cmu_model = CMUNet(img_ch=768, output_ch=len(gene_list), l=7, k=7)
    cmu_model = cmu_model.cuda()

    cmu_model.load_state_dict(torch.load(model_path))
    cmu_model.eval()

    with torch.no_grad():
        img = img.to('cuda')
        predict_gene_matrix = cmu_model(img)
        
    return predict_gene_matrix.squeeze(0).cpu().numpy()



def PredictTumorOnhiST(
        sample_id: str,
        sample_list: list,
        all_sample_features: list,
        model_path: str,
        seed: int = 42,
        label: bool = False,
        mask: bool = False
):
    seed_torch(seed)
    idx = list(sample_list).index(sample_id)
    img = all_sample_features[idx]
    img = img.view(80,64,768)
    img = img.permute(2,0,1)
    img = img.unsqueeze(0)

    cmu_model = CMUNet(img_ch=768, output_ch=1, l=7, k=7)
    cmu_model = cmu_model.cuda()

    cmu_model.load_state_dict(torch.load(model_path))
    cmu_model.eval()

    with torch.no_grad():
        img = img.to('cuda')
        output = cmu_model(img)
        output = torch.sigmoid(output)
        if mask:
            predict_mask_matrix = torch.where(output > 0.5, output, torch.tensor(0.0).cuda())
        elif label:
            predict_mask_matrix = torch.where(output > 0.5, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
        else:
            predict_mask_matrix = output
        
    return predict_mask_matrix.squeeze(0).cpu().numpy()


def GetTissueData(
    sample_id,
    plot_df,
    tile_dir
):
    tile_path = os.path.join(tile_dir,sample_id)
    rows = []
    cols = []
    for file in os.listdir(tile_path):
        if file.lower().endswith(('jpeg', 'jpg')):
            tile = np.array(Image.open(os.path.join(tile_path,file)))
            if np.all(tile == 255):
                row,col,_ = file.split('-')
                rows.append(row)
                cols.append(col)
    in_tissue_pos = pd.DataFrame({'row': (np.array(rows,dtype=np.int32) * 20) + 10,
                                  'col': (np.array(cols,dtype=np.int32) * 20) + 10})
    df2drop = pd.merge(plot_df,in_tissue_pos,left_on=['row','col'],right_on=['row','col'],how='inner')
    df = pd.concat([plot_df,df2drop]).drop_duplicates(['row','col'],keep = False)
    return df
    
    
def GetGenePlotData(
    sample_id:str,
    gene:str,
    gene_list:list,
    sample_list:list,
    features:list,
    HE_path:str,
    model_path:str
):
    geneMatrix = PredictGeneOnhiST(
        sample_id=sample_id,
        gene_list=gene_list,
        sample_list=sample_list,
        all_sample_features=features,
        model_path=model_path
    )
    HE_img = Image.open(HE_path)
    HE_resized = HE_img.resize((1280,1600))
    gene_idx = gene_list.index(gene)
    gene_m = geneMatrix[gene_idx,:,:]
    rows = np.repeat(np.arange(80), 64)
    cols = np.tile(np.arange(64), 80)
    gene_values = gene_m.flatten()
    plot_df = pd.DataFrame({'row': (rows * 20) + 10,
                            'col': (cols * 20) + 10,
                            'gene_values': gene_values})
    return plot_df, HE_resized


def GetTumorPlotData(
    sample_id:str,
    sample_list:list,
    features:list,
    HE_path:str,
    model_path:str,
    label:bool=False,
    mask:bool=False
):
    mask_matrix = PredictTumorOnhiST(
        sample_id=sample_id,
        sample_list=sample_list,
        all_sample_features=features,
        model_path=model_path,
        label=label,
        mask=mask
    )
    HE_img = Image.open(HE_path)
    HE_resized = HE_img.resize((1280,1600))
    rows = np.repeat(np.arange(80), 64)
    cols = np.tile(np.arange(64), 80)
    tumor_infos = mask_matrix.flatten()
    plot_df = pd.DataFrame({'row': (rows * 20) + 10,
                            'col': (cols * 20) + 10,
                            'is_tumor': tumor_infos})
    return plot_df, HE_resized


def VisualizeGeneHE_compare(
        sample_id1:str,
        sample_id2:str,
        gene:str,
        sample_list:list,
        gene_list:list,
        features:list,
        sample_HEpath_dict:dict,
        model_path:str,
        tile_dir=None,
        alpha=1,
        out_dir=None
):
    plot_df1,HE1 = GetGenePlotData(sample_id1,gene,gene_list,sample_list,features,sample_HEpath_dict[sample_id1],model_path)
    plot_df2,HE2 = GetGenePlotData(sample_id2,gene,gene_list,sample_list,features,sample_HEpath_dict[sample_id2],model_path)
    if tile_dir:
        plot_df1 = GetTissueData(sample_id1,plot_df1,tile_dir)
        plot_df2 = GetTissueData(sample_id2,plot_df2,tile_dir)
    mapped_rgb1, mapped_rgb2, mapper, minima, maxima = NormalizeColor_together(plot_df1['gene_values'],plot_df2['gene_values'])
    fig, axs = plt.subplots(1, 2,figsize=(8, 4))
    axs[0].imshow(np.array(HE1))
    axs[0].scatter(plot_df1['col'],
                   plot_df1['row'],
                   c=mapped_rgb1, marker='o', alpha= alpha, s=2)
    axs[0].set_title(sample_id1)
    axs[0].axis('off')
    axs[1].imshow(np.array(HE2))
    axs[1].scatter(plot_df2['col'],
                   plot_df2['row'],
                   c=mapped_rgb2, marker='o', alpha= alpha, s=2)
    axs[1].set_title(sample_id2)
    axs[1].axis('off')
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f'{sample_id1}_{sample_id2}_{gene}_FeaturePlot.pdf'))
    plt.show()

def VisualizeGeneHE(
        sample_id:str,
        gene:str,
        sample_list: list,
        gene_list:list,
        features:list,
        sample_HEpath_dict:dict,
        model_path:str,
        tile_dir=None,
        alpha=1,
        out_dir=None
):
    plot_df, HE = GetGenePlotData(sample_id,gene,gene_list,sample_list,features,sample_HEpath_dict[sample_id],model_path)
    if tile_dir:
        plot_df = GetTissueData(sample_id,plot_df,tile_dir)
    mapped_rgb, mapper, minima, maxima = NormalizeColor(plot_df['gene_values'])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.array(HE))
    ax.scatter(plot_df['col'],
               plot_df['row'],
               c=mapped_rgb, marker='o', alpha= alpha, s=2)
    ax.set_title(sample_id)
    ax.axis('off')
    
    cbar = plt.colorbar(mapper, ax=ax)
    cbar.set_label('Expression Level')
    cbar.set_ticks([minima,maxima])
    cbar.set_ticklabels(['Low', 'High'])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f'{sample_id}_{gene}_FeaturePlot.pdf'))
    plt.show()
    
    
def VisualizeTumorHE(
        sample_id:str,
        sample_list:list,
        features:list,
        sample_HEpath_dict:dict,
        model_path:str,
        tile_dir=None,
        alpha=1,
        colors=["#8dd3c7","#ffed6f"],
        label = False,
        out_dir = None
):
    plot_df, HE = GetTumorPlotData(sample_id,sample_list,features,sample_HEpath_dict[sample_id],model_path,label)
    if tile_dir:
        plot_df = GetTissueData(sample_id,plot_df,tile_dir)
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.array(HE))
    scatter = ax.scatter(plot_df['col'],
                         plot_df['row'],
                         c=plot_df['is_tumor'], marker='o',
                         alpha= alpha, s=2, cmap=cmap,
                         vmin=0,vmax=1)
    ax.set_title(sample_id)
    ax.axis('off')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Tumor Probability')
    if label:
        cbar.set_ticks([0.5, 1])
    else:
        cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Normal', 'Tumor'])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f'{sample_id}_TumorRegion.pdf'))
    plt.show()
    
    
def GetPredictGMList(
    sample_list: list,
    gene_list: list,
    all_sample_features: list,
    model_path: str,
    seed: int = 42
):
    seed_torch(seed)
    predict_gene_matrix_list = []
    pbar = tqdm(sample_list, leave=True)
    for sample_id in pbar:
        pbar.set_description_str(desc = f"Processing sample: {sample_id}")
        pbar.refresh()
        predict_gene_matrix = PredictGeneOnhiST(sample_id, gene_list, sample_list,all_sample_features,
                                                model_path=model_path)
        predict_gene_matrix_list.append(predict_gene_matrix)
        
    return predict_gene_matrix_list

def GetPredictTMList(
    sample_list: list,
    all_sample_features: list,
    model_path: str,
    seed: int = 42
):
    seed_torch(seed)
    predict_mask_matrix_list = []
    pbar = tqdm(sample_list, leave=True)
    for sample_id in pbar:
        pbar.set_description_str(desc = f"Processing sample: {sample_id}")
        pbar.refresh()
        predict_mask_matrix = PredictTumorOnhiST(sample_id, sample_list,all_sample_features,
                                                model_path=model_path, mask=True)
        predict_mask_matrix_list.append(predict_mask_matrix)
        
    return predict_mask_matrix_list

def GetPredictTMList_TCGA(
    sample_list: list,
    all_sample_features: list,
    model_path: str,
    label: bool = False,
    seed: int = 42
):
    seed_torch(seed)
    predict_mask_matrix_list = []
    pbar = tqdm(sample_list, leave=True)
    for sample_id in pbar:
        pbar.set_description_str(desc = f"Processing sample: {sample_id}")
        pbar.refresh()
        predict_mask_matrix = PredictTumorOnhiST(sample_id, sample_list,all_sample_features,
                                                model_path=model_path, label=label)
        predict_mask_matrix_list.append(predict_mask_matrix)
        
    return predict_mask_matrix_list