import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append("../../src")

import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import PIL.Image as Image
import matplotlib.pyplot as plt
from util.predict import *
from util.cluster import *
from util.seed import seed_torch
from FeatureExtraction.feature import load_features
from PredictionModule.solver import TumorSolver, GeneSolver

seed = 42
seed_torch(seed)
Image.MAX_IMAGE_PIXELS = 100000000000

sample_list = ['CRC1','CRC2','M1042T']
gene_list = list(pd.read_csv('../../resource/CRC_SVG346_list.txt',header=None).iloc[:,0])

print(f"Samples num:{len(sample_list)}",
      f"Gene num:{len(gene_list)}")

all_sample_features = load_features('../output/features/all_sample_features.pkl')


print("Tumor model start training")
Tumorsolver = TumorSolver(
    seed = seed,
    mask_ch=1, img_ch=768, epochs=200, lr=0.001, weight_decay=1e-4, verbose=False
)

Tumorsolver.train_loo(
    sample_list=sample_list,
    all_sample_features=all_sample_features,
    batch_size=1,
    mask_dir='../data/mask_png',
    mask_ext='.png',
    HE_path='../data/HE',
    HE_ext='.jpg',
    out_dir='../output/model/tumor/checkpoint_loo/'
)

checkpoint_dir = '../output/model/tumor/checkpoint_loo/'
subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
latest_dir = subdirs[-1]

with open(os.path.join(checkpoint_dir, latest_dir,'predict_masks/predict_masks.pkl'), 'rb') as f:
    predict_masks = pickle.load(f)
    
for sample in sample_list:
    VisualizeTumorST(
            sample_id=sample,
            sample_list=sample_list,
            predict_masks=predict_masks,
            tissue_positions_path = '../data/tissue_positions_list/',
            scale_factor_path = '../data/scale_factor/',
            HE_path = '../data/HE/',
            mask_path = '../data/mask_png/0/',
            img_format = '.jpg',
            scale = True,
            predict=True,
            compare=True,
            alpha=0.5,
            cmap=plt.cm.get_cmap('Set3'),
            title=True,
            out_dir=os.path.join(checkpoint_dir, latest_dir,'predict_results')
    )

print("Gene model start training")
Genesolver = GeneSolver(
    seed = seed,
    img_ch=768, epochs=200, lr=0.001, weight_decay=1e-4, verbose=False
)

Genesolver.train_loo(
        gene_list=gene_list,
        sample_list=sample_list,
        all_sample_features=all_sample_features,
        batch_size=1,
        gene_dir='../data/geneMatrix/',
        out_dir='../output/model/gene/checkpoint_loo/'
)

checkpoint_dir = '../output/model/gene/checkpoint_loo/'
subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
latest_dir = subdirs[-1]

pearson_df = pd.read_csv(os.path.join(checkpoint_dir, latest_dir,f'cor_df/pearson_cor.csv'),index_col=0)
print("Top 10 genes with high Pearson correlation:\n",pearson_df.mean(axis = 1).sort_values(ascending = False).head(10))

for sample in sample_list:
    VisualizeGeneST(
        sample_id=sample,
        sample_list=sample_list,
        gene='ACTB',
        gene_list=gene_list,
        all_sample_features=all_sample_features,
        model_path=os.path.join(checkpoint_dir, latest_dir,f'last_model/{sample}_200_model.pth'),
        rds_path='../data/geneMatrix/',
        tissue_positions_path = '../data/tissue_positions_list/',
        scale_factor_path = '../data/scale_factor/',
        HE_path = '../data/HE/',
        img_format='.jpg',
        scale = True,
        predict=True,
        compare=True,
        alpha=0.5,
        title=True,
        out_dir=os.path.join(checkpoint_dir, latest_dir,'visualization')
    )
    
plot_df = pd.concat([pearson_df],keys = ('Pearson'))
plot_df = plot_df.stack()
plot_df = plot_df.rename_axis(index = ['HiST','gene','sample'])
plot_df = plot_df.reset_index(level = [0,2], name = 'correlation')
sns.set(style="whitegrid")
plt.figure(figsize=(3, 6))
ax = sns.boxplot(data = plot_df, x='sample', hue = 'HiST' ,y = 'correlation',showfliers = False)
ax.get_legend().remove()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.figure.savefig(os.path.join(checkpoint_dir, latest_dir,'visualization',"loo_correlation_benchmark.pdf"),bbox_inches = 'tight')

cor_df = pd.DataFrame({'HiST': pearson_df.values.flatten().tolist()})
plt.figure(figsize=(3, 6))
ax = sns.violinplot(data=cor_df)
sns.set(style="whitegrid")
ax.set_ylabel('Pearson Correlation')
ax.figure.savefig(os.path.join(checkpoint_dir, latest_dir,'visualization',"violin_correlation_benchmark.pdf"),bbox_inches = 'tight')