import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append("../../src")

import pandas as pd
from util.predict import VisualizeGeneST
from FeatureExtraction.feature import load_features

sample_list = ['CRC1','CRC2','M1042T']
gene_list = list(pd.read_csv('../../resource/CRC_SVG346_list.txt',header=None).iloc[:,0])

all_sample_features = load_features('../output/features/all_sample_features.pkl')

VisualizeGeneST(
    sample_id='M1042T',
    sample_list=sample_list,
    gene='ACTB',
    gene_list=gene_list,
    all_sample_features=all_sample_features,
    model_path='../data/model_weights/M1042T_200_model.pth',
    rds_path='../data/geneMatrix',
    tissue_positions_path='../data/tissue_positions_list',
    scale_factor_path='../data/scale_factor',
    HE_path='../data/HE',
    img_format='.jpg',
    scale = True,
    predict=True,
    compare=True,
    alpha=0.5,
    title=True,
    out_dir='../output/reproduction_Figure3b'
)