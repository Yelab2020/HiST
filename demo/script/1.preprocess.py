import os
import sys
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append("../../src")

from util.patch import *
from util.seed import seed_torch
from util.mask2png import MaskRds2png
from FeatureExtraction.feature import extract_features, load_features

seed = 42
seed_torch(seed)
# torch.set_num_threads(50)
# torch.cuda.set_per_process_memory_fraction(0.3)
Image.MAX_IMAGE_PIXELS = 100000000000

sample_list = ['CRC1','CRC2','M1042T']
HE_dir = "../data/HE"
position_dir = "../data/tissue_positions_list"
scalefactor_dir = "../data/scale_factor"

for sample_id in sample_list:
    print(sample_id, "start")
    HE_path = os.path.join(HE_dir,sample_id+".jpg")
    position_path = os.path.join(position_dir,sample_id+".csv")
    scalefactor_path = os.path.join(scalefactor_dir,sample_id+".csv")
    tile_fullres(
        HE_path,
        out_path = "../output/tile",
        sample_id = sample_id,
        position_path = position_path,
        scalefactor_path = scalefactor_path)
    
all_sample_features = extract_features(
                            tile_path = '../output/tile',
                            img_ids = sample_list,
                            model_weight_path = '../../resource/ctranspath.pth',
                            save = True,
                            seed = seed,
                            file = '../output/features/all_sample_features.pkl')

# all_sample_features = load_features('../output/features/all_sample_features.pkl')