import os
import time
import torch
import pickle
import torch.nn as nn
from tqdm import tqdm
from util.seed import seed_torch
from torchvision import transforms, models
from FeatureExtraction.model import ctranspath
from FeatureExtraction.dataset import RoiDataset


def extract_features(
        tile_path:str,
        img_ids:list,
        model_weight_path = './resource/ctranspath.pth',
        save = True,
        seed = 42,
        file = './features/all_sample_features.pkl'
):
    seed_torch(seed)
    dir_path = '/'.join(file.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok = True)
        
    ctranspath_model = ctranspath()
    ctranspath_model.head = nn.Identity()
    ctranspath_model = ctranspath_model.to("cuda")
    td = torch.load(model_weight_path)
    ctranspath_model.load_state_dict(td['model'], strict=True)
    ctranspath_model.eval()

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trnsfrms_valid = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]
    )

    time_start = time.time()

    #All dataset
    all_sample_features = []
    with tqdm(
                total=len(img_ids),
                desc="Img Feature Extracting",
                bar_format="{l_bar}{bar} [ time left: {remaining} ]",
            ) as pbar:
        for sample_id in img_ids:
            patch_list = os.listdir(tile_path+'/'+sample_id)
            patch_list = [file for file in patch_list if file.lower().endswith(('jpeg', 'jpg'))]
            sorted_patch_list = sorted(patch_list, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
            patch_sorted_full_path = [os.path.join(tile_path+'/'+sample_id, file_name) for file_name in sorted_patch_list]
            test_datat=RoiDataset(patch_sorted_full_path,trnsfrms_valid)
            database_loader = torch.utils.data.DataLoader(test_datat, batch_size=80, shuffle=False, drop_last=False, num_workers=0)
            all_features = []
            with torch.no_grad():
                for batch in database_loader:
                    batch = batch.to("cuda")
                    features = ctranspath_model(batch)
                    all_features.append(features)
            all_features = torch.cat(all_features, dim=0)
            all_sample_features.append(all_features)
            pbar.set_description(desc=f"{sample_id} Feature Extracting")
            pbar.update(1)
        
    all_sample_features = [tensor.cpu() for tensor in all_sample_features]

    time_end = time.time()
    time_cost = time_end - time_start
    print('Extracting Feature Time Cost: %f min'%(time_cost/60))

    if save:
        with open(file,'wb') as f:
            pickle.dump(all_sample_features,f)

    return all_sample_features


def load_features(
        feature_path:str
):
    with open(feature_path,'rb') as f:
        all_sample_features = pickle.load(f)

    return all_sample_features