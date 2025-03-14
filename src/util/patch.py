import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union


def tile(
    sample_id: str,
    HE_path: str,
    out_path: str,
    img = None, # PIL.Image.Image object
    target_size: int = 224,
    row: int = 80,
    col: int = 64
):
    """
    Tiling HE image that has a ratio(height/width) of 1.25 to square patches
    """
    if img is not None:
        img = img
    else:
        img = Image.open(HE_path)
    width, height = img.size
    patch_width = width // col
    patch_height = height // row
    
    out_path = os.path.join(out_path, sample_id)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    total_patches = row * col
    progress_bar = tqdm(total=total_patches, desc=f'{sample_id} processing:', position=0, leave=True)
    
    for i in range(row):
        for j in range(col):
            left = j * patch_width
            upper = i * patch_height
            right = left + patch_width
            lower = upper + patch_height
            patch = img.crop((left, upper, right, lower))
            patch = patch.resize((target_size,target_size))
            
            patch.save(os.path.join(out_path, f'{i}-{j}-{target_size}.jpg'))
            progress_bar.update(1)
    
    progress_bar.close()



def padding(HE_path):
    img = Image.open(HE_path)
    width, height = img.size
    case =  width*1.25 - height
    if case>=0:
        padding_height = int(case//2)
        padded_image = Image.new('RGB', (width, height + 2 * padding_height), (255,255,255))
        position = (0, padding_height)
        padded_image.paste(img, position)
        pad_axis = 'height'
        pad_percent = padding_height/(height + 2 * padding_height)
    else:
        padding_width = int(-(case/1.25)//2)
        padded_image = Image.new('RGB', (width + 2 * padding_width, height), (255,255,255))
        position = (padding_width, 0)
        padded_image.paste(img, position)
        pad_axis = 'width'
        pad_percent = padding_width/(width + 2 * padding_width)
        
    return padded_image, pad_axis, pad_percent




def tile_HE(
    sample_id: str,
    HE_path: str,
    out_path:str = '../data/processed_HE/',
    target_size: int = 224,
    row: int = 80,
    col: int = 64
):
    padded_image,_,_ = padding(HE_path)
    
    patch_width = padded_image.width // col
    patch_height = padded_image.height // row
    
    out_path = os.path.join(out_path, sample_id)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    total_patches = row * col
    progress_bar = tqdm(total=total_patches, desc=f'{sample_id} processing:', position=0, leave=True)
    
    for i in range(row):
        for j in range(col):
            left = j * patch_width
            upper = i * patch_height
            right = left + patch_width
            lower = upper + patch_height
            patch = padded_image.crop((left, upper, right, lower))
            patch = patch.resize((target_size,target_size))
            
            patch.save(os.path.join(out_path, f'{i}-{j}-{target_size}.jpg'))
            progress_bar.update(1)
    
    progress_bar.close()




def tile_fullres(
    sample_path: Union[Path, str],
    out_path: Union[Path, str] = "./tiling",
    sample_id: str = 'sample',
    position_path = Union[Path, str],
    scalefactor_path = Union[Path, str],
    target_size: int = 224,
    resize: bool = False,
    padding = True,
    tile_rownum = 80,
    tile_colnum = 64,
    verbose: bool = False
):


    # Check the exist of out_path
    if sample_id is not None:
        out_path = str(out_path) + '/' + sample_id
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path, exist_ok = True)

    image_pillow = Image.open(sample_path)
    image = np.array(image_pillow)
    
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)

    if img_pillow.mode == "RGBA":
        img_pillow = img_pillow.convert("RGB")

    tile_names = []

    position = pd.read_csv(position_path,
            header = 0, names = ['barcode','in_tissue','array_row','array_col','imagerow','imagecol'])
    
    #scale_factor
    scale_factor_data = pd.read_csv(scalefactor_path,header = 0)
    
    crop_size = int(scale_factor_data.iloc[0,1])#fiducial diameter
    
    if padding:
        total = tile_rownum * tile_colnum
    else:
        total = position.shape[0]
    
    with tqdm(
        total=total,
        desc=f"Tiling image",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for array_row,array_col,imagerow,imagecol in zip(position['array_row'],position['array_col'],position["imagerow"], position["imagecol"]):
            array_row = int(array_row)
            if array_row % 2 == 0:
                array_col = array_col // 2
            else:
                array_col = (array_col - 1) // 2
            imagerow = int(imagerow)
            imagecol = int(imagecol)
            
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up)
            )
            if padding:
                array_row = array_row + (tile_rownum-78)//2
                array_col = array_col + (tile_colnum-64)//2
            if resize:
                tile = tile.resize((target_size, target_size), Image.Resampling.LANCZOS)
                tile_name = str(array_row) + "-" + str(array_col) + "-" + str(crop_size)+ "-" + str(target_size)
            else:
                tile_name = str(array_row) + "-" + str(array_col) + "-" + str(crop_size)
            
            out_tile = Path(out_path) / (tile_name + ".jpeg")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {}, {})".format(
                        str(out_path), str(imagerow), str(imagecol)
                    )
                )
            tile.save(out_tile, "JPEG")

            pbar.update(1)
            
        if padding:
            for row in range(tile_rownum):
                for col in range(tile_colnum):
                    filename = '{}-{}-{}.jpeg'.format(str(row),str(col),str(crop_size))
                    output_path = os.path.join(out_path, filename)
                    # check if already exit
                    if not os.path.exists(output_path):
                        # create blank jpeg
                        image = Image.new('RGB', (crop_size,crop_size), (255,255,255))
                        image.save(output_path)
                    
                        pbar.update(1)



def tile_hires(
    sample_path: Union[Path, str],
    out_path: Union[Path, str] = "./tiling",
    sample_id: str = 'sample',
    position_path = Union[Path, str],
    scalefactor_path = Union[Path, str],
    crop_size: int = 30,
    target_size: int = 224,
    resize: bool = False,
    padding = True,
    tile_rownum = 80,
    tile_colnum = 64,
    verbose: bool = False
):


    # Check the exist of out_path
    if sample_id is not None:
        out_path = str(out_path) + '/' + sample_id
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path, exist_ok = True)

    image_pillow = Image.open(sample_path)
    image = np.array(image_pillow)
    
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)

    if img_pillow.mode == "RGBA":
        img_pillow = img_pillow.convert("RGB")

    tile_names = []

    position = pd.read_csv(position_path,
            header = 0, names = ['barcode','in_tissue','array_row','array_col','imagerow','imagecol'])
    
    #scale_factor
    scale_factor_data = pd.read_csv(scalefactor_path,header = 0)

    hires_scale_factor = scale_factor_data.iloc[0,2]
    
    
    if padding:
        total = tile_rownum * tile_colnum
    else:
        total = position.shape[0]
    
    with tqdm(
        total=total,
        desc="Tiling image",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for array_row,array_col,imagerow,imagecol in zip(position['array_row'],position['array_col'],position["imagerow"], position["imagecol"]):
            array_row = int(array_row)
            if array_row % 2 == 0:
                array_col = array_col // 2
            else:
                array_col = (array_col - 1) // 2
            imagerow = int(int(imagerow) * hires_scale_factor)
            imagecol = int(int(imagecol) * hires_scale_factor)
            
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up)
            )
            if padding:
                array_row = array_row + (tile_rownum-78)//2
                array_col = array_col + (tile_colnum-64)//2
            if resize:
                tile = tile.resize((target_size, target_size), Image.Resampling.LANCZOS)
                tile_name = str(array_row) + "-" + str(array_col) + "-" + str(crop_size)+ "-" + str(target_size)
            else:
                tile_name = str(array_row) + "-" + str(array_col) + "-" + str(crop_size)
            
            out_tile = Path(out_path) / (tile_name + ".jpeg")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {}, {})".format(
                        str(out_path), str(imagerow), str(imagecol)
                    )
                )
            tile.save(out_tile, "JPEG")

            pbar.update(1)
            
        if padding:
            for row in range(tile_rownum):
                for col in range(tile_colnum):
                    filename = '{}-{}-{}.jpeg'.format(str(row),str(col),str(crop_size))
                    output_path = os.path.join(out_path, filename)
                    # check if already exit
                    if not os.path.exists(output_path):
                        # create blank jpeg
                        image = Image.new('RGB', (crop_size,crop_size), (255,255,255))
                        image.save(output_path)
                    
                        pbar.update(1)
                        


def _open_images(paths):
    for path in paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (20, 20))
        if img is not None:
            yield img

def reconstruct(tile_path):
    """reconstruct the tiles to full image

    Args:
        tile_path (str): path to the tiles of single sample    eg:'../data/TCGA-COAD/cleaned_tile/TCGA-3L-AA1B-01/'
    """
    patch_list = os.listdir(tile_path)
    patch_list = [file for file in patch_list if file.lower().endswith(('jpeg', 'jpg'))]
    sorted_patch_list = sorted(patch_list, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    patch_sorted_full_path = [os.path.join(tile_path,file_name) for file_name in sorted_patch_list]
    img_list = list(_open_images(patch_sorted_full_path))
    h_patchs = [cv2.hconcat(img_list[i:i+64]) for i in range(0, len(img_list), 64)]
    full_img = cv2.vconcat(h_patchs)
    resized_img = cv2.resize(full_img, (1280, 1600))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    resized_img = resized_img.astype(np.uint8)
    plt.imshow(resized_img)
    plt.axis('off')
    plt.show()