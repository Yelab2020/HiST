import os
import cv2
import argparse
import openslide
import numpy as np
from PIL import Image
import multiprocessing
from patch import tile
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = 1000000000000

def rotate_and_pad(img : Image.Image):
    # rotate
    width, height = img.size
    if width > height:
        img = img.rotate(90, expand = True)
        width, height = img.size
    # pad
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



def crop_tissue(
    slide_path : str,
    verbose = False
):
    # load & preprocess
    slide = openslide.open_slide(slide_path)
    
    level = len(slide.level_dimensions)-1
    dimension = slide.level_dimensions[level]
    
    smaller_region = slide.read_region((0, 0), level, dimension)
    smaller_region_RGB = smaller_region.convert('RGB')
    smaller_region_array = np.array(smaller_region_RGB)
    region_gray = cv2.cvtColor(smaller_region_array, cv2.COLOR_BGR2GRAY)

    # calculate ratio
    ratio = slide.level_dimensions[0][0] // slide.level_dimensions[level][0]

    # get x,y gradient using Sobel
    gradX = cv2.Sobel(region_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(region_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)


    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.intp(cv2.boxPoints(rect))

    if verbose:
        for i in range(len(box)):
            if i < len(box) - 1:
                plt.plot(box[i:i+2, 0], box[i:i+2, 1], 'r-')
            else:
                plt.plot([box[i, 0], box[0, 0]], [box[i, 1], box[0, 1]], 'r-')

    # vertical crop
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    
    Img2show = smaller_region_array[y1:y2, x1:x2]

    if verbose:
        plt.imshow(Img2show)

    x1 = x1 * ratio
    x2 = x2 * ratio
    y1 = y1 * ratio
    y2 = y2 * ratio
    
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    x2 = slide.level_dimensions[0][0] if x2 > slide.level_dimensions[0][0] else x2
    y2 = slide.level_dimensions[0][1] if y2 > slide.level_dimensions[0][1] else y2
    
    #  crop and save slide on level 0
    Img2save = slide.read_region((x1, y1), 0, (x2-x1, y2-y1))
    Img2save = Img2save.convert('RGB')
    Img2save,_,_ = rotate_and_pad(Img2save)

    return Img2save


def get_patches(
    slide_path : str,
    save_dir : str = './TCGA/',
    verbose = False
):
    img = crop_tissue(slide_path, verbose)
    
    HE2save = img.resize((2560,3200))
    HE_dir = os.path.join(save_dir, 'HE')
    if not os.path.exists(HE_dir):
        os.makedirs(HE_dir, exist_ok = True)
    sample_id = os.path.basename(slide_path).split('.')[0][0:15]
    HE2save.save(os.path.join(HE_dir, sample_id+'.jpg'), 'JPEG')
    
    tile_dir = os.path.join(save_dir, 'tiles')
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir, exist_ok = True)
    tile(sample_id = sample_id,
         HE_path='',
         out_path=tile_dir,
         img = img)


def main(
    slide_dir : str,
    save_dir : str = './TCGA/',
    verbose = False,
    suffix = '.svs',
    cores = 8
):
    he_files = []
    for root, dirs, files in os.walk(slide_dir):
        for file in files:
            if file.endswith(suffix):
                full_path = os.path.join(root, file)
                he_files.append(full_path)
    
    with multiprocessing.Pool(cores) as p:
        p.starmap(get_patches, [(slide, save_dir, verbose) for slide in he_files])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSI tile cleanup')
    parser.add_argument('--data_path', type=str, default='./data/TCGA-COAD/GDCdata/TCGA-COAD/Biospecimen/Slide_Image', help='data path')
    parser.add_argument('--output_path', type=str, default='./TCGA/', help='output path')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose')
    parser.add_argument('--suffix', type=str, default='.svs', help='suffix of slide files')
    parser.add_argument('--cores', type=int, default=16, help='number of cores')
    args = parser.parse_args()

    main(
        args.data_path,
        args.output_path,
        args.verbose,
        args.suffix,
        args.cores)