import os
import shutil
import argparse
from tqdm import tqdm
import multiprocessing
import PIL.Image as Image
import wsi_tile_cleanup as cleanup


def bg2blanca(
    patient_id, source_root_path, output_path, tile_size, threshold, cutoff
):
    try:
        in_path = os.path.join(source_root_path, patient_id)
        out_path = os.path.join(output_path, patient_id)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        progress_bar = tqdm(total=len(os.listdir(in_path)), desc=f'{patient_id} processing:', position=0, leave=True)
        for filename in os.listdir(in_path):
            if filename.lower().endswith(('jpeg', 'jpg')):
                try:
                    tile_path = os.path.join(in_path, filename)
                    vi_tile = cleanup.utils.read_image(tile_path)
                    bands = cleanup.utils.split_rgb(vi_tile)
                    colors = ["red", "green", "blue"]

                    perc_list = []

                    for color in colors:
                        perc = cleanup.filters.pen_percent(bands, color)
#                         print(f"{color}: {perc*100:.3f}%")
                        perc_list.append(perc)

                    perc = cleanup.filters.blackish_percent(bands)
#                     print(f"blackish: {perc*100:.3f}%")
                    perc_list.append(perc)

                    otsu_threshold = cleanup.filters.otsu_threshold(vi_tile)
                    perc = cleanup.filters.background_percent(vi_tile, otsu_threshold-threshold)
#                     print(f"background: {perc*100:.3f}%")
                    perc_list.append(perc)

                    if max(perc_list) > cutoff:#replace the bg tile with a white tile
                        blanca_tile = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))
                        blanca_tile.save(os.path.join(out_path, filename))
                    else:#copy the original tile
                        shutil.copy(tile_path, os.path.join(out_path, filename))
                        
                    perc_str = ', '.join(f"{color}: {perc*100:.3f}%" for perc, color in zip(perc_list, ["red", "green", "blue", "blackish", "background"]))
                    progress_bar.set_description(f'{patient_id} - {filename}: {perc_str}')
                    progress_bar.update(1)

                except Exception as e:
                    print(f"Error occurred in processing patch: {tile_path}, error: {e}")
                    
        progress_bar.close()

    except Exception as e:
        print(f"Error occurred in processing patient ID: {patient_id}, error: {e}")


def main(
    patient_list,
    source_root_path,
    output_path,
    tile_size,
    threshold,
    cutoff,
    cores
):
    with multiprocessing.Pool(cores) as p:
        p.starmap(bg2blanca, [(patient_id, source_root_path, output_path, tile_size, threshold, cutoff) for patient_id in patient_list])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSI tile cleanup')
    parser.add_argument('--source_root_path', type=str, default='./data/TCGA-COAD/tile', help='source root path')
    parser.add_argument('--output_path', type=str, default='./data/TCGA-COAD/cleaned_tile', help='output path')
    parser.add_argument('--tile_size', type=int, default=224, help='tile size')
    parser.add_argument('--threshold', type=int, default=40, help='otsu_threshold minus this threshold')
    parser.add_argument('--cutoff', type=float, default=0.75, help='cutoff')
    parser.add_argument('--cores', type=int, default=16, help='number of cores')
    args = parser.parse_args()

    patient_list = os.listdir(args.source_root_path)
    
    main(
        patient_list,
        args.source_root_path,
        args.output_path,
        args.tile_size,
        args.threshold,
        args.cutoff,
        args.cores
    )

