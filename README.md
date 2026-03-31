# HiST: Histological Image Reconstruct Tumor Spatial Transcriptomics via MultiScale Fusion Deep Learning

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Method 1: Docker](#method1-docker-image)
  - [Method 2: Conda environment.yml](#method2-use-environment-file-to-setup)
  - [Method 3: Manual installation](#method3-follow-the-instructions-to-config-environment)
- [Quick Start: Run Demo](#run-demo)
  - [1. Download demo data](#1-download-demo-data)
  - [2. Run demo](#2-run-demo)
  - [3. Check output results](#3-check-the-results)
- [HiST Tutorial](#hist-tutorial)
  - [0. Prepare your own data](#0-prepare-data)
  - [1. Preprocess module](#1-preprocess-module)
  - [2. Prediction module](#2-prediction-module)
  - [3. Application module](#3-application-module)
    - [A. Survival model](#a-survival-model)
    - [B. Immunotherapy response model](#b-immunotherapy-response-model)
- [Credits and Acknowledgments](#credits-and-acknowledgments)
- [License](#license)
- [Citation](#citation)

---

## Introduction
<img src="./img/HiST%20architecture.jpg" width = "570" height = "546" alt="HiST Architecture" align=center />

Spatial transcriptomics (ST) offers valuable insights into the tumor microenvironment by integrating molecular features with spatial context, but its clinical diagnostic application is limited due to its high cost. 

To address this, we develop multi-scale convolutional deep learning framework, HiST, which utilizes ST to learn the relationship between spatially resolved gene expression profiles (GEPs) and histological morphology. HiST accurately predicts tumor regions (e.g., breast cancer, area under curve: 0.96), which are highly concordant with pathologist annotations. Then HiST reconstructs spatially resolved GEPs with an average Pearson correlation coefficient of 0.74 across five cancer types, which is >3 folds greater than that of the best previously reported tool. HiST's application module performs well in predicting cancer patient prognosis for five cancer types from the Cancer Genome Atlas (e.g., a concordance index 0.78 in breast cancer) and immunotherapy outcomes. Moreover, spatial GEPs aid to unveil regulatory networks and key regulators to immunotherapy. 

In summary, HiST’s robust performance in tumor identification and reconstruction of spatial GEPs and its applications in prognosis prediction and immunotherapy response offer great potential for advancing tumor profiling and improving personalized cancer treatment.

---

## Installation
- We recommend run HiST on **Linux**

To get started, clone the repository and install the required dependencies:
```bash
git clone https://github.com/Yelab2020/HiST.git
cd HiST
```
### Method1: Docker image

You can either pull the pre-built Docker image from [Docker Hub](https://hub.docker.com/r/bejsernia/hist/tags):
```bash
docker pull bejsernia/hist:251021
```
or build the image locally using the provided Dockerfile:
```bash
docker build -t hist .
```

### Method2: Use environment file to setup
```bash
conda env create -f environment.yml -n HiST
#Install modified timm for CTranspath(Feature extraction model)
pip install ./resource/timm-0.5.4.tar
```

### Method3: Follow the instructions to config environment
```bash
#Create env
conda create -n HiST python=3.8.18
conda activate HiST
pip install numpy==1.22 pandas matplotlib scikit-learn imgaug albumentations scanpy
#Use Conda to manage the non-Python dependencies.
conda install python-spams=2.6.1 openslide-python opencv-python rpy2 -c conda-forge
pip install staintools lifelines openpyxl palettable leidenalg ipykernel
#Install torch from the specific source
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
#Install modified timm for CTranspath(Feature extraction model)
pip install ./resource/timm-0.5.4.tar
#ipykernel for jupyter notebook
python -m ipykernel install --user --name HiST --display-name HiST
```

**Used for gene selection method (Optional):[R package sf installation instructions](https://r-spatial.github.io/sf/)**
```bash
sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
sudo apt-get update
sudo apt-get install libudunits2-dev libgdal-dev libgeos-dev libproj-dev libsqlite0-dev
conda install r-Seurat r-tidyverse r-sf r-spdep -c r
```

---

## Run demo

To support reproducibility and rapid functional validation, we provide both executable scripts and a step-by-step notebook for two types of experiments: (i) a lightweight smoke test using a demo CRC dataset (three samples), including training, validation, and visualization steps, which can be completed within approximately ten minutes on a single GPU, and (ii) reproduction of the HiST CRC prediction results using our released pretrained HiST model weights.

### 1. Download demo data

#### (A)Pre-trained model weights for feature extraction can be downloaded [here](https://drive.google.com/file/d/1g36qBWfnqoot1JpgTisBjPg4GrUOU4wT/view?usp=sharing), and please put it in `/your_working_directory/HiST/resource/`.

#### (B)Demo data and HiST model weights for reproduction of CRC can be downloaded [here](https://drive.google.com/file/d/1z_PJMdffWsyalKtY01G3lvUDIy4C-cF2/view?usp=sharing). 
Please unzip data.zip and put the contents in `/your_working_directory/HiST/demo/data/`

```
./demo/data/
├── HE
│   ├── CRC1.jpg
│   ├── CRC2.jpg
│   └── M1042T.jpg
├── geneMatrix
│   ├── CRC1.rds.gz
│   ├── CRC2.rds.gz
│   └── M1042T.rds.gz
├── gene_list
│   └── CRC_SVG346_list.txt
├── mask_png
│   └── 0
│       ├── CRC1.png
│       ├── CRC2.png
│       └── M1042T.png
├── model_weights
│   └── M1042T_200_model.pth
├── scale_factor
│   ├── CRC1.csv
│   ├── CRC2.csv
│   └── M1042T.csv
└── tissue_positions_list
    ├── CRC1.csv
    ├── CRC2.csv
    └── M1042T.csv
```

### 2. Run demo
(Executable scripts or step-by-step notebook)

#### (A) Executable scripts (docker)
```bash
python ./demo/script/1.preprocess.py
python ./demo/script/2.LOO.py
python ./demo/script/3.reproduce.py
```

#### (B) Notebook (conda env)
*Please refer to the [notebook](./demo/notebook/LOOtest&Reproduction.ipynb).*

### 3. Check the results
Data folder structure:
- features: Extracted H&E features for each sample.
- model: Leave-one-out cross-validation results.

  (i) gene: Spatial gene expression prediction model.

  - cor_df: Pearson correlation results for validation genes in each fold.
  - visualization：Spatial visualization of predicted gene expression.

  (ii) tumor: Tumor spot prediction model.

  - predict_masks: Intermediate prediction results (binary tumor masks).
  - predict_results: Spatial visualization of tumor region predictions.

- reproduction_Figure3b: Visualization results comparing HiST predictions with ground-truth expression profiles using pretrained model weights and raw data.

```
output/
├── features
├── model
│   ├── gene
│   │   └── checkpoint_loo
│   │       └── xx
│   │           ├── best_model
│   │           ├── cor_df
│   │           ├── correlations
│   │           ├── last_model
│   │           ├── log
│   │           └── visualization
│   └── tumor
│       └── checkpoint_loo
│           └── xx
│               ├── best_model
│               ├── last_model
│               ├── log
│               ├── predict_masks
│               └── predict_results
├── reproduction_Figure3b
├── tile
│   ├── CRC1
│   ├── CRC2
│   └── M1042T
```

## HiST tutorial
### 0. Prepare data

Put your own data in `/your_working_directory/HiST/data/`

Data folder structure:
- HE: Full resolution HE images.
- hires_HE: High resolution HE images provided by spaceranger.
- seurat_obj: ST sample Seurat objects.
```
./data
├── HE
│   ├── CRC1.jpg
│   └── CRC2.jpg
├── hires_HE
│   ├── CRC1_tissue_hires_image.png
│   └── CRC2_tissue_hires_image.png
├── seurat_obj
│   ├── CRC1.rds.gz
│   └── CRC2.rds.gz
``` 


### 1. Preprocess module
For preprocess module, we obtained the histological information and spatial context of the original whole slice imaging (WSI), avoiding the high GPU memory requirements of high-resolution WSI.
- Step1(Optional): Gene selection `./R/1.gene_select.R`.
Sample file: `./resource/CRC_SVG346_list.txt`
```bash
Rscript ./R/1.gene_select.R
```
- Step2: Create gene matrix and mask matrix `./R/2.get_matrix.R`
```bash
Rscript ./R/2.get_matrix.R
```
- Step3: Prepare mask and patch & feature extraction.
Run in python, referring to the [vignette](./vignettes/1.preprocess_module.ipynb).

### 2. Prediction module
We used an improved U-Net framework on prediction module with two prediction tasks, including **tumor spots identification and tumor spatial transcriptomics prediction.**
- *Please refer to the [vignette](./vignettes/2.prediction_module.ipynb) for specific steps.*

### 3. Application module
We utilized the ST profiles obtained from prediction module as the molecular features of HE histology images and trained the model for **disease prognosis and immunotherapy response prediction.**

##### A. Survival model
- Step0: Download slide images from [TCGA](https://portal.gdc.cancer.gov/).
- Step1: Prepare WSI patches.

(i) Cut WSIs into patches
Output: HE(resized smaller TCGA HE images) and tiles.
Usage: 
```bash
nohup python ./util/TCGA_HE_preprocess.py --data_path './data/TCGA/Biospecimen/Slide_Image' \
--output_path './output/TCGA/' \
--cores 8 > ./HE_preprocess.log 2>&1 &
```
(ii) Clean up tiles (Optional): [source:wsi-tile-cleanup](https://github.com/lucasrla/wsi-tile-cleanup)
Output: Tiles only containing tissue sections.
Installation:
```bash
conda create -n wsi_cleanup --channel conda-forge python=3.6 libvips pyvips numpy
conda activate wsi_cleanup
python3.6 -m pip install git+https://github.com/lucasrla/wsi-tile-cleanup.git
pip install pillow ipykernel tqdm pandas
```
Usage:
```bash
nohup python tile_cleanup.py --source_root_path './output/TCGA/tiles' \
--output_path '../output/TCGA/clean_tiles_75/' \
--cutoff 0.75 --cores 16 > ./TCGA_tile_cleanup.log 2>&1 &
```
*Please refer to the [vignette](./vignettes/3.1application_module_survival.ipynb) for the following steps.*
- Step3: Feature extraction.
- Step4: Spatial gene profiles prediction by HiST gene prediction module.
- Step5: Training survival model.

##### B. Immunotherapy response model
*Please refer to the [vignette](./vignettes/3.2application_module_ICB.ipynb) for the following steps.*
- Step0: Download slide images from [NGDC](https://ngdc.cncb.ac.cn/).
- Step1: Prepare WSI patches.
- Step3: Feature extraction.
- Step4: Spatial gene profiles prediction by HiST gene prediction module.
- Step5: Training classfication model.
---

## Credits and Acknowledgments

Ground truth of tumor segmentation was inferred by [Cottrazm](https://github.com/Yelab2020/Cottrazm)

Pretrained model weights are from [CTransPath](https://github.com/Xiyue-Wang/TransPath)

Tiles clean up method using [wsi-tile-cleanup](https://github.com/lucasrla/wsi-tile-cleanup)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation
```
@article{HiST,
    title={HiST: Histological Image Reconstruct Tumor Spatial Transcriptomics via MultiScale Fusion Deep Learning},
    author={Wei Li#, Dong Zhang#, Eryu Peng, Shijun Shen, Hamid Alinejad-Rokny, Yao Liu*, Junke Zheng*, Cizhong Jiang*, Youqiong Ye*},
    journal={Advanced Science},
    year={2025},
    doi={10.1002/advs.202514351}
}
```

---

## Star History

[![Star History Chart](https://api.star-history.com/image?repos=Yelab2020/HiST&type=date&legend=top-left)](https://www.star-history.com/?repos=Yelab2020%2FHiST&type=date&legend=bottom-right)