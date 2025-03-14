# HiST: Histological Image Reconstruct Tumor Spatial Transcriptomics via MultiScale Fusion Deep Learning

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [0.Download data](#0-download-data)
  - [1.Preprocess module](#1-preprocess-module)
  - [2.Prediction module](#2-prediction-module)
  - [3.Application Module](#3-application-module)
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
**Method1 :Use requirement file(Not recommended):**
```bash
conda create -n HiST python=3.8.18 mamba
conda activate HiST
mamba install --yes -n HiST -c conda-forge --file requirements.txt
pip install ./resource/timm-0.5.4.tar
```
Use `nvcc -V`to check cuda version on your device
**Method2 :Follow the instructions:**
```bash
conda create -n HiST python=3.8.18 mamba
conda activate HiST
#Obtain the corresponding CUDA version of torch on your device:https://pytorch.org/get-started/locally/
#Or Install by mamba(Recommended):
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
#Author dependent configuration (used to reproduce):
#pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
#Other dependencies
mamba install -c conda-forge python-spams=2.6.1
pip install numpy==1.22 imgaug albumentations pandas matplotlib scikit-learn opencv-python staintools lifelines torchsurv openpyxl palettable leidenalg ipykernel tqdm scanpy
#Install modified timm for CTranspath(Feature extraction model)
pip install ./resource/timm-0.5.4.tar
```

**install seurat in R(conda env HiST)**

```bash
mamba install rpy2 r-tidyverse r-Seurat -c r
```
```R
# Run in R console
packages <- c("Seurat", "tidyverse")
install.packages("sf", repos = "https://cran.r-project.org")
```

**Used for gene selection method (Optional):[R package sf installation instructions](https://r-spatial.github.io/sf/)**
```bash
sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
sudo apt-get update
sudo apt-get install libudunits2-dev libgdal-dev libgeos-dev libproj-dev libsqlite0-dev
mamba install r-sf r-spdep -c r
```

**Other dependencies(Optional; if WSIs are used for training or prediction)**
```bash
sudo apt update && apt install -y openslide-tools
pip install openslide-python
```
---

## Usage

We use two sample from CRC dataset of 10x Visium technology as an example.

### 0. Download data

##### (A)Pre-trained model weights for feature extraction can be downloaded [here](https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view?usp=sharing), and please put it in `/your_working_directory/HiST/resource/`.

##### (B)Two test sample data of CRC can be downloaded [here](https://drive.google.com/file/d/1-87C3EQf4UK-EsNiWlMGFWUDvxNX-Sb_/view?usp=sharing). Please unzip data.zip and put the contents in `/your_working_directory/HiST/data/`
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
- Step0: Download slide images and metadata from [NGDC](https://ngdc.cncb.ac.cn/).
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
(Unpublished now)
```
@article{HiST,
    title={HiST: Histological Image Reconstruct Tumor Spatial Transcriptomics via MultiScale Fusion Deep Learning},
    author={Wei Li1#, Dong Zhang#, Eryu Peng, Shijun Shen, Yao Liu*, Junke Zheng*, Cizhong Jiang*, Youqiong Ye*},
    journal={XX},
    year={2025},
    doi={xx}
}
```