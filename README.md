# HiST: Histological Image Reconstruct Tumor Spatial Transcriptomics via MultiScale Fusion Deep Learning

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Reference](#reference)
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
**Use requirement file(Not recommended):**
```bash
conda create -n HiST python=3.8.18 mamba
conda activate HiST
mamba install --yes -n HiST -c conda-forge --file requirements.txt
```
Use `nvcc -V`to check cuda version on your device
**Follow the instructions:**
```bash
conda create -n HiST python=3.8.18 mamba
conda activate HiST
#Obtain the corresponding CUDA version of torch on your device:https://pytorch.org/get-started/locally/
#Or Install by mamba(Recommended):
mamba search pytorch-cuda -c pytorch -c nvidia
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
#Author dependent configuration (used to reproduce):
#pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
#Other dependencies
mamba install -c conda-forge python-spams=2.6.1
pip install numpy==1.22 imgaug albumentations pandas matplotlib scikit-learn opencv-python staintools lifelines torchsurv openpyxl palettable leidenalg ipykernel tqdm
#Install modified timm for CTranspath(Feature extraction model)
pip install ./resource/timm-0.5.4.tar
```
~~If it occurs error when importing torch~~
~~`ImportError: /home/usr/miniconda3/envs/HiST/lib/python3.8/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12` run(Can't work in Jupyter):~~
~~#replace the usr to your user name~~
~~`ln -s /home/usr/miniconda3/envs/HiST/lib/python3.8/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12 /home/usr/miniconda3/envs/HiST/lib/python3.8/site-packages/nvidia/cusparse/lib/libnvJitLink.so.12`~~
~~`export LD_LIBRARY_PATH=/home/usr/miniconda3/envs/HiST/lib/python3.8/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH`~~
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

**Other dependencies(Optional; if WSIs are used for prediction)**
```bash
sudo apt update && apt install -y openslide-tools
pip install openslide-python
```
---

## Usage

We use two sample from CRC dataset of 10x Visium technology as an example.

### 0. Download Data

##### (A)Pre-trained model weights for feature extraction can be downloaded [here](https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view), and please put it under `/your_working_directory/HiST/resource/`.

##### (B)Pre-trained model weights for CRC tumor identification and spatial gene profiles prediction can be downloaded [here]().

##### (C)Two test sample data of CRC can be downloaded [here]().

### 1. Preprocess module
For preprocess module, we obtained the histological information and spatial context of the original whole slice imaging (WSI), avoiding the high GPU memory requirements of high-resolution WSI.
- Step1: Gene selection `./R/1.gene_select.R`(Optional)
Sample file: `./resource/CRC_SVG346_list.txt`
```bash
Rscript ./R/1.gene_select.R
```
- Step2: Create gene matrix and mask matrix `./R/2.get_matrix.R`
```bash
Rscript ./R/2.get_matrix.R
```
- Step3: Prepare mask and patch & Feature extraction
Run in python, referring to the [vignette](./vignettes/1.preprocess.ipynb).

### 2. Prediction module
We used an improved U-Net framework on prediction module with two prediction tasks, including **tumor spots identification and tumor spatial transcriptomics prediction.**
- Please refer to the [vignette]() for specific steps.

### 3. Application module
We utilized the ST profiles obtained from prediction module as the molecular features of HE histology images and trained the model for **disease prognosis and immune response prediction.**
- Please refer to the [vignette]() for specific steps.

---

## Reference

Ground truth of tumor segmentation was inferred by [Cottrazm](https://github.com/Yelab2020/Cottrazm)
Pretrained model weights are from [CTransPath](https://github.com/Xiyue-Wang/TransPath)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation
(Unpublished now)
```
@article{HiST,
    title={HiST: Histological Image Reconstruct Tumor Spatial Transcriptomics via MultiScale Fusion Deep Learning},
    author={Wei Li#, Dong Zhang#, Yao Liu, Junke Zheng*, Cizhong Jiang*, Youqiong Ye*},
    journal={XX},
    year={2025},
    doi={xx}
}
```