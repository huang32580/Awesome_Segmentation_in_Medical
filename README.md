---

# **Awesome Segmentation in Medical Imaging: A Breast Ultrasound Benchmark 🚀**

A robust and reproducible framework for benchmarking deep learning models on Breast Ultrasound (BUS) image segmentation, designed to ensure fair evaluation and prevent test set leakage.  

The core philosophy of this project is to establish a fair and unbiased evaluation pipeline, addressing common pitfalls in machine learning research. By strictly separating cross-validation from final testing, this framework produces publication-ready results that accurately reflect a model's true generalization performance.  
  

## **Table of Contents**

- [Our Vision](#our-vision-)
- [Key Features](#key-features-)
- [Supported Models & Datasets](#supported-models--datasets-)
- [Project Structure](#project-structure-)
- [Installation](#installation-)
- [Running Experiments](#running-experiments-)
- [Understanding the Results](#understanding-the-results-)
- [How to Extend](#how-to-extend-️)
- [Acknowledgements](#acknowledgements-)
- [License & Citation](#license--citation-)

---

## **Our Vision 💡**

While research in medical image segmentation is advancing rapidly, fair model comparison remains a significant challenge due to:

1. **Data Leakage:** The test set is often inadvertently used during cross-validation, leading to inflated and unreliable performance metrics.  
  
2. **Inconsistent Evaluation:** Different studies use different data splits, preprocessing steps, and evaluation metrics, making direct comparisons impossible.  
  
3. **Limited Scope:** Most benchmarks focus only on binary (benign vs. malignant) classification, failing to incorporate the 'normal' class, which is crucial for real-world clinical applications.
  
  
Awesome-BUS-Benchmark is engineered to solve these problems. We provide a standardized benchmark built on the principles of **strict data separation** and **stratified sampling** to ensure that all models are evaluated under the exact same conditions, leading to truly comparable and reproducible results.

---

## **Key Features ✨**

* **🥇 Strict Data Splitting:** A dedicated, held-out test set is created **once** before any training or cross-validation begins. This test set is never seen during model development or selection, preventing any form of data snooping and ensuring a truly unbiased final evaluation of generalization performance.  
  
* **⚖️ Stratified K-Fold Cross-Validation:** Implements Stratified K-Fold to handle the inherent class imbalance in BUS datasets (e.g., the small number of normal cases). This ensures that each fold's class distribution is representative of the overall dataset, leading to more stable training and reliable validation metrics.  
  
* **📚 Comprehensive Dataset Support:** Natively supports multiple public Breast US datasets and crucially **includes the normal class**, offering a more complete and realistic benchmark than typical binary (benign vs. malignant) studies.  
  
* **🧩 Modular & Extensible Architecture:** The code is structured to be highly modular. You can easily add new datasets, models (both CNN & Transformer-based), loss functions, and metrics with minimal code changes.  
  
* **⚙️ Automated & Configurable Pipelines:** Comes with powerful shell scripts (run\_cnn.sh, run\_vit.sh) that automate the entire workflow: k-fold training, testing, and results aggregation. All experiment parameters (model choice, learning rate, epochs, etc.) are controlled via central YAML configuration files, allowing for rapid and reproducible experiments.
  
  
---
  

## **Supported Models & Datasets 📖**

### **Model Zoo**
|     Model     |                        Original code                         |                          Reference                           |
| :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      **U-Net**      | [Caffe](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net) |      [MICCAI'15](https://arxiv.org/pdf/1505.04597.pdf)       |
| **Attention U-Net** | [Pytorch](https://github.com/ozan-oktay/Attention-Gated-Networks) |       [MIDL'18](https://arxiv.org/pdf/1804.03999.pdf)       |
|     **U-Net++**     |    [Pytorch](https://github.com/MrGiovanni/UNetPlusPlus)     | [MICCAI'18](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7329239/pdf/nihms-1600717.pdf) |
|    **U-Net 3+**     |    [Pytorch](https://github.com/ZJUGiveLab/UNet-Version)     |        [ICASSP'20](https://arxiv.org/pdf/2004.08790)         |
|    **TransUnet**    |      [Pytorch](https://github.com/Beckschen/TransUNet)       |       [Arxiv'21](https://arxiv.org/pdf/2102.04306.pdf)       |
|      **MedT**       | [Pytorch](https://github.com/jeya-maria-jose/Medical-Transformer) |      [MICCAI'21](https://arxiv.org/pdf/2102.10662.pdf)       |
|      **UNeXt**     | [Pytorch](https://github.com/jeya-maria-jose/UNeXt-pytorch)  |      [MICCAI'22](https://arxiv.org/pdf/2203.04967.pdf)       |
|    **SwinUnet**     |    [Pytorch](https://github.com/HuCaoFighting/Swin-Unet)     |       [ECCV'22](https://arxiv.org/pdf/2105.05537.pdf)        |
|     **CMU-Net**     |       [Pytorch](https://github.com/FengheTan9/CMU-Net)       |       [ISBI'23](https://arxiv.org/pdf/2210.13012.pdf)        |
|     **CMUNeXt**     |       [Pytorch](https://github.com/FengheTan9/CMUNeXt)       |       [ISBI'24](https://arxiv.org/pdf/2308.01239.pdf)       |
|     **U-KAN**     |       [Pytorch](https://github.com/CUHK-AIM-Group/U-KAN)       |       [AAAI'25](https://arxiv.org/pdf/2406.02918.pdf)       |

### **Datasets**

| Dataset | Official Source | Download link|
| :---- | :---- | :---- |
| **BUSI** | [Dataset of Breast Ultrasound Images \- Al-Dhabyani et al.](https://www.sciencedirect.com/science/article/pii/S2352340919312181?via%3Dihub) | [Download Data]([https://scholar.cu.edu.eg/?q=afahmy/pages/dataset](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset?resource=download)) |
| **BUSBRA** | [Breast Ultrasound Bi-Rads Classification... \- Ribeiro et al.](https://pubmed.ncbi.nlm.nih.gov/37937827/) | [Download Data](https://zenodo.org/records/8231412) |
| **BUS-UC** | [Breast Ultrasound Cancer Image Classification \- Garodia et al.](https://www.sciencedirect.com/science/article/pii/S0952197623014768?via%3Dihub) | [Download Data](https://data.mendeley.com/datasets/3ksd7w7jkx/1) |
| **BUS-UCLM** | [BUS-UCLM: Breast ultrasound lesion segmentation dataset. \- Noelia Vallez et al.,2025](https://www.nature.com/articles/s41597-025-04562-3) | [Download Data](https://github.com/noeliavallez/BUS-UCLM-Dataset) |
| **Yap2018** | [Breast ultrasound lesions recognition: a preliminary study... \- Yap et al.](https://pubmed.ncbi.nlm.nih.gov/28796627/) | [Download Data](https://www2.docm.mmu.ac.uk/STAFF/m.yap/files/BUS_ReleaseAgreement.pdf) |

  
---
  
## **Project Structure 📂**

The repository is organized logically to separate concerns and facilitate ease of use and extension.
```
Awesome_Segmentation_in_Medical/  
│
├── data/  
│   ├── preprocessing/  
│   │   ├── augmentation.py        # Data augmentation logic (rotations, flips, etc.)  
│   │   └── preprocess.py          # Data preprocessing logic (resizing, normalization, etc.)  
│   ├── prepare_datasets.py        # Script to standardize raw datasets and create CSVs with fold splits  
│   └── synthetic_datasets.py      # Script for synthetic data generation (optional)  
│  
├── data_loader/  
│   └── data_loaders.py            # Defines PyTorch DataLoaders for training/validation/testing  
│  
├── datasets/                      # (User-supplied) Directory to store raw downloaded datasets  
│  
├── src/  
│   ├── models/                    # PyTorch model architecture definitions  
│   │   ├── cnn_based/             # --- CNN-based models like UNet, AttUNet, UNet++, UNeXt, CMUNet, U-KAN  
│   │   └── ViT_based/             # --- Transformer-based models like TransUnet, MedT, Swin-Unet  
│   │  
│   ├── trainer/  
│   │   └── trainer.py             # Core training and validation loop logic (epochs, backprop, etc.)  
│   │  
│   └── utils/                     # Core utilities and helper functions  
│       ├── LovaszSoftmax/pytorch/ # --- The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks     
│       │   └── lovasz_losses.py   # --- Standalone PyTorch implementation of the Lovász hinge and Lovász-Softmax for the Jaccard index
│       ├── losses.py              # --- Loss functions for segmentation (DiceLoss, BCELoss, etc.)  
│       ├── metrics.py             # --- Evaluation metrics (Dice, IoU, HD95, etc.)  
│       ├── parse_config.py        # --- Functionality to read and parse the config.json file  
│       └── util.py                # --- Other useful functions, such as logging  
│  
├── results/                       # Stores all experiment outputs, including CSVs with metrics per fold  
│  
├── .gitignore                     # List of files to be ignored by Git  
├── config.json                    # Central configuration file to control all experiments (models, hyperparameters, etc.)  
├── environment.yml                # File for Conda environment setup  
│  
├── run_cnn.sh                     # Entrypoint script to run CNN-based model experiments  
├── run_vit.sh                     # Entrypoint script to run Transformer-based model experiments  
├── run_transfer.sh                # Entrypoint script for transfer learning experiments  
│  
├── train.py                       # Main executable file to start model training  
└── test.py                        # Main executable file to test a trained model     
```

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Jonghwan-dev/Awesome_Segmentation_in_Medical.git
   cd Awesome_Segmentation_in_Medical
   ```

2. **Conda (recommended)**

   ```bash
   conda env create -f environment.yml
   conda activate awesome_seg
   ```

3. **Download datasets**

   - Create a `datasets/` folder in the project root
   - For each public dataset (BUSI, BUSBRA, BUS-UC, BUS-UCLM, Yap2018), follow the original authors’ instructions to place images and masks under:
     ```
     datasets/
     ├── BUSI/
     │   ├── images/
     │   └── masks/
     └── BUSBRA/
         ├── images/
         └── masks/
     ```

---

## 🛠️ Data Preparation

Standardize raw data, generate CSV manifests and split files:

```bash
python -c "from data.prepare_datasets import PrepareDataset; PrepareDataset().run(['busi','busbra','bus_uc','bus_uclm','yap'])"
```

This produces:

- All dataset information is managed in CSV files located in the data/csv/ directory.  
  
- Each file (e.g., busi.csv, busbra.csv) corresponds to a single dataset.  
  
- Data splits for training and testing are defined by a split column directly within these files.  
  
---

## 🚀 Running Experiments

### 1. CNN Models

```bash
$ chmod u+x ./run_cnn.sh
bash ./run_cnn.sh
```

Performs k-fold cross-validation for UNet, AttUNet, UNet++ and UNet3+, then evaluates on the held‑out test set.

### 2. Transformer Models

```bash
$ chmod u+x ./run_vit.sh
bash ./run_vit.sh
```

Trains TransUNet, Swin-Unet and MedT with the same splits.

### 3. Transfer Learning

```bash
$ chmod u+x ./transfer_run.sh
bash ./transfer_run.sh
```

Fine-tunes CNN or Transformer backbones pretrained on natural images.

All logs and per-fold metrics are saved under `results/`.

---

## **Understanding the Results 📊**

The performance metrics reported in the tables represent the mean ± standard deviation derived from a 5-fold cross-validation process. The evaluation methodology is as follows:  
  
1. A dedicated, held-out test set is created and separated before any training begins.  
  
2. The remaining data is used for 5-fold cross-validation, which results in 5 independently trained models.  
  
3. Each of these 5 models is then individually evaluated on the entire held-out test set. This yields 5 separate performance scores for each metric (e.g., 5 Dice scores, 5 IoU scores).  
  
4. The final value reported in the table (e.g., Dice: 0.7095 ± 0.0300) is the average and standard deviation of these 5 scores.  
  
This method effectively demonstrates the model's stability and generalization performance across different training data subsets.  
  
### **BUSI Dataset Performance**

<p align="center">
<<<<<<< Updated upstream
  <em>TBD (To Be Determined): Some models are currently under training/validation. 
  <br>Results will be updated upon completion.</em>  
=======
  <em>TBD (To Be Determined): Some models are currently under training/validation. <br>Results will be updated upon completion.</em>
>>>>>>> Stashed changes
</p>

| Model | Dice (DSC) | IoU | HD95 | GFLOPs | Params (M) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **UNet** | 0.7095 ± 0.0300 | 0.6290 ± 0.0346 | 36.6824 ± 9.7886 | 50.11 | 34.53 |
| **AttUNet** | 0.7400 ± 0.0257 | 0.6631 ± 0.0270 | 37.5227 ± 3.6763 | 50.96 | 34.88 |
| **UNet++** | 0.7307 ± 0.0220 | 0.6545 ± 0.0230 | 38.3222 ± 7.5356 | 28.73 | 26.90 |
| **UNet 3+** | 0.7194 ± 0.0268 | 0.6402 ± 0.0303 | 34.5574 ± 6.3702 | 152.87 | 26.97 |
| **UNeXt** | 0.6955 ± 0.0305 | 0.6150 ± 0.0322 | 40.1467 ± 6.0638 | 0.42 | 1.47 |
| **CMUNet** | 0.6913 ± 0.0223 | 0.6129 ± 0.0223 | 41.1387 ± 4.6279 | 69.81 | 49.93 |
| **CMUNeXt** | 0.7217 ± 0.0092 | 0.6439 ± 0.0092 | 35.5400 ± 6.5903 | 5.66 | 3.15 |
| **U-KAN** | 0.7427 ± 0.0086 | 0.6689 ± 0.0105 | 37.5375 ± 6.3250 | 5.25 | 9.38 |
| **TransUnet** | 0.7226 ± 0.0166 | 0.6412 ± 0.0183 | 32.3411 ± 3.4289 | 75.17 | 179.07 |
| **MedT** | 0.5759 ± 0.0435 | 0.4900 ± 0.0461 | 53.7967 ± 9.0494 | 4.33 | 1.13 |
| **SwinUnet** | TBD | TBD | TBD | TBD | TBD |


### **BUS-UC Dataset Performance**

| Model | Dice (DSC) | IoU | HD95 | GFLOPs | Params (M) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **UNet** | 0.8905 ± 0.0068 | 0.8189 ± 0.0077 | 16.8561 ± 0.9736 | 50.11 | 34.53 |
| **AttUNet** | 0.8948 ± 0.0021 | 0.8234 ± 0.0027 | 16.4403 ± 0.6161 | 50.96 | 34.88 |
| **UNet++** | 0.9113 ± 0.0075 | 0.8463 ± 0.0099 | 10.8447 ± 1.8236 | 28.73 | 26.90 |
| **UNet 3+** | 0.8900 ± 0.0060 | 0.8181 ± 0.0070 | 16.3550 ± 0.8623 | 152.87 | 26.97 |
| **UNeXt** | 0.8921 ± 0.0153 | 0.8192 ± 0.0182 | 13.8724 ± 1.8751 | 0.42 | 1.47 |
| **CMUNet** | 0.8983 ± 0.0132 | 0.8271 ± 0.0181 | 11.8369 ± 2.4165 | 69.81 | 49.93 |
| **CMUNeXt** | 0.9049 ± 0.0025 | 0.8371 ± 0.0037 | 12.5182 ± 0.6259 | 5.66 | 3.15 |
| **U-KAN** | 0.8978 ± 0.0045 | 0.8282 ± 0.0057 | 12.9284 ± 1.4239 | 5.25 | 9.38 |
| **TransUnet** | TBD | TBD | TBD | TBD | TBD |
| **MedT** | 0.8703 ± 0.0046 | 0.7850 ± 0.0045 | 15.8071 ± 1.2704 | 4.33 | 1.13 |
| **SwinUnet** | TBD | TBD | TBD | TBD | TBD |


### **BUS-UCLM Dataset Performance**

| Model | Dice (DSC) | IoU | HD95 | GFLOPs | Params (M) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **UNet** | 0.7411 ± 0.0211 | 0.7115 ± 0.0196 | 42.8127 ± 3.2033 | 50.11 | 34.53 |
| **AttUNet** | 0.7264 ± 0.0444 | 0.6961 ± 0.0468 | 44.5037 ± 4.6944 | 50.96 | 34.88 |
| **UNet++** | 0.8214 ± 0.0177 | 0.7908 ± 0.0187 | 25.3912 ± 3.8422 | 28.73 | 26.90 |
| **UNet 3+** | 0.7864 ± 0.0336 | 0.7550 ± 0.0332 | 29.9108 ± 8.2026 | 152.87 | 26.97 |
| **UNeXt** | 0.7744 ± 0.0219 | 0.7434 ± 0.0233 | 33.6618 ± 3.4316 | 0.42 | 1.47 |
| **CMUNet** | 0.7625 ± 0.0248 | 0.7321 ± 0.0270 | 38.7341 ± 6.3967 | 69.81 | 49.93 |
| **CMUNeXt** | 0.7821 ± 0.0220 | 0.7490 ± 0.0226 | 34.9975 ± 4.9092 | 5.66 | 3.15 |
| **U-KAN** | 0.7540 ± 0.0912 | 0.7266 ± 0.0939 | 47.3326 ± 13.5931 | 5.25 | 9.38 |
| **TransUnet** | 0.7675 ± 0.0221 | 0.7352 ± 0.0219 | 34.6927 ± 7.3986 | 75.17 | 179.07 |
| **MedT** | TBD | TBD | TBD | TBD | TBD |
| **SwinUnet** | TBD | TBD | TBD | TBD | TBD |


### **BUSBRA Dataset Performance**

| Model | Dice (DSC) | IoU | HD95 | GFLOPs | Params (M) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **UNet** | 0.8704 ± 0.0063 | 0.7958 ± 0.0080 | 12.8747 ± 1.4224 | 50.11 | 34.53 |
| **AttUNet** | 0.8748 ± 0.0061 | 0.8021 ± 0.0074 | 11.5454 ± 1.0852 | 50.96 | 34.88 |
| **UNet++** | 0.8789 ± 0.0024 | 0.8064 ± 0.0031 | 10.6655 ± 0.6798 | 28.73 | 26.90 |
| **UNet 3+** | 0.8787 ± 0.0054 | 0.8053 ± 0.0064 | 11.5249 ± 1.0132 | 152.87 | 26.97 |
| **UNeXt** | 0.8562 ± 0.0064 | 0.7751 ± 0.0064 | 13.7112 ± 2.1483 | 0.42 | 1.47 |
| **CMUNet** | 0.8705 ± 0.0050 | 0.7950 ± 0.0061 | 11.2522 ± 1.0243 | 69.81 | 49.93 |
| **CMUNeXt** | 0.8756 ± 0.0043 | 0.8013 ± 0.0048 | 11.3662 ± 1.2649 | 5.66 | 3.15 |
| **U-KAN** | 0.8797 ± 0.0064 | 0.8049 ± 0.0065 | 10.1306 ± 0.8699 | 5.25 | 9.38 |
| **TransUnet** | TBD | TBD | TBD | TBD | TBD |
| **MedT** | 0.8151 ± 0.0053 | 0.7157 ± 0.0071 | 15.5866 ± 0.6548 | 4.33 | 1.13 |
| **SwinUnet** | TBD | TBD | TBD | TBD | TBD |


### **Yap2018 Dataset Performance**

| Model | Dice (DSC) | IoU | HD95 | GFLOPs | Params (M) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **UNet** | 0.5451 ± 0.0530 | 0.4506 ± 0.0448 | 62.3235 ± 21.4890 | 50.11 | 34.53 |
| **AttUNet** | 0.6072 ± 0.0411 | 0.5057 ± 0.0357 | 45.5852 ± 26.4043 | 50.96 | 34.88 |
| **UNet++** | 0.5769 ± 0.0541 | 0.4882 ± 0.0534 | 53.9371 ± 21.8446 | 28.73 | 26.90 |
| **UNet 3+** | 0.6294 ± 0.0263 | 0.5329 ± 0.0274 | 32.0159 ± 11.0018 | 152.87 | 26.97 |
| **UNeXt** | 0.5403 ± 0.0315 | 0.4455 ± 0.0317 | 58.4016 ± 10.0436 | 0.42 | 1.47 |
| **CMUNet** | 0.5511 ± 0.0295 | 0.4524 ± 0.0327 | 55.6650 ± 13.0657 | 69.81 | 49.93 |
| **CMUNeXt** | 0.6313 ± 0.0463 | 0.5446 ± 0.0466 | 53.3332 ± 17.7294 | 5.66 | 3.15 |
| **U-KAN** | 0.6083 ± 0.0410 | 0.5130 ± 0.0542 | 43.6525 ± 9.3918 | 5.25 | 9.38 |
| **TransUnet** | 0.6715 ± 0.0581 | 0.5806 ± 0.0582 | 35.7003 ± 11.6601 | 75.17 | 179.07 |
| **MedT** | TBD | TBD | TBD | TBD | TBD |
| **SwinUnet** | TBD | TBD | TBD | TBD | TBD |


---

## **How to Extend 🛠️**

### **Add a New Model**

1. Place your model's .py file in models/cnn\_models/ or models/transformer\_models/.  
2. Import your model in the corresponding \_\_init\_\_.py file.  
3. Add the new model's name (as a string) to the model\_name list in the relevant configs/\*.yml file.

### **Add a New Dataset**

1. Add the raw dataset folder to the datasets/ directory.  
2. In data/prepare\_datasets.py, add a new preparation method (e.g., \_prepare\_mynew\_dataset) inside the PrepareDataset class.  
3. Register your new method in the dispatcher dictionary within the run method.

---

## GitHub Topics
```text
awesome-list
medical-image-segmentation
breast-ultrasound
BUS
deep-learning
benchmark
reproducibility
computer-vision
python
pytorch
tensorflow
```

---

## Categories

- **📚 Papers**: Key papers and review materials
- **🗂️ Datasets**: Public BUS datasets and download information
- **🤖 Models**: Implemented segmentation model architectures
- **📈 Metrics & Evaluation**: Main evaluation metrics (Dice, IoU, HD95, etc.)
- **⚙️ Tools & Scripts**: Automation scripts and utility functions
- **💡 Examples & Demo**: Example code, GIF demonstrations, and usage guides
- **🏆 Benchmarks & Leaderboard**: Model performance comparison tables and rankings
- **❓ FAQ**: Frequently asked questions and troubleshooting tips

---

## **Acknowledgements 🙏**

+ This project's structure and methodology are heavily inspired by [Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks).

+ Helper functions from [CMU-Net](https://github.com/FengheTan9/CMU-Net) and [Image\_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation) were also utilized. We extend our gratitude to the authors of these repositories for making their excellent work public.

---

## License & Citation 📜

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
If you use this benchmark in your research, please consider citing it:
```
@misc{awesomebusbenchmark2024,  
  author \= {Jonghwan Kim},  
  title \= {Awesome Segmentation in Medical Imaging: A Breast Ultrasound Benchmark},  
  year \= {2025},  
  publisher \= {GitHub},  
  journal \= {GitHub repository},  
  howpublished \= {\\url{https://github.com/Jonghwan-dev/Awesome\_Segmentation\_in\_Medical}},  
}
```
