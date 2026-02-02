# Tsuikami: Robust Cross-Channel Source Speaker Tracing Under Voice Conversion

This repository contains the official implementation.  
It supports dataset construction, training/testing data preparation, model training, and evaluation based on **VoxCeleb** and **LibriSpeech** datasets.

---

## 1. Environment Setup

### 1.1 Install Dependencies

It is recommended to use `conda` or `virtualenv`.

```bash
pip install -r requirements.txt
```

## 2. Dataset Preparation
### 2.1 Download Datasets

Please manually download and extract the following datasets:

+ VoxCeleb1&2 Official site: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

+ LibriSpeech Official site: https://www.openslr.org/12

## 3. Build Clean Dataset CSV Files
### 3.1 Voxceleb1&2 and LibriSpeech Clean CSV

Convert Voxceleb1&2 and LibriSpeech audio samples into a CSV annotation file:

```bash
data/csv/librispeech_clean_data_demo.csv
```

## 4. Build Voice Conversion (VC) CSV Files
### 4.1 VC Training and Test CSV

Construct VC training pairs (source–target speaker pairs) using LibriSpeech and VoxCeleb with `get_train_vc_csv.py`.
Construct VC test pairs based on LibriSpeech with `get_test_csv_librispeech.py`.

Key CSV fields:


+ src_audio: source speaker audio
+ tar_audio: target speaker audio
+ vc_audio: voice conversion audio


## 5. Generate VC Audio

Using the selected voice conversion algorithm, generate converted speech (`vc_audio`) based on
`src_audio` and `tar_audio` specified in the training and test CSV files.

+ Training set: used for model learning
+ Test set: used for evaluation

The generated `vc_audio` paths should be written back to the corresponding CSV files.

## 6. Train / Validation Split

According to the VC training CSV file in Step 4, you need to split the dataset into training and dev sets with a 95:5 ratio. The VC training data is organized as follows:

```
data/csv/DDDMVC_train_data/
├── all.csv
├── train.csv
├── dev.csv
```

+ train.csv: training set
+ dev.csv: validation set
+ all.csv: full training set

## 7. Configuration File Setup

Edit the hyperparameter configuration file: `hparams/vctrace_circle_xxxx_fbank.yaml`.

### 7.1 Weights & Biases Configuration

```
wandb_project: your_project_name
wandb_name: your_experiment_name
wandb_id: your_experiment_id
```

### 7.2 Dataset Annotation Paths

```
train_annotation: ./data/csv/DDDMVC_train_data/train.csv
valid_annotation: ./data/csv/DDDMVC_train_data/dev.csv
vc_annotation: ./data/csv/DDDMVC_train_data/all.csv
voxceleb_annotation: ./data/csv/voxceleb_clean_data.csv
librispeech_annotation: ./data/csv/librispeech_clean_data.csv
```

## 8. Model Training

Start training using torchrun:

```bash
torchrun --standalone --nproc_per_node=1 \
    train_midlayer_mixup.py \
    hparams/vctrace_circle_xxxx_fbank.yaml
```

## 9. Model Testing

For evaluation, please refer to the script:

```bash
scripts/overall/circle.sh
```

Modify model paths and evaluation settings as needed before running.

