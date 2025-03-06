# MSLSTM Project

## Overview
This project focuses on analyzing MSLSTM using deep learning models with PyTorch Lightning. It includes feature extraction, classification, and model evaluation modules.

## Dependencies
To run this project, install the following dependencies:

### Required Python Version
- Python 3.12

### Required Libraries
Install the dependencies using `pip`:
```sh
pip install torch torchvision torchaudio pytorch-lightning
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install tqdm
```
Alternatively, if using `conda`:
```sh
conda create -n bgp python=3.12
conda activate bgp
conda install pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge pytorch-lightning numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Project Structure
```
BGP-Security/
│── code/
│   ├── sendXiaohui/
│   │   ├── test_model.py          # Script to test the model
│   │   ├── classification.py      # Model definition
│   │   ├── feature_extraction.py  # Feature extraction utilities
│── data/                          # Directory for storing data
│── checkpoints/                   # Directory for storing model checkpoints
```

## Input and Output
### Input:
- BGP data stored in text files.
- Model checkpoint (`.ckpt`) for loading pre-trained models.

### Expected Output:
- Logs with training/testing results.
- Classification results on test data.

## Running the Project from Scratch

### Step 1: Clone the Repository
```sh
git clone <repository_url>
cd BGP-Security
```

### Step 2: Activate Environment
```sh
conda activate bgp  # If using conda
```

### Step 3: Prepare Data
Place your BGP dataset inside the `data/` directory.

### Step 4: Train the Model (Optional)
If training from scratch, run:
```sh
python code/sendXiaohui/train_model.py
```

### Step 5: Run Model Testing
If using a pre-trained model, make sure the checkpoint is available and run:
```sh
python code/sendXiaohui/test_model.py --checkpoint_path checkpoints/model.ckpt
```

### Step 6: Debugging Feature Extraction Issues
If you encounter issues with feature extraction (e.g., invalid indices like `pytorch-lightning_version`), ensure that:
- Indices are properly converted from strings to integers.
- Invalid keys are skipped or handled correctly.

## Troubleshooting
- If you see an error like `expected string or bytes-like object, got 'NoneType'`, check the dataset loading logic.
- If `pytorch-lightning_version` appears in indices, modify `__getitem__` in `feature_extraction.py` to skip it properly.

## Acknowledgments
This project is based on PyTorch Lightning for deep learning model development.

## License
This project is licensed under the MIT License.

