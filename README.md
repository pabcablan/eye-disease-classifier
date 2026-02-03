<div align="center">

# 👁️ Eye Disease Classifier

### *Automated Eye Disease Classification using Deep Learning*

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue?style=for-the-badge)](LICENSE)

*Eye disease classification system using convolutional neural networks for detection of cataracts, glaucoma, diabetic retinopathy, and normal eyes.*

</div>

## 📜 Description

**Eye Disease Classifier** is a research project that compares different **Convolutional Neural Network (CNN)** architectures for classifying eye diseases from fundus images.

The system implements two complementary approaches:

- **5 custom CNN architectures** trained from scratch, experimenting with different layer configurations, activations, and optimizers
- **Transfer Learning with DenseNet169** pre-trained on ImageNet, applying selective fine-tuning to adapt it to the medical domain

The goal is to automatically detect pathologies such as **cataracts, glaucoma, and diabetic retinopathy**, distinguishing them from healthy eyes. This project is based on findings from the study [*Deep Learning for Automated Detection and Classification of Eye Diseases*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12464438/) (PMC, 2024).

## 📊 Dataset

We use the public **[Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)** dataset from Kaggle, which contains fundus images classified into four categories:

- **Cataract**
- **Diabetic Retinopathy**
- **Glaucoma**
- **Normal** (healthy eye)

**Data split:** 80% Training | 10% Validation | 10% Testing

The dataset is automatically downloaded using `opendatasets` when running the notebooks.

## 🏗️ Implemented Architectures

### 🔬 CNNs from Scratch (5 variants)

We experimented with five custom convolutional architectures, varying:

- Network depth (3-5 convolutional blocks)
- Activation functions (ReLU, LeakyReLU)
- Optimizers (Adam, AdamW, SGD)
- Regularization techniques (Dropout, Batch Normalization, 2D Dropout)
- Pooling strategies (Max Pooling, Global Average Pooling)

#### Comparative Results

| Model | Key Architecture | Optimizer | Epochs | Test Accuracy |
|--------|-------------------|-----------|--------|---------------|
| **Model 1** | 4 deep blocks (→256 ch) | Adam | 113 | 90.05% |
| **Model 2** | 3 shallow blocks (→128 ch) | Adam | 45 | 86.02% |
| **Model 3** | 5 blocks + 2D Dropout | AdamW + ReduceLR | 69 | 87.91% |
| **Model 4** | 4 blocks + LeakyReLU | SGD + StepLR | 61 | **90.52%** 🥇 |
| **Model 5** | 4 blocks + Global Avg Pooling | Adam + CosineAnnealing | 92 | **90.52%** 🥇 |

**Key Conclusions:**
- **Models 4 and 5** achieve the best performance (~90.5%) with more efficient architectures
- Model 2 shows **clear overfitting** (98% train vs 86% test) due to lack of regularization
- Greater depth (Model 3) does not guarantee better accuracy
- **Classic SGD** (Model 4) competes with Adam in final results
- **Global Average Pooling** (Model 5) reduces parameters without sacrificing precision

### 🚀 Transfer Learning with DenseNet169

We implemented transfer learning using DenseNet169 pre-trained on ImageNet with a **selective fine-tuning** strategy:

- **Frozen**: DenseBlock 1 and 2 (general features)
- **Trained**: DenseBlock 3, 4, norm5, and custom head
- **Optimizer**: Adamax with ReduceLROnPlateau
- **Loss**: CrossEntropyLoss with label smoothing (0.1)

```python
class CustomHead(nn.Module):        
    def __init__(self, in_features, num_classes):
        super(CustomHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

**Transfer Learning Results:**

| Metric | Training | Validation | Testing |
|---------|----------|------------|---------|
| **Accuracy** | 99.88% | 94.55% | **94.79%** |
| **Loss** | 0.352 | 0.497 | - |

## 📈 Final Comparison

| Approach | Best Model | Test Accuracy | Main Advantage |
|---------|--------------|---------------|-------------------|
| **CNN from scratch** | Model 4 / Model 5 | 90.52% | Full architecture control |
| **Transfer Learning** | DenseNet169 | **94.79%** | Fast convergence + better precision |

**Relative improvement:** +4.27% with Transfer Learning

## 🔧 Requirements

- **Python 3.12+** (tested on Python 3.12.7)
- **PyTorch 2.0+** (installation depends on GPU/CPU from [pytorch.org](https://pytorch.org/))
- **CUDA GPU** (optional but recommended)
- **Kaggle account** (for dataset download)

**Check GPU (Windows/Linux):**
```bash
nvidia-smi
```

## 🚀 Installation and Usage

### 1️⃣ Clone repository

```bash
git clone https://github.com/pabcablan/eye-disease-classifier.git
cd eye-disease-classifier
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install PyTorch

> ⚠️ **Important:** PyTorch must be installed according to your hardware configuration (CPU/GPU).

**Check your GPU (if you have NVIDIA):**
```bash
nvidia-smi
```

Then install PyTorch from the **[official website](https://pytorch.org/)** by selecting your CUDA configuration.

**Example for GPU with CUDA 12.8:**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Example for CPU:**
```bash
pip install torch torchvision
```

### 4️⃣ Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Configure Kaggle API

Create `~/.kaggle/kaggle.json`:
```json
{
  "username": "your_kaggle_username",
  "api_key": "your_api_key"
}
```

**On Linux/Mac:**
```bash
chmod 600 ~/.kaggle/kaggle.json
```

> 💡 Get your credentials at [Kaggle Settings](https://www.kaggle.com/settings) → API → Create New Token

### 6️⃣ Run notebooks

**CNNs from scratch:**
```bash
jupyter notebook convolutional_neural_network.ipynb
```

**Transfer Learning:**
```bash
jupyter notebook transfer_learning_cnn.ipynb
```

The notebooks automatically download the dataset, detect GPU, and save models in `models/`.

## 📁 Project Structure

```
eye-disease-classifier/
│
├── models/                                # Experiment results
│   ├── model1/
│   │   ├── *.pth                         # Trained model
│   │   ├── accuracy_graphic.png
│   │   ├── confusion_matrix.png
│   │   └── loss_graphic.png
│   ├── model2/
│   ├── model3/
│   ├── model4/
│   ├── model5/
│   └── transfer_learning/
│       ├── best_model.pth
│       ├── accuracy_graphic.png
│       ├── confusion_matrix.png
│       └── loss_graphic.png
│
├── .gitignore
├── LICENSE
├── README.md
├── convolutional_neural_network.ipynb    # 5 CNNs from scratch
├── requirements.txt
└── transfer_learning_cnn.ipynb           # Transfer Learning DenseNet169
```

> **Note:** The `data/` folder (with the dataset) is automatically generated when running the notebooks and is excluded from the repository via `.gitignore`.

## 🔬 Technologies

- **PyTorch 2.0+** – Deep learning framework
- **DenseNet169** – Architecture pre-trained on ImageNet
- **Weights & Biases** – Experiment tracking
- **scikit-learn** – Metrics and evaluation
- **opendatasets** – Automatic download from Kaggle
- **Jupyter Notebook** – Interactive development environment

## 👥 Authors

This project was created by [ArtHead](https://github.com/ArtHead-Devs), with two members:

- **👨‍💻 Fabio Nesta Arteaga Cabrera**: [NestX10](https://github.com/NestX10)
- **👨‍💻 Pablo Cabeza Lantigua**: [pabcablan](https://github.com/pabcablan)

## 📚 References

1. "Deep Learning for Automated Detection and Classification of Eye Diseases". *PMC*, 2024. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12464438/)
2. [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 📄 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.
