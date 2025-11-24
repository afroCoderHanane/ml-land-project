# CNN-Based Land Use Classification

A deep learning project for classifying satellite imagery into urban (City) and agricultural (Farmland) land use categories using Convolutional Neural Networks and transfer learning.

## 👥 Team Members

- **Haider Amin** - amin.h@northeastern.edu
- **Abdoul-Hanane Gbadamassi** - gbadamassi.a@northeastern.edu

## 🎯 Project Objectives

This project develops a CNN-based classifier capable of distinguishing between two key land use categories using satellite imagery:

- **City**: Urban areas including residential, industrial, and highway infrastructure
- **Farmland**: Agricultural land including annual crops, permanent crops, and pastures

### Significance

Land Use and Land Cover (LULC) classification is critical for:
- Urban planning and development
- Environmental monitoring
- Natural resource management
- Infrastructure planning and policy development

By leveraging deep learning, this project demonstrates how automated, scalable classification can provide data-driven insights for city planners, engineers, and architects to understand land usage patterns and plan for sustainable growth.

## 📊 Dataset

**EuroSAT Dataset**
- Source: Sentinel-2 satellite imagery
- Total Images: 27,000 labeled patches (64×64 pixels)
- Original Classes: 10 land use categories
- Our Binary Mapping:
  - **City (Class 0)**: Residential, Industrial, Highway
  - **Farmland (Class 1)**: AnnualCrop, PermanentCrop, Pasture
  - **Excluded**: Forest, HerbaceousVegetation, River, SeaLake

The dataset is automatically downloaded via PyTorch's `torchvision.datasets.EuroSAT`.

## 🏗️ Model Architecture

**Transfer Learning with ResNet-50**

- **Backbone**: ResNet-50 pre-trained on ImageNet
- **Strategy**: Fine-tuning with frozen early layers
- **Classification Head**:
  ```
  Linear(2048 → 512) → ReLU → Dropout(0.3) → Linear(512 → 2)
  ```
- **Input**: 64×64 RGB satellite images
- **Output**: Binary classification (City vs Farmland)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**:
   ```bash
   cd ml-land-project
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### 1. Training

Train the model with default settings:

```bash
python src/train.py
```

**Custom training options**:

```bash
python src/train.py --epochs 30 --batch_size 64 --lr 0.001 --device auto
```

**Arguments**:
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device to use (`auto`, `cuda`, `mps`, `cpu`)
- `--patience`: Early stopping patience (default: 5)
- `--data_root`: Data directory (default: `./data`)
- `--checkpoint_dir`: Checkpoint directory (default: `./checkpoints`)

**Outputs**:
- Best model: `checkpoints/best_model.pth`
- Training history: `results/training_history_<timestamp>.csv`

### 2. Evaluation

Evaluate the trained model on the test set:

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

**Outputs**:
- Metrics JSON: `results/metrics.json`
- Confusion matrix: `results/confusion_matrix.png`
- Training curves: `results/training_curves.png`
- Sample predictions: `results/sample_predictions.png`

**Sample Metrics**:
```
Overall Accuracy: 89.45%

Per-Class Metrics:
-------------------------------------------------------------
Class           Precision    Recall       F1-Score     Support
-------------------------------------------------------------
City            0.9123      0.8756      0.8935      1245
Farmland        0.8801      0.9134      0.8964      1312
-------------------------------------------------------------
Weighted Avg    0.8957      0.8945      0.8950
```

### 3. Inference

Predict land use for a single satellite image:

```bash
python src/inference.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth
```

**Output**:
- Predicted class with confidence
- Visualization showing input image and prediction probabilities

### 4. Data Exploration

Explore the dataset interactively:

```bash
jupyter notebook notebooks/exploration.ipynb
```

## 📁 Project Structure

```
ml-land-project/
├── data/                    # Dataset (auto-downloaded)
│   └── eurosat/
├── src/                     # Source code
│   ├── __init__.py
│   ├── dataset.py          # Data loading and preprocessing
│   ├── model.py            # Model architecture
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── inference.py        # Inference script
├── notebooks/               # Jupyter notebooks
│   └── exploration.ipynb   # Data exploration
├── checkpoints/             # Saved model checkpoints
├── results/                 # Evaluation results and plots
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## 🔬 Technical Details

### Data Augmentation (Training)
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Normalization (ImageNet statistics)

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy with class weights
- **Learning Rate**: 0.001 with step decay (γ=0.1 every 10 epochs)
- **Early Stopping**: Patience of 5 epochs
- **Data Split**: 70% train, 15% validation, 15% test

### Hardware Requirements
- **GPU**: NVIDIA CUDA or Apple Metal (MPS) recommended
- **CPU**: Works on CPU (slower training)
- **Memory**: 8GB+ RAM recommended
- **Storage**: ~500MB for dataset

## 📈 Expected Results

Based on literature and preliminary experiments:
- **Target Accuracy**: >85% on test set
- **Training Time**: 30-60 minutes on GPU
- **Convergence**: Around epoch 15-20
- **Balanced Performance**: Precision and recall >0.80 for both classes

## 🔮 Future Enhancements

1. **Multi-Class Classification**: Expand to all 10 EuroSAT classes
2. **Advanced Architectures**: EfficientNet, Vision Transformers (ViT)
3. **Interpretability**: Class Activation Mapping (CAM) for visualization
4. **Deployment**: Web interface using Streamlit or Gradio
5. **Production Ready**: Export to ONNX for optimized inference

## 📚 References

1. Mijanur, R. (2024). *Deep Learning CNN Model for Land Use Land Cover Classification Using Remote Sensing Images*. Medium. [Link](https://medium.com/@rmijanur10266/deep-learning-cnn-model-for-land-use-land-cover-classification-u...)

2. Helber, P., et al. (2023). *Land Use and Land Cover Classification Meets Deep Learning: A Review*. PMC. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025814/)

3. Study Hacks (Institute of GIS and Remote sensing). (2024). *Deep Learning CNN Model for Land Use Land Cover Classification Using Remote Sensing Images* [Video]. YouTube. [Link](https://www.youtube.com/watch?v=kwZZ1YM5TYw)

4. Climate Change AI. (2024). *Classifying Land Cover with Deep Learning: Sentinel-2 & PyTorch Tutorial* [Video]. YouTube. [Link](https://www.youtube.com/watch?v=8gn-Sg9GzoM)

5. Climate Change AI. (2024). *Land Use and Land Cover Classification using Pytorch: Part 1* [Video]. YouTube. [Link](https://www.youtube.com/watch?v=6Cdwwlkkz80)

## 📝 License

This project is for educational purposes as part of coursework at Northeastern University.

## 🤝 Contributing

For questions or collaboration:
- Haider Amin: amin.h@northeastern.edu
- Abdoul-Hanane Gbadamassi: gbadamassi.a@northeastern.edu

---

**Built with ❤️ for sustainable urban development and planning**
