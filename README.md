# 🚀 Distributed Deep Learning for Breast Cancer Classification

A powerful distributed training system that leverages multiple workers to train a Convolutional Neural Network for breast cancer detection from medical imaging data. Built with modern technologies like Celery, Redis, and TensorFlow/Keras, this project demonstrates how to scale machine learning workloads across multiple compute nodes.

## 🎯 What This Project Does

This isn't just another machine learning project - it's a complete distributed training pipeline that solves real-world scalability challenges. The system automatically:

- **Distributes training data** across multiple workers for parallel processing
- **Aggregates model updates** using federated averaging techniques - Sending model directly where the data exists, precisely, this project implements simulated federated learning (Split the data between the workers and running training on its own model instance and get weights form models and weights are aggregated using layer wise averaging.)
- **Applies online data augmentation** using Albumentations during training for better generalization and model robustness
- **Tracks comprehensive metrics** throughout the training process
- **Saves the final trained model** for deployment (best model selection coming in next update)


### The Problem We're Solving

Training deep learning models on large datasets can take hours or even days on a single machine. This project tackles that challenge by:

1. **Breaking down large training batches** into smaller chunks that multiple workers can process simultaneously
2. **Coordinating model updates** so each worker contributes to a shared, continuously improving model
3. **Monitoring training progress** with real-time metrics and validation scores

## 🏗️ Architecture Overview

```txt
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Worker 1      │    │   Worker 2      │    │   Worker 3      │
│                 │    │                 │    │                 │
│ • Trains on     │    │ • Trains on     │    │ • Trains on     │
│   data batch    │    │   data batch    │    │   data batch    │
│ • Computes      │    │ • Computes      │    │ • Computes      │
│   gradients     │    │   gradients     │    │   gradients     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Aggregation     │
                    │ Task            │
                    │                 │
                    │ • Averages      │
                    │   weights       │
                    │ • Evaluates     │
                    │   performance   │
                    │ • Updates       │
                    │   global model  │
                    └─────────────────┘
```

## 🛠️ Technologies Under the Hood

- **🧠 TensorFlow/Keras**: Deep learning framework for building and training the CNN
- **⚡ Celery**: Distributed task queue for coordinating workers
- **🗄️ Redis**: High-performance message broker and result backend
- **📊 scikit-learn**: Model evaluation and metrics calculation
- **🖼️ OpenCV**: Image preprocessing and augmentation
- **📈 NumPy**: Numerical computations and array operations

## 🚀 Quick Start Guide

### Prerequisites

Make sure you have these installed:
- Python >=3.13
- Redis server
- At least 8GB RAM (recommended for the breast cancer dataset)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Distributed_training_
```

2. **Set up the virtual environment**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. **Install dependencies**

**Option A: For exact reproducibility (recommended)**
```bash
pip install -r requirements_lock.txt
```
The `requirements_lock.txt` file contains all sub-dependencies with pinned versions used during development. This ensures you get the exact same environment for complete reproducibility.

**Option B: For latest compatible versions**
```bash
pip install -r requirements.txt
```
This installs the direct dependencies and lets pip resolve the latest compatible sub-dependencies.

**Option C: Upgrade existing installation**
```bash
pip install -r requirements.txt --upgrade
```
Use this to update to the latest compatible versions of all packages.

4. **Start Redis server**
```bash
# On Windows (if installed via Chocolatey):
redis-server
# On Linux:
sudo systemctl start redis
# On Mac (via Homebrew):
brew services start redis
```

### Setting Up Your Data

1. **Prepare your dataset**
   - Place your breast cancer images in `vanitha_mam_breast_cancer_dataset/`
   - Organize them in folders: `0/` for non-cancerous, `1/` for cancerous images

2. **Preprocess the data**
```bash
python preprocessing/load_and_create_dataset.py
```

This creates train/validation/test splits and saves them as numpy arrays in the `preprocessed/` folder.

### Running Distributed Training

1. **Start Celery workers** (run this in separate terminals for each worker)
```bash
# Terminal 1
celery --app=celery_app worker --pool=solo --loglevel=info --hostname=worker1@%h

# Terminal 2
celery --app=celery_app worker --pool=solo --loglevel=info --hostname=worker2@%h

# Terminal 3
celery --app=celery_app worker --pool=solo --loglevel=info --hostname=worker2@%h
```

> **Note**: A convenient `.bat` file will be introduced in a future update to streamline the process of starting multiple workers simultaneously.


2. **Launch the training process**
```bash
python distributed_training_index.py
```

Workers collaborate to train the model! 🎉

## ⚙️ Configuration

Customize your training in `helpers/global_config.py`:

```python
# Sample view of the class. Peak for config params.
class Config:
    INPUT_SHAPE = (128,128)     # Image resolution
    num_workers = 3             # Number of distributed workers
    EPOCHS = 10                 # Training epochs
    BASE_LR = 0.001            # Base learning rate
    TRAIN_BATCH_SIZE = 2       # Batch size for tensorflow batch chain
```


> **Note**: In future iterations, this implementation will support dynamic scaling by automatically adjusting the number of workers based on the actual number of available clients detected in the system.

## 📊 What You'll Get

During training, the system automatically tracks:

- **Training Loss**: How well the model fits the training data
- **Validation Accuracy**: Model performance on unseen data
- **Precision & Recall**: Detailed classification metrics
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating curve

All metrics are saved to CSV files in the `metrics/` directory for easy analysis.

## 🏆 Model Architecture

The CNN architecture is specifically designed for medical image classification:

```python
Sequential([
    # Feature extraction layers
    Conv2D(128, kernel_size=3, activation="relu"),
    MaxPooling2D((2,2)),
    BatchNormalization(),
    Dropout(0.2),

    Conv2D(64, kernel_size=(3,3), activation="relu"),
    MaxPooling2D((2,2)),
    BatchNormalization(),
    Dropout(0.1),

    Conv2D(32, kernel_size=(2,2), activation='relu'),
    BatchNormalization(),

    # Classification layers
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

## 🔧 Key Components

### `distributed_training_index.py`
The main orchestrator that:
- Loads preprocessed data
- Distributes training batches across workers
- Saves the final trained model

### `tasks/train_task.py`
Individual worker logic that:
- Receives a data batch and current model weights
- Performs one epoch of training
- Returns updated weights and predict on validation data.

### `tasks/aggregate_task.py`
The aggregation logic that:
- Collects weights from all workers
- Averages them using federated learning principles
- Evaluates the aggregated model
- Records performance metrics

### `helpers/model_instance_helper.py`
Model management utilities:
- Creates consistent model architecture across workers
- Handles weight serialization/deserialization
- Manages model compilation and optimization

## 🎓 Learning Opportunities

This project demonstrates several advanced concepts:

- **Distributed Computing**: Learn how to coordinate multiple processes across multiple workers
- **Federated Learning**: Understand weight averaging and model synchronization without centralizing data
- **Asynchronous Processing**: Master Celery task queues and Redis for coordinating distributed workloads
- **Online Data Augmentation**: Apply real-time image transformations during training to improve model generalization
- **MLOps**: Practice model versioning, metrics tracking, and reproducibility in production settings

## 🚨 Troubleshooting

**Redis Connection Issues**
```bash
# Check if Redis is running
redis-cli ping
# Should return "PONG"
```

**Worker Not Starting**
```bash
# Check Celery worker logs
celery -A celery_app worker --loglevel=debug
```

**Out of Memory**
- Reduce `TRAIN_BATCH_SIZE` in config
- Use fewer workers
- Resize images to smaller resolution

## 🤝 Upcoming but also for Contributing

This project is open for improvements! Some ideas that may be covered in upcoming updates:

- Add support for different model architectures
- Implement dynamic worker scaling
- Add real-time training visualization
- Support for other medical imaging datasets
- GPU acceleration for workers (**Will be working on it.**)


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

