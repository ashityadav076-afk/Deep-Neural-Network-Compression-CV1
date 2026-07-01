# Deep-Neural-Network-Compression-CV1
Developed a deep learning and computer vision project focused on compressing CNN models through pruning, quantization, and Huffman coding to enable efficient deployment on resource-constrained devices.
A PyTorch-based implementation of a neural network compression framework that combines iterative pruning, weight quantization, and compressed model storage to reduce the memory footprint of a VGG-inspired CNN trained on the CIFAR-10 dataset. The project focuses on achieving efficient deployment while maintaining competitive classification performance.
Overview
Deep neural networks often contain millions of parameters, making them expensive to deploy on devices with limited computational resources. This project demonstrates a complete compression workflow that progressively reduces model size without significantly affecting prediction accuracy.
The pipeline includes:
Training a baseline convolutional network
Magnitude-based weight pruning
Fine-tuning the sparse network
K-Means weight quantization
Model compression and storage
Reconstruction for inference
Compression Workflow
The complete pipeline is executed in the following order:
Train the original CNN
Perform iterative weight pruning
Fine-tune the pruned model
Quantize remaining weights using K-Means clustering
Save the compressed parameters
Reload the compressed model
Validate inference performance
Features
Weight Pruning
Iterative magnitude-based pruning strategy
Removes parameters with low importance
Binary masks preserve sparsity during training
Weight rewinding improves optimization after each pruning stage
Weight Quantization
Applies K-Means clustering to non-zero weights
Replaces floating-point values with shared centroids
Supports configurable cluster counts
Current implementation uses 8 clusters (3-bit representation)
Model Compression
Stores compressed weights using NumPy's compressed format (.npz)
Significantly lowers storage requirements
Enables efficient serialization and loading
Deployment
Reconstructs the compressed network
Loads parameters directly from compressed files
Performs inference without additional retraining
Installation
Create a virtual environment and install the required packages.
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision numpy matplotlib scikit-learn
Running the Project
Execute Complete Compression Pipeline
python main.py
The script performs:
Baseline training
Iterative pruning
Fine-tuning
K-Means quantization
Model compression
Performance evaluation
Test the Compressed Model
python deploy.py
This script:
Loads the compressed network
Restores the model parameters
Executes inference on sample inputs
Determine Optimal Number of Clusters
python analysis/kmeans_elbow.py
This utility generates an elbow curve to help select an appropriate number of quantization clusters.
Project Directory
.
├── models/
│   └── model.py                 # CNN architecture
│
├── data/
│   └── data_loader.py           # Dataset loading and preprocessing
│
├── utils/
│   ├── train.py                 # Training and evaluation functions
│   └── metrics.py               # Compression and sparsity metrics
│
├── compression/
│   ├── prune.py                 # Magnitude pruning
│   ├── rewind.py                # Weight rewinding
│   ├── quantize.py              # K-Means quantization
│   └── huffman.py               # Compression utilities
│
├── deployment/
│   └── load_model.py            # Load compressed models
│
├── analysis/
│   └── kmeans_elbow.py          # Cluster selection analysis
│
├── compressed_models/           # Saved compressed networks
├── config.py                    # Configuration and hyperparameters
├── main.py                      # Complete training pipeline
└── deploy.py                    # Inference script
Using the Compressed Model
Run
python deploy.py
or import it inside another script:
from deployment.load_model import (
    load_compressed_model,
    run_inference,
)

model = load_compressed_model(
    "compressed_models/vgg_compressed.npz"
)

run_inference(model)
Performance Summary
Metric	Result
Original Model Size	16.31 MB
Compressed Model Size	1.59 MB
Compression Ratio	10.28×
Accuracy Reduction	Less than 2%
Effective Weight Precision	~3 bits
Highlights
Multi-stage CNN compression pipeline
Sparse training through iterative pruning
K-Means-based weight sharing
Compact model storage using NumPy
End-to-end deployment workflow
Minimal loss in classification accuracy
Additional Notes
Training is computationally intensive on CPU and may require several hours.
For faster experimentation, reduce the number of training epochs in config.py.
Ensure the compressed_models/ directory exists before saving compressed checkpoints.
Technologies Used
Python
PyTorch
NumPy
Scikit-learn
Matplotlib
