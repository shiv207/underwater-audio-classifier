# Underwater Acoustic Classifier

## Advancing Marine Sound Intelligence

This project introduces a sophisticated AI-powered system for the classification of underwater acoustic events. Leveraging state-of-the-art deep learning methodologies, this classifier is designed to accurately identify and categorize diverse marine soundscapes.

## Core Capabilities

The system precisely distinguishes underwater sounds into four distinct categories:

-   **Marine Fauna**: Comprehensive detection and classification of whales, dolphins, and other aquatic species.
-   **Vessel Presence**: Identification of sounds originating from maritime vessels, including boats, ships, and submarines.
-   **Natural Hydroacoustics**: Analysis of environmental sounds such as waves, bubbles, and other water-borne phenomena.
-   **Anthropogenic & Other Sounds**: Detection of various human-made noises and unclassified acoustic signatures.

## Getting Started

To deploy and utilize the Underwater Acoustic Classifier, follow these streamlined instructions:

### 1. Environment Setup

Ensure all necessary dependencies are installed by executing the following command:

```bash
pip install -r requirements.txt
```

### 2. Model Training (Optional)

Pre-trained models are provided for immediate use. For custom training or model fine-tuning, execute:

```bash
python train_minimal.py
```

### 3. Launch Application

Initiate the interactive web application to access the classification interface:

```bash
streamlit run app.py
```

Upon successful launch, navigate to the provided local URL in your web browser. From there, you can select a classification model, upload an audio file or choose from integrated samples, and proceed with sound analysis.

## Integrated Models

The system integrates two robust deep learning models, each optimized for performance and accuracy:

### Fine-tuned ResNet18 (Recommended)

-   **Accuracy**: Achieves an exceptional 98.33% classification accuracy.
-   **Methodology**: Employs transfer learning from a pre-trained ImageNet model, adapted for specific underwater acoustic features.
-   **Deployment**: Available at `models/best_model_finetuned.pth`.
-   **Training Profile**: Approximately 10 minutes over 4 epochs, demonstrating efficient convergence.

### Simple Convolutional Neural Network (CNN)

-   **Accuracy**: Delivers a strong 93% classification accuracy.
-   **Methodology**: Custom-built and trained from scratch to provide a foundational classification capability.
-   **Deployment**: Available at `models/best_model_simple.pth`.
-   **Training Profile**: Approximately 20 minutes over 18 epochs.

## Command Line Interface

For direct interaction and batch processing, the following command-line utilities are provided:

```bash
# Classify an audio file using the fine-tuned model
python predict_minimal.py --audio sound.wav

# Classify an audio file using the simple CNN model
python predict_minimal.py --audio sound.wav --model models/best_model_simple.pth

# Generate a JSON output compliant with Grand Challenge UDA specifications
python generate_json.py --audio sound.wav --output result.json
```

## Project Architecture

The project is structured for modularity and scalability:

```
uda_model/
├── app.py                          # Streamlit-based web interface for interactive classification
├── train_minimal.py                # Script for model training and fine-tuning
├── predict_minimal.py              # Script for executing predictions on audio inputs
├── simple_model.py                 # Defines the architectural structure of the CNN model
├── simple_data.py                  # Manages data loading, preprocessing, and augmentation
├── simple_config.py                # Centralized configuration parameters for the system
├── requirements.txt                # Lists all requisite Python dependencies
├── data/training/                  # Repository for training datasets
└── models/                         # Storage for trained model checkpoints
```

## Operational Overview

### Fine-Tuning Methodology

1.  **Pre-trained Initialization**: A ResNet18 model, pre-trained on ImageNet, is loaded.
2.  **Feature Extraction Freezing**: Early layers are frozen to preserve generalized visual features.
3.  **Domain Adaptation**: Subsequent layers are fine-tuned to adapt to the unique characteristics of underwater acoustics.
4.  **Classifier Integration**: A new classification head is trained to categorize the four distinct sound classes.

### Data Processing Pipeline

1.  **Audio Ingestion**: Audio files are processed, with a maximum duration of 10 seconds.
2.  **Spectrogram Conversion**: Audio signals are transformed into 128x1024 mel spectrograms.
3.  **Augmentation**: Advanced augmentation techniques (speed, pitch, noise injection) are applied to enhance model robustness.
4.  **Normalization & Inference**: Processed data is normalized and fed into the deep learning model for inference.

## System Configuration

Parameters such as batch size, learning rate, number of epochs, and audio processing settings can be adjusted within `simple_config.py` to optimize performance or adapt to specific requirements.

## Performance Benchmarks

| Model          | Accuracy | Balanced Accuracy | Training Duration |
| :------------- | :------- | :---------------- | :---------------- |
| Fine-tuned     | 98.33%   | 91.67%            | ~10 min           |
| Simple CNN     | 93%      | 74%               | ~20 min           |

## Technical Specifications

-   **Python Version**: 3.8 or higher
-   **Core Framework**: PyTorch 2.0 or higher
-   **User Interface**: Streamlit
-   **Audio Processing**: librosa
-   Refer to `requirements.txt` for a comprehensive list of dependencies.

## Recommendations

-   For optimal classification performance, the fine-tuned ResNet18 model is highly recommended.
-   Note that classifications for "Natural Sounds" and "Other Sounds" may exhibit comparatively lower accuracy due to limited training data (13 samples per category).
-   Enhancing the diversity and volume of training data will significantly improve overall system performance.
-   The system is fully compatible with Apple Silicon (MPS) and standard CPU architectures.

## Licensing

This project is released under the MIT License, promoting open use, modification, and distribution.

---

*Engineered with PyTorch, Streamlit, and a commitment to innovation.*
