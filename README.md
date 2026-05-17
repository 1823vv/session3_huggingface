# IMDB Sentiment Analysis Neural Network

A machine learning project for sentiment analysis on IMDB movie reviews using TF-IDF + Feedforward Neural Network.

## Project Structure

```
├── data/
│   └── imdb_balanced_10k.csv    # Training dataset
├── model/
│   ├── model.pt                  # Trained model weights
│   ├── vectorizer.pkl            # TF-IDF vectorizer
│   ├── label_encoder.pkl         # Label encoder
│   ├── config.json               # Model configuration
│   └── metrics.json              # Evaluation metrics
├── train.py                      # Training script
├── predict.py                    # Inference script
├── requirements.txt              # Dependencies
└── .github/
    └── workflows/
        └── train-and-upload.yml  # CI/CD workflow
```

## Features

- **TF-IDF Text Vectorization**: Convert text reviews into numerical features
- **Feedforward Neural Network**: Deep learning model for sentiment classification
- **CI/CD Pipeline**: Automated training and deployment to Hugging Face Hub

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

This will:
1. Load and preprocess the IMDB dataset
2. Train the neural network model
3. Evaluate on test set
4. Save model artifacts to `model/` directory

## Prediction

```bash
python predict.py
```

## Model Architecture

```
Input (TF-IDF features) → Hidden Layer (256 units, ReLU) → 
Dropout (0.5) → Hidden Layer (128 units, ReLU) → 
Output (Sigmoid for binary classification)
```

## CI/CD Workflow

On push to `main` branch:
1. Checkout code
2. Install dependencies
3. Train model
4. Upload artifacts to Hugging Face Hub

## Requirements

- Python 3.11+
- PyTorch
- scikit-learn
- pandas
- huggingface-hub

## License

MIT