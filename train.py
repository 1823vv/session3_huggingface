import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class SentimentNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test, dtype=torch.float32))
        predictions = (outputs >= 0.5).float()
        accuracy = accuracy_score(y_test, predictions.numpy())
        report = classification_report(y_test, predictions.numpy())
    return accuracy, report

def main():
    df = pd.read_csv('data/imdb_balanced_10k.csv')
    
    X = df['review']
    y = df['sentiment']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    
    input_dim = X_train_tfidf.shape[1]
    hidden_dim = 256
    output_dim = 1
    
    model = SentimentNN(input_dim, hidden_dim, output_dim)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TensorDataset(torch.tensor(X_train_tfidf, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("Training model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=15)
    
    print("\nEvaluating model...")
    accuracy, report = evaluate_model(model, X_test_tfidf, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    os.makedirs('model', exist_ok=True)
    
    torch.save(model.state_dict(), 'model/model.pt')
    
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    config = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'max_features': 5000,
        'test_accuracy': accuracy
    }
    with open('model/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }
    with open('model/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nModel artifacts saved to 'model/' directory")
    print("Files saved: model.pt, vectorizer.pkl, label_encoder.pkl, config.json, metrics.json")

if __name__ == "__main__":
    main()