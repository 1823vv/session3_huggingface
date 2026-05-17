import os
import json
import pickle
import torch
import torch.nn as nn

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

def load_model():
    with open('model/config.json', 'r') as f:
        config = json.load(f)
    
    model = SentimentNN(config['input_dim'], config['hidden_dim'], config['output_dim'])
    model.load_state_dict(torch.load('model/model.pt', map_location=torch.device('cpu')))
    model.eval()
    
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, vectorizer, label_encoder

def predict_sentiment(text, model, vectorizer, label_encoder):
    model.eval()
    with torch.no_grad():
        text_tfidf = vectorizer.transform([text]).toarray()
        output = model(torch.tensor(text_tfidf, dtype=torch.float32))
        prediction = (output >= 0.5).int().item()
        sentiment = label_encoder.inverse_transform([prediction])[0]
        confidence = output.item() if prediction == 1 else 1 - output.item()
    return sentiment, confidence

def main():
    if not os.path.exists('model/model.pt'):
        print("Error: Model not found. Please train the model first by running train.py")
        return
    
    model, vectorizer, label_encoder = load_model()
    
    while True:
        text = input("Enter a movie review (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        
        sentiment, confidence = predict_sentiment(text, model, vectorizer, label_encoder)
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()