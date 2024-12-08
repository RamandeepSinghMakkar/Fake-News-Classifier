import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader

# Load the dataset
data = pd.read_csv("train.csv")
data = data.dropna()
data.reset_index(inplace=True)

X = data['title']
y = data['label']

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_transformed = vectorizer.fit_transform(X).toarray()

# Save the vectorizer using joblib
joblib.dump(vectorizer, 'vectorizer.pkl')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.33, random_state=42)

# PyTorch Dataset class
class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model architecture
class FakeNewsModel(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
model = FakeNewsModel(input_dim)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, loss_fn, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

train_model(model, train_loader, loss_fn, optimizer)

# Save the trained model
torch.save(model.state_dict(), '/Users/raman/Desktop/Projects/Fake News Classifier/fake_news_model.pth')
