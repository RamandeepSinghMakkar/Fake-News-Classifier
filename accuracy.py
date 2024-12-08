
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# Define the PyTorch model (ensure this is defined in your script)
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

# Load the model
model = FakeNewsModel(input_dim=5000)  # Match the input_dim used during training
model.load_state_dict(torch.load('/Users/raman/Desktop/Projects/Fake News Classifier/fake_news_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load the vectorizer
vectorizer = joblib.load('/Users/raman/Desktop/Projects/Fake News Classifier/vectorizer.pkl')

# Load the test data (example, you need to provide the correct path)
import pandas as pd

# Assuming your test data is in a CSV file
test_data = pd.read_csv('/Users/raman/Desktop/Projects/Fake News Classifier/train.csv')  # Replace with your test data file
test_data = test_data.dropna(subset=['text'])
# Assuming your test data has a 'text' column for the news articles and 'label' for the true labels
texts = test_data['text'].values
labels = test_data['label'].values
# Drop rows where 'news_text' is NaN




# Transform the text data using the vectorizer
test_transformed = vectorizer.transform(texts).toarray()

# Convert the transformed data to a tensor
test_tensor = torch.tensor(test_transformed, dtype=torch.float32)

# Make predictions with the model
with torch.no_grad():  # No need to calculate gradients during inference
    predictions = model(test_tensor).squeeze()

# Convert predictions to binary (real or fake)
predicted_labels = (predictions > 0.5).numpy()  # Assuming 0.5 as the threshold for binary classification

# Calculate accuracy
accuracy = accuracy_score(labels, predicted_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')
