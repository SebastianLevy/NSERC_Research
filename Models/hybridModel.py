import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataLoading.datasetLoading import sEMGDataset, ComputeBasicFeatures
from dataLoading.dataProcessing import load_data_from_file, segment_data
import datetime

class sEMGLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(sEMGLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Forward propagate the LSTM
        out, _ = self.lstm(x)
        
        # Only take the output from the final timetep
        out = self.fc(out[:, -1, :])
        return out

def evaluate_model(model, test_loader):
    # Evaluation loop
    with torch.no_grad():
        correct = 0
        total = 0
        for sample in test_loader:
            inputs = sample['data']
            labels = torch.tensor([label_mapping[label] for label in sample['label']])
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Set the random seed for reproducibility
torch.manual_seed(0)

# Define the hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# Load the data
data, labels = load_data_from_file('data.csv')
segments, segment_labels = segment_data(data, labels, window_size=100, overlap=0.5)
train_dataset = sEMGDataset(segments[:800], segment_labels[:800], transform=ComputeBasicFeatures())
test_dataset = sEMGDataset(segments[800:], segment_labels[800:], transform=ComputeBasicFeatures())

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the label mapping
label_mapping = {"Cylindrical": 0, "Tip": 1, "Hook": 2, "Palmar": 3, "Lateral": 4}

# Create the model, loss function, and optimizer
model = sEMGLSTM(input_size=8, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, sample in enumerate(train_loader):
        inputs = sample['data']
        labels = torch.tensor([label_mapping[label] for label in sample['label']])
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

# Evaluate the model
accuracy = evaluate_model(model, test_loader)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# Save the model with date and time
now = datetime.datetime.now()
model_name = f"lstmModel_{now.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
torch.save(model.state_dict(), model_name)
print(f"Model saved as {model_name}.")