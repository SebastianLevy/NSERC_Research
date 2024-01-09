import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataLoading.datasetLoading import sEMGDataset, ComputeBasicFeatures
from dataLoading.dataProcessing import load_data_from_file, segment_data

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
    # Set the model to evaluation mode
    model.eval()

    # Track the number of correct predictions and the total number of predictions
    correct_predictions = 0
    total_predictions = 0

    # We don't need to track gradients during evaluation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            inputs = sample['data']
            labels = torch.tensor([label_mapping[label] for label in sample['label']])
            
            # Forward pass
            outputs = model(inputs)
            
            # Get the predicted class with the highest score
            _, predicted = outputs.max(1)
            
            # Update the number of correct predictions and the total number of predictions
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    # Compute the accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy

filename = 'data/example.txt'
# Parameters
WINDOW_SIZE = 100  # Example window size
BATCH_SIZE = 32

# Load data
data, labels = load_data_from_file('data.txt')

# Segment data
segmented_data, segmented_labels = segment_data(data, labels, WINDOW_SIZE)

# Instantiate your dataset
dataset = sEMGDataset(segmented_data, segmented_labels, transform=ComputeBasicFeatures())

# Split the dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Instantiate your data loader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


   
# Hyperparameters
input_size = 100  # Input feature dimension, which is the window size
hidden_size = 64  # Number of hidden states in LSTM
num_layers = 2  # Number of LSTM layers
num_classes = 5  # We have 5 types of grasps
learning_rate = 0.001

# Create the model, loss function, and optimizer
model = sEMGLSTM(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training parameters
num_epochs = 50

# Convert labels to tensor and map them to integers for training
label_mapping = {"Cylindrical": 0, "Tip": 1, "Hook": 2, "Palmar": 3, "Lateral": 4}
segmented_labels_tensor = torch.tensor([label_mapping[label] for label in segmented_labels])

# Training loop
for epoch in range(num_epochs):
    for batch_idx, sample in enumerate(dataloader):
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
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
accuracy = evaluate_model(model, test_loader)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")