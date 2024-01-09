import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataProcessing import load_data_from_file, segment_data


class sEMGDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (List or Tensor): Your sEMG data.
            labels (List or Tensor): Corresponding labels for each data point.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class ComputeBasicFeatures:
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        
        # Compute features
        mean = data.mean()
        variance = data.var()
        
        # Concatenate features to the original data (or you can replace)
        features = [mean, variance]
        combined_data = torch.cat([data, torch.tensor(features)])
        
        return {'data': combined_data, 'label': label}
    

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

# Instantiate your data loader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
