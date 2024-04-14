import torch
import torchvision
import numpy as np
from data import labels, label_to_num, get_video_label, files

# Determine the number of classes (e.g., 600)
num_classes = len(np.unique(list(labels.values())))

class VideoClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(VideoClassifier, self).__init__()
        self.cnn = torchvision.models.video.r3d_18(weights="DEFAULT")
        self.cnn.fc = torch.nn.Linear(512, num_classes)  # Update number of classes
        
    def forward(self, x):
        return self.cnn(x)
    
model = VideoClassifier(num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, X, y, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for i in range(len(X)):
            # Ensure input tensor is in the correct format and type
            x = X[i].unsqueeze(0).float() / 255.0  # Normalize and adjust dimensions
            label = torch.tensor([label_to_num[y[i]]], dtype=torch.long)  # Ensure labels are torch.long type
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(X)}, Loss: {loss.item()}", end='\r')
            
X = files[:100]
y = [get_video_label(file) for file in X]
# Assuming X and y are properly defined elsewhere
train(model, X, y, criterion, optimizer, epochs=3)