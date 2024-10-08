import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TaskPredictor(nn.Module):
    def __init__(self, input_size):
        super(TaskPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

def prepare_data():
    # Load data from MongoDB using pandas
    df = pd.read_csv('task_history.csv')  # You'll need to export MongoDB data
    
    features = ['category', 'estimatedHours']
    X = pd.get_dummies(df[features], columns=['category'])
    y = df['actualHours']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return torch.FloatTensor(X_scaled), torch.FloatTensor(y.values)

def train_model():
    X, y = prepare_data()
    model = TaskPredictor(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        outputs = model(X)
        loss = criterion(outputs, y.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), 'task_predictor.pth')

if __name__ == "__main__":
    train_model()