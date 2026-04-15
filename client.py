import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import f1_score

class FLClient:
    def __init__(self, client_id, name, X_train, y_train, X_val, y_val, model_class, input_dim, device=None):
        self.client_id = client_id
        self.name = name
        self.device = device or torch.device("cpu")
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        self.X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        self.model = model_class(input_dim).to(self.device)
        
        num_pos = self.y_train.sum()
        num_neg = len(self.y_train) - num_pos
        pos_w = (num_neg / num_pos) if num_pos > 0 else torch.tensor(1.0)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32, device=self.device))
        
    def set_weights(self, global_weights):
        self.model.load_state_dict(copy.deepcopy(global_weights))
        
    def train(self, epochs=2, lr=0.01):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(epochs):
            for X_b, y_b in loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_b)
                loss = self.criterion(outputs, y_b)
                loss.backward()
                optimizer.step()
            
        return copy.deepcopy(self.model.state_dict())
        
    def evaluate(self, weights=None):
        if weights is not None:
            self.model.load_state_dict(weights)
            
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_val)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            f1 = f1_score(self.y_val.detach().cpu().numpy(), preds.detach().cpu().numpy(), zero_division=0)
            
        return f1
