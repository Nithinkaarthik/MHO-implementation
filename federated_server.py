import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import copy
import numpy as np

class FLServer:
    def __init__(self, model_class, input_dim, global_X, global_y, device=None):
        self.device = device or torch.device("cpu")
        self.global_model = model_class(input_dim).to(self.device)
        self.global_X = torch.tensor(global_X, dtype=torch.float32).to(self.device)
        self.global_y = torch.tensor(global_y, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        num_pos = self.global_y.sum()
        num_neg = len(self.global_y) - num_pos
        pos_w = (num_neg / num_pos) if num_pos > 0 else torch.tensor(1.0)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32, device=self.device))
        
    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())
        
    def set_global_weights(self, weights):
        self.global_model.load_state_dict(weights)
        
    def aggregate(self, local_weights, aggregation_weights):
        """
        Aggregates local model updates. 
        local_weights: list of state_dicts
        aggregation_weights: array of size len(local_weights) summing to 1.
        """
        agg_weights = copy.deepcopy(local_weights[0])
        for k in agg_weights.keys():
            agg_weights[k] = torch.zeros_like(agg_weights[k], dtype=torch.float32)
            for i, w in enumerate(local_weights):
                agg_weights[k] += w[k] * aggregation_weights[i]
                
        return agg_weights
        
    def evaluate(self, weights=None):
        if weights is not None:
            self.global_model.load_state_dict(weights)
            
        self.global_model.eval()
        with torch.no_grad():
            outputs = self.global_model(self.global_X)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            y_np = self.global_y.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            
            acc = accuracy_score(y_np, preds_np)
            f1 = f1_score(y_np, preds_np, zero_division=0)
            
            try:
                auc = roc_auc_score(y_np, probs_np)
            except ValueError:
                auc = 0.5
                
        return f1, acc, auc
