import numpy as np

def fitness_function(server, local_updates, selection_mask, agg_weights, clients_f1, alpha=0.6, beta=0.3, gamma=0.1):
    """
    Evaluates a candidate subset.
    selection_mask: binary array of size K
    agg_weights: continuous array of size K
    """
    K = len(local_updates)
    selected_indices = np.where(selection_mask == 1)[0]
    
    if len(selected_indices) == 0:
        return 999.0 # heavy penalty for no selection
        
    subset_updates = [local_updates[i] for i in selected_indices]
    
    # Softmax normalize the weights of selected clients
    w = agg_weights[selected_indices]
    w = np.exp(w) / np.sum(np.exp(w))
    
    # Aggregate
    agg_model = server.aggregate(subset_updates, w)
    
    # Evaluate global F1
    global_f1, _, _ = server.evaluate(agg_model)
    
    # Evaluate fairness (variance of local F1s for selected clients)
    # We use clients_f1 passed from earlier local evaluation
    if len(selected_indices) > 1:
        f1_var = np.var([clients_f1[i] for i in selected_indices])
    else:
        f1_var = 0.0
        
    cost = len(selected_indices) / K
    
    # Minimize objective
    fitness = alpha * (1.0 - global_f1) + beta * f1_var + gamma * cost
    return fitness

class BaseOptimizer:
    def __init__(self, K):
        self.K = K

class RandomSelector(BaseOptimizer):
    def select(self, selection_ratio=0.5):
        m = max(1, int(self.K * selection_ratio))
        indices = np.random.choice(self.K, m, replace=False)
        mask = np.zeros(self.K)
        mask[indices] = 1
        weights = np.ones(self.K) # Will be normalized anyway
        return mask, weights

class GreedySelector(BaseOptimizer):
    def select(self, clients_f1, selection_ratio=0.5):
        m = max(1, int(self.K * selection_ratio))
        # Select clients with highest F1
        indices = np.argsort(clients_f1)[-m:]
        mask = np.zeros(self.K)
        mask[indices] = 1
        weights = np.ones(self.K)
        return mask, weights

class AllSelector(BaseOptimizer):
    def select(self):
        mask = np.ones(self.K)
        weights = np.ones(self.K)
        return mask, weights

class ClusteredSelector(BaseOptimizer):
    def select(self, clients_f1, num_clusters=3):
        # Sort clients by F1 and group them into tiers (clusters) to ensure representation
        indices = np.argsort(clients_f1)
        clusters = np.array_split(indices, num_clusters)
        
        # Pick representative from each cluster (often maximum or random, we do max here for stability)
        selected = []
        for c in clusters:
            if len(c) > 0:
                selected.append(c[-1]) # highest of that cluster
                
        mask = np.zeros(self.K)
        mask[selected] = 1
        weights = np.ones(self.K)
        return mask, weights


class MetaheuristicOptimizer(BaseOptimizer):
    def __init__(self, K, num_particles=10, max_iter=5):
        super().__init__(K)
        self.num_particles = num_particles
        self.max_iter = max_iter

    def optimize(self, fitness_cb):
        # Default empty implementation
        pass

class PSO(MetaheuristicOptimizer):
    def optimize(self, fitness_cb):
        P = self.num_particles
        # Position: K for mask (sigmoid), K for weights
        pos = np.random.uniform(-2, 2, (P, 2*self.K))
        vel = np.random.uniform(-1, 1, (P, 2*self.K))
        
        pbest = np.copy(pos)
        pbest_fit = np.ones(P) * 999
        gbest = None
        gbest_fit = 999
        
        for k in range(P):
            mask = (1 / (1 + np.exp(-pos[k, :self.K])) > 0.5).astype(int)
            fit = fitness_cb(mask, pos[k, self.K:])
            pbest_fit[k] = fit
            if fit < gbest_fit:
                gbest_fit = fit
                gbest = np.copy(pos[k])
                
        for _ in range(self.max_iter):
            for k in range(P):
                r1, r2 = np.random.rand(2)
                vel[k] = 0.5 * vel[k] + 1.5*r1*(pbest[k] - pos[k]) + 1.5*r2*(gbest - pos[k])
                pos[k] += vel[k]
                
                mask = (1 / (1 + np.exp(-pos[k, :self.K])) > 0.5).astype(int)
                fit = fitness_cb(mask, pos[k, self.K:])
                if fit < pbest_fit[k]:
                    pbest_fit[k] = fit
                    pbest[k] = np.copy(pos[k])
                    if fit < gbest_fit:
                        gbest_fit = fit
                        gbest = np.copy(pos[k])
                        
        mask = (1 / (1 + np.exp(-gbest[:self.K])) > 0.5).astype(int)
        if np.sum(mask) == 0:
            mask[np.random.randint(0, self.K)] = 1
        return mask, gbest[self.K:]

class GWO(MetaheuristicOptimizer):
    def optimize(self, fitness_cb):
        P = self.num_particles
        pos = np.random.uniform(-2, 2, (P, 2*self.K))
        
        alpha_pos, alpha_fit = None, 999
        beta_pos, beta_fit = None, 999
        delta_pos, delta_fit = None, 999
        
        for k in range(P):
            mask = (1 / (1 + np.exp(-pos[k, :self.K])) > 0.5).astype(int)
            fit = fitness_cb(mask, pos[k, self.K:])
            
            if fit < alpha_fit:
                delta_pos, delta_fit = beta_pos, beta_fit
                beta_pos, beta_fit = alpha_pos, alpha_fit
                alpha_pos, alpha_fit = np.copy(pos[k]), fit
            elif fit < beta_fit:
                delta_pos, delta_fit = beta_pos, beta_fit
                beta_pos, beta_fit = np.copy(pos[k]), fit
            elif fit < delta_fit:
                delta_pos, delta_fit = np.copy(pos[k]), fit
                
        for t in range(self.max_iter):
            a = 2.0 - t * (2.0 / self.max_iter)
            for k in range(P):
                r1, r2 = np.random.rand(2), np.random.rand(2)
                A1, C1 = 2*a*r1[0] - a, 2*r2[0]
                X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - pos[k])
                
                r1, r2 = np.random.rand(2), np.random.rand(2)
                A2, C2 = 2*a*r1[0] - a, 2*r2[0]
                X2 = (beta_pos) - A2 * np.abs(C2 * (beta_pos) - pos[k]) if beta_pos is not None else X1
                    
                r1, r2 = np.random.rand(2), np.random.rand(2)
                A3, C3 = 2*a*r1[0] - a, 2*r2[0]
                X3 = (delta_pos) - A3 * np.abs(C3 * (delta_pos) - pos[k]) if delta_pos is not None else X1
                
                pos[k] = (X1 + X2 + X3) / 3.0
                
                mask = (1 / (1 + np.exp(-pos[k, :self.K])) > 0.5).astype(int)
                fit = fitness_cb(mask, pos[k, self.K:])
                
                if fit < alpha_fit:
                    delta_pos, delta_fit = beta_pos, beta_fit
                    beta_pos, beta_fit = alpha_pos, alpha_fit
                    alpha_pos, alpha_fit = np.copy(pos[k]), fit
                elif fit < beta_fit:
                    delta_pos, delta_fit = beta_pos, beta_fit
                    beta_pos, beta_fit = np.copy(pos[k]), fit
                elif fit < delta_fit:
                    delta_pos, delta_fit = np.copy(pos[k]), fit

        mask = (1 / (1 + np.exp(-alpha_pos[:self.K])) > 0.5).astype(int)
        if np.sum(mask) == 0:
            mask[np.random.randint(0, self.K)] = 1
        return mask, alpha_pos[self.K:]

class HybridPSOGWO(MetaheuristicOptimizer):
    def optimize(self, fitness_cb):
        P = self.num_particles
        pos = np.random.uniform(-2, 2, (P, 2*self.K))
        vel = np.random.uniform(-1, 1, (P, 2*self.K))
        
        pbest = np.copy(pos)
        pbest_fit = np.ones(P) * 999
        gbest = None
        gbest_fit = 999
        
        alpha_pos, alpha_fit = None, 999
        beta_pos, beta_fit = None, 999
        delta_pos, delta_fit = None, 999
        
        for k in range(P):
            mask = (1 / (1 + np.exp(-pos[k, :self.K])) > 0.5).astype(int)
            fit = fitness_cb(mask, pos[k, self.K:])
            pbest_fit[k] = fit
            if fit < gbest_fit:
                gbest_fit = fit
                gbest = np.copy(pos[k])
            
            if fit < alpha_fit:
                delta_pos, delta_fit = beta_pos, beta_fit
                beta_pos, beta_fit = alpha_pos, alpha_fit
                alpha_pos, alpha_fit = np.copy(pos[k]), fit
            elif fit < beta_fit:
                delta_pos, delta_fit = beta_pos, beta_fit
                beta_pos, beta_fit = np.copy(pos[k]), fit
            elif fit < delta_fit:
                delta_pos, delta_fit = np.copy(pos[k]), fit
                
        for t in range(self.max_iter):
            a = 2.0 - t * (2.0 / self.max_iter)
            inertia = 0.9 - 0.5 * (t / max(1, self.max_iter - 1))
            
            for k in range(P):
                r1, r2 = np.random.rand(2), np.random.rand(2)
                A1, C1 = 2*a*r1[0] - a, 2*r2[0]
                D_alpha = np.abs(C1 * alpha_pos - pos[k])
                X1 = alpha_pos - A1 * D_alpha
                
                r1, r2 = np.random.rand(2), np.random.rand(2)
                A2, C2 = 2*a*r1[0] - a, 2*r2[0]
                if beta_pos is not None:
                    D_beta = np.abs(C2 * (beta_pos) - pos[k])
                    X2 = (beta_pos) - A2 * D_beta
                else:
                    X2 = X1
                    
                r1, r2 = np.random.rand(2), np.random.rand(2)
                A3, C3 = 2*a*r1[0] - a, 2*r2[0]
                if delta_pos is not None:
                    D_delta = np.abs(C3 * (delta_pos) - pos[k])
                    X3 = (delta_pos) - A3 * D_delta
                else:
                    X3 = X1
                
                # GWO-guided point.
                c1, c2 = 1.0, 1.2
                gwo_pos = (X1 + X2 + X3) / 3.0
                
                # Hybrid PSO velocity based on personal best and GWO leaders.
                vel[k] = (
                    inertia * vel[k]
                    + c1 * np.random.rand() * (pbest[k] - pos[k])
                    + c2 * np.random.rand() * (gwo_pos - pos[k])
                )
                pos[k] += vel[k]

                # Mild mutation to escape local minima.
                if np.random.rand() < 0.08:
                    pos[k] += np.random.normal(0, 0.1, size=2*self.K)
                
                mask = (1 / (1 + np.exp(-pos[k, :self.K])) > 0.5).astype(int)
                fit = fitness_cb(mask, pos[k, self.K:])
                
                if fit < pbest_fit[k]:
                    pbest_fit[k] = fit
                    pbest[k] = np.copy(pos[k])
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = np.copy(pos[k])
                    
                if fit < alpha_fit:
                    delta_pos, delta_fit = beta_pos, beta_fit
                    beta_pos, beta_fit = alpha_pos, alpha_fit
                    alpha_pos, alpha_fit = np.copy(pos[k]), fit
                elif fit < beta_fit:
                    delta_pos, delta_fit = beta_pos, beta_fit
                    beta_pos, beta_fit = np.copy(pos[k]), fit
                elif fit < delta_fit:
                    delta_pos, delta_fit = np.copy(pos[k]), fit

        best = gbest if gbest is not None else alpha_pos
        mask = (1 / (1 + np.exp(-best[:self.K])) > 0.5).astype(int)
        if np.sum(mask) == 0:
            mask[np.random.randint(0, self.K)] = 1
        return mask, best[self.K:]
