# 🧠 Bayesian Optimization

Bayesian optimization is an intelligent search strategy that uses past evaluation results to guide the search toward promising hyperparameter regions. Unlike grid or random search, it learns from previous trials.

---

## How Bayesian Optimization Works

### The Core Idea

```
Random/Grid Search:
  Each trial is independent
  No learning from past results
  Blind exploration

Bayesian Optimization:
  Build model of objective function
  Use model to predict promising regions
  Focus search on high-potential areas
  Learn and improve with each trial
```

### The Algorithm

```
1. Initialize with a few random trials
   ↓
2. Build a surrogate model of the objective function
   (predicts performance given hyperparameters)
   ↓
3. Use acquisition function to select next hyperparameters
   (balances exploration vs exploitation)
   ↓
4. Evaluate the selected hyperparameters
   ↓
5. Update the surrogate model with new result
   ↓
6. Repeat steps 3-5 until budget exhausted
```

### Visual Intuition

```
After 5 trials:

Performance
    │      ?           ?
    │   ╱╲    ?    ╱╲
    │  ╱  ╲      ╱    ╲
    │ ╱    ╲    ╱      ╲
    │╱      ╲──╱        ╲
    └─────────────────────► Hyperparameter
      ●    ●  ●    ●   ●
      
● = Evaluated points
? = Uncertain regions (explore here)
╱╲ = Surrogate model prediction

Acquisition function decides: explore uncertain regions
or exploit near known good points?
```

---

## Exploration vs Exploitation

### The Trade-off

```
Exploitation:
  "Try hyperparameters similar to the best found so far"
  + Refines good solutions
  - May miss better regions

Exploration:
  "Try hyperparameters in uncertain regions"
  + Discovers new good regions
  - May waste evaluations

Good acquisition function balances both!
```

### Acquisition Functions

```python
# Common acquisition functions:

# 1. Expected Improvement (EI)
#    "How much improvement can we expect?"
#    Balances mean prediction and uncertainty

# 2. Upper Confidence Bound (UCB)
#    "Optimistic estimate of performance"
#    UCB = mean + κ * std
#    Higher κ = more exploration

# 3. Probability of Improvement (PI)
#    "What's the chance of beating current best?"
#    Can be too greedy (exploitation-heavy)
```

---

## Optuna: Modern Bayesian Optimization

Optuna is a state-of-the-art hyperparameter optimization framework that makes Bayesian optimization easy to use.

### Basic Usage

```python
import optuna

def objective(trial):
    """
    Objective function to minimize/maximize.
    trial object suggests hyperparameters.
    """
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_int('hidden_size', 32, 512)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    
    # Build and train model
    model = build_model(hidden_size, dropout)
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Train
    train(model, optimizer, train_loader)
    
    # Evaluate and return metric
    accuracy = evaluate(model, val_loader)
    return accuracy

# Create study (maximize accuracy)
study = optuna.create_study(direction='maximize')

# Run optimization
study.optimize(objective, n_trials=100)

# Results
print(f"Best trial: {study.best_trial.number}")
print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best hyperparameters: {study.best_params}")
```


### Suggest Methods

Optuna provides various methods to suggest hyperparameters:

```python
def objective(trial):
    # Continuous (float)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Integer
    n_layers = trial.suggest_int('n_layers', 1, 5)
    hidden_size = trial.suggest_int('hidden_size', 32, 512, step=32)
    
    # Categorical
    activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'tanh'])
    
    # Conditional hyperparameters
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    if optimizer == 'sgd':
        momentum = trial.suggest_float('momentum', 0.0, 0.99)
    
    # Build model with suggested hyperparameters
    model = build_model(n_layers, hidden_size, dropout, activation)
    # ... train and evaluate ...
    return accuracy
```

### Log Scale for Learning Rate

```python
# Without log=True (bad for learning rate)
lr = trial.suggest_float('lr', 0.0001, 0.1)
# Most samples will be near 0.1

# With log=True (good for learning rate)
lr = trial.suggest_float('lr', 0.0001, 0.1, log=True)
# Samples uniformly across orders of magnitude
```

---

## Pruning: Early Stopping for Bad Trials

Optuna can stop unpromising trials early, saving significant computation.

### How Pruning Works

```
Trial 1: epoch 1→0.5, epoch 2→0.6, epoch 3→0.7 ... epoch 10→0.85 ✓
Trial 2: epoch 1→0.3, epoch 2→0.35 → PRUNED (clearly worse)
Trial 3: epoch 1→0.55, epoch 2→0.65, epoch 3→0.75 ... epoch 10→0.88 ✓
Trial 4: epoch 1→0.2 → PRUNED (clearly worse)

Pruning saves time by not fully training bad configurations!
```

### Implementation

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    
    # Build model
    model = build_model(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with pruning
    for epoch in range(100):
        train_one_epoch(model, optimizer, train_loader)
        
        # Evaluate
        accuracy = evaluate(model, val_loader)
        
        # Report intermediate value
        trial.report(accuracy, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy

# Create study with pruner
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,    # Don't prune first 5 trials
        n_warmup_steps=10,     # Don't prune before epoch 10
        interval_steps=1       # Check pruning every epoch
    )
)

study.optimize(objective, n_trials=100)
```

### Pruner Types

```python
# MedianPruner: Prune if below median of previous trials
pruner = optuna.pruners.MedianPruner()

# PercentilePruner: Prune if below percentile
pruner = optuna.pruners.PercentilePruner(percentile=25.0)

# HyperbandPruner: Aggressive early pruning (good for deep learning)
pruner = optuna.pruners.HyperbandPruner()

# SuccessiveHalvingPruner: Halves trials at each step
pruner = optuna.pruners.SuccessiveHalvingPruner()
```

---

## Complete PyTorch Example

```python
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def create_model(trial, input_size, output_size):
    """Create model with trial-suggested hyperparameters."""
    n_layers = trial.suggest_int('n_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    layers = []
    in_features = input_size
    
    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 32, 256)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_features = out_features
    
    layers.append(nn.Linear(in_features, output_size))
    return nn.Sequential(*layers)

def objective(trial):
    # Hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model
    model = create_model(trial, input_size=784, output_size=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        momentum = trial.suggest_float('momentum', 0.0, 0.99)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training
    n_epochs = 20
    for epoch in range(n_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                _, predicted = torch.max(output, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total
        
        # Report for pruning
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy

# Run optimization
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=100, timeout=3600)  # 1 hour timeout

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best hyperparameters: {study.best_params}")
```

---

## Visualization

Optuna provides built-in visualization tools:

```python
import optuna.visualization as vis

# Optimization history
fig = vis.plot_optimization_history(study)
fig.show()

# Parameter importance
fig = vis.plot_param_importances(study)
fig.show()

# Parallel coordinate plot
fig = vis.plot_parallel_coordinate(study)
fig.show()

# Contour plot (2 parameters)
fig = vis.plot_contour(study, params=['lr', 'dropout'])
fig.show()

# Slice plot (1 parameter)
fig = vis.plot_slice(study, params=['lr'])
fig.show()

# Hyperparameter relationships
fig = vis.plot_edf(study)  # Empirical distribution function
fig.show()
```

---

## Advanced Features

### Saving and Resuming Studies

```python
# Save to SQLite database
study = optuna.create_study(
    study_name='my_study',
    storage='sqlite:///optuna_study.db',
    direction='maximize',
    load_if_exists=True  # Resume if exists
)

# Run more trials later
study.optimize(objective, n_trials=50)
```

### Multi-Objective Optimization

```python
def objective(trial):
    # ... build and train model ...
    accuracy = evaluate_accuracy(model)
    latency = measure_latency(model)
    
    return accuracy, latency  # Return multiple objectives

study = optuna.create_study(
    directions=['maximize', 'minimize']  # Max accuracy, min latency
)
study.optimize(objective, n_trials=100)

# Get Pareto front
pareto_trials = study.best_trials
```

### Distributed Optimization

```python
# Worker 1
study = optuna.load_study(
    study_name='distributed_study',
    storage='mysql://user:pass@host/db'
)
study.optimize(objective, n_trials=25)

# Worker 2 (same code, runs in parallel)
study = optuna.load_study(
    study_name='distributed_study',
    storage='mysql://user:pass@host/db'
)
study.optimize(objective, n_trials=25)
```

---

## Advantages and Disadvantages

### Advantages

| Advantage | Description |
|-----------|-------------|
| **Efficient** | Learns from past trials, focuses on promising regions |
| **Pruning** | Stops bad trials early, saves computation |
| **Flexible** | Handles continuous, integer, categorical parameters |
| **Conditional** | Supports dependent hyperparameters |
| **Visualization** | Built-in analysis tools |
| **Scalable** | Supports distributed optimization |

### Disadvantages

| Disadvantage | Description |
|--------------|-------------|
| **Overhead** | Surrogate model adds computation |
| **Complexity** | More complex than grid/random search |
| **Sequential** | Core algorithm is sequential (though parallelizable) |
| **May overfit** | Can overfit to validation set with many trials |

---

## When to Use Bayesian Optimization

### Good Use Cases

- **Expensive evaluations** (training takes hours)
- **Limited budget** (can only run 50-100 trials)
- **Complex search spaces** (many hyperparameters)
- **Need best possible performance**

### Consider Alternatives When

- **Very cheap evaluations** (random search may be fine)
- **Highly parallel resources** (random search parallelizes better)
- **Simple problems** (grid search may suffice)

---

## Key Takeaways

1. **Bayesian optimization learns** from past trials to guide search
2. **Optuna** is a modern, easy-to-use framework
3. **Pruning saves computation** by stopping bad trials early
4. **Use log scale** for parameters spanning orders of magnitude
5. **Visualization helps** understand the search process
6. **More efficient than random search** for expensive evaluations
7. **Supports advanced features** like multi-objective and distributed search

---

*Bayesian optimization with Optuna is often the best choice for serious hyperparameter tuning. It's more efficient than random search and provides powerful features like pruning and visualization.*
