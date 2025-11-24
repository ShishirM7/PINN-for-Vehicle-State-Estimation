import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load Simulink data
data = scipy.io.loadmat('C:\\CM_Projects\\SSA_SM\\src_cm4sl\\SM_PY_import\\Simulink simulation output for PINN training\\Track 1_Base Track\\GRill.mat')

# Extract variables
beta_sm = data['beta_sm'].squeeze().astype(np.float32)
delta = data['delta'].squeeze().astype(np.float32)

# Extract constants
l_v = float(np.ravel(data['l_v'])[0])
l = float(np.ravel(data['l'])[0])

# Normalize data
delta_mean = delta.mean()

delta_std = delta.std()

delta_norm = (delta - delta_mean) / delta_std


# Prepare inputs and outputs
N = delta.shape[0]
l_v_array = np.full_like(delta, l_v)
l_array = np.full_like(delta, l)
X = np.stack([delta_norm, l_v_array, l_array], axis=1)
y = beta_sm

# Convert to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
delta_tensor = torch.tensor(delta, dtype=torch.float32).unsqueeze(1).to(device)

# PINN model
class PINN(nn.Module):
    def __init__(
        self,
        num_inputs: int = 3,
        num_layers: int = 3,
        num_neurons: int = 64,
        act: nn.Module = nn.Tanh(),
        dropout_prob: float = 0.1
        ) -> None:
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.num_layers = num_layers

        layers = []

        # input layer
        layers.append(nn.Linear(self.num_inputs, num_neurons))

        # hidden layers with linear layer and activation
        for _ in range(num_layers):
            layers.extend([nn.Linear(num_neurons, num_neurons), act])

        # output layer
        layers.append(nn.Linear(num_neurons, 1))

        # build the network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Xavier initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        
# Physics loss
def physics_loss(beta_pred, delta, l_v, l):
    eps = 1e-6
    tan_delta = torch.tan(delta.clamp(-np.pi/2 + eps, np.pi/2 - eps))
    rhs = (l_v / l) * tan_delta
    beta_th = torch.atan(rhs)
    return torch.mean((beta_pred - beta_th) ** 2)
  
# Initialize model and optimizer
model = PINN(num_inputs=3).to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
loss_fn = nn.MSELoss()
lambda_phy = 0.1

# Training
epochs = 1000
loss_history = []
data_loss_history = []
physics_loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    beta_pred = model(X_tensor)
    loss_data = loss_fn(beta_pred, y_tensor)
    loss_phy = physics_loss(beta_pred, delta_tensor, l_v, l)
    loss = loss_data + lambda_phy * loss_phy
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())
    data_loss_history.append(loss_data.item())
    physics_loss_history.append(loss_phy.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.6f} | Data Loss: {loss_data.item():.6f} | Physics Loss: {loss_phy.item():.6f}")

# Evaluation
model.eval()
with torch.no_grad():
    beta_pred = model(X_tensor).cpu().numpy()
    eps = 1e-6
    delta_eval_tensor = torch.tensor(delta, dtype=torch.float32)
    tan_delta = torch.tan(delta_eval_tensor.clamp(-np.pi/2 + eps, np.pi/2 - eps))
    rhs = (l_v / l) * tan_delta
    beta_th = torch.atan(rhs).numpy()


# Plot results
plt.figure(figsize=(12, 6))
plt.plot(beta_sm, label='Measured β', linewidth=2)
plt.plot(beta_pred, label='Predicted β (PINN)', linestyle='--')
plt.plot(beta_th, label='Theoretical β̇ (physics)', linestyle=':')
plt.title('β: Simulink vs. PINN vs. Physics')
plt.xlabel('Sample')
plt.ylabel('β (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Total Loss')
plt.plot(data_loss_history, label='Data Loss')
plt.plot(physics_loss_history, label='Physics Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# delta_tensor = delta_tensor.unsqueeze(1)

x = torch.cat([delta_tensor, torch.from_numpy(l_v_array).float().unsqueeze(1), 
    torch.from_numpy(l_array).float().unsqueeze(1)], dim=1)

torch.onnx.export(
    model,
    x,
    "G_Rill.onnx",
    input_names=['input'],          
    output_names=['beta_pred'],     
    dynamic_axes={
        'input': {0: 'batch_size'},
        'beta_pred': {0: 'batch_size'}
    },
    opset_version=17,              
    do_constant_folding=True,       
    export_params=True,             
#    dynamic_axes=None,            
)


traced = torch.jit.trace(model, x)
traced.save("G_Rill.pt")
print("Model exported successfully.")