import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load Simulink data
data = scipy.io.loadmat('C:\\CM_Projects\\SSA_SM\\src_cm4sl\\SM_PY_import\\Simulink simulation output for PINN training\\Track 1_Base Track\\DSchramm_FrontAxle_simpl.mat')

# Extract variables
alpha_v = data['alpha_v'].squeeze().astype(np.float32)
v = data['v'].squeeze().astype(np.float32)
psi_dot = data['psi_dot'].squeeze().astype(np.float32)
beta_sm = data['beta_sm'].squeeze().astype(np.float32)
delta = data['delta'].squeeze().astype(np.float32)  
l_v = float(np.ravel(data['l_v'])[0])

# Filter valid data
valid_indices = (v < -1e-3) | (v > 1e-3)
alpha_v = alpha_v[valid_indices]
v = v[valid_indices]
psi_dot = psi_dot[valid_indices]
beta_sm = beta_sm[valid_indices]
delta = delta[valid_indices]

# Normalize inputs
v_mean = v.mean()
psi_dot_mean = psi_dot.mean()
alpha_v_mean = alpha_v.mean()
delta_mean = delta.mean()

v_std = v.std()
psi_dot_std = psi_dot.std()
alpha_v_std = alpha_v.std()
delta_std = delta.std()

v_norm = (v - v_mean) / v_std
psi_dot_norm = (psi_dot - psi_dot_mean) / psi_dot_std
alpha_v_norm = (alpha_v - alpha_v_mean) / alpha_v_std
delta_norm = (delta - delta_mean) / delta_std

# l_v_array = np.full_like(v_norm, l_v)

# Prepare inputs and outputs
# X = np.stack([alpha_v_norm, v_norm, psi_dot_norm, delta_norm, l_v_array], axis=1)
X = np.stack([alpha_v_norm, v_norm, psi_dot_norm, delta_norm], axis=1)
y = beta_sm

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
v_tensor = torch.tensor(v, dtype=torch.float32)
psi_dot_tensor = torch.tensor(psi_dot, dtype=torch.float32)
delta_tensor = torch.tensor(delta, dtype=torch.float32)
alpha_v_tensor = torch.tensor(alpha_v, dtype=torch.float32)
# l_v_tensor = torch.tensor(l_v, dtype=torch.float32)

# PINN model
class PINN(nn.Module):
    def __init__(
        self,
        num_inputs: int = 4,
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
        return self.network(x).squeeze()
    
# Xavier initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# Physics loss
'''
def physics_loss(beta_pred, v, psi_dot, delta, l_v):
    alpha_v = delta - beta_pred - l_v * (psi_dot / v)
    return torch.mean(alpha_v ** 2)
'''
def physics_loss(beta_pred, v, psi_dot, delta, l_v, alpha_v):
    beta_th = delta - alpha_v - l_v * (psi_dot / v)
    return torch.mean((beta_pred - beta_th) ** 2)

# Initialize model and optimizer
model = PINN()
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
lambda_phy = 0.1

# Training loop
epochs = 1000
loss_history = []
data_loss_history = []
physics_loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    beta_pred = model(X_tensor)
    loss_data = loss_fn(beta_pred, y_tensor)
    loss_phy = physics_loss(beta_pred, v_tensor, psi_dot_tensor, delta_tensor, l_v, alpha_v)
    loss = loss_data + lambda_phy * loss_phy
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    data_loss_history.append(loss_data.item())
    physics_loss_history.append(loss_phy.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.6f} | Data Loss: {loss_data.item():.6f} | Physics Loss: {loss_phy.item():.6f}")

# Evaluation
model.eval()
with torch.no_grad():
    beta_pred = model(X_tensor).cpu().numpy()
    beta_th = delta - alpha_v - l_v * (psi_dot / v)

# Plot predicted vs measured
plt.figure(figsize=(12, 6))
plt.plot(beta_sm, label='Measured β (beta_sm)', linewidth=2)
plt.plot(beta_pred, label='Predicted β (PINN)', linestyle='--')
plt.plot(beta_th, label='Theoretical β̇ (physics)', linestyle=':')
plt.legend()
plt.title('Comparison of β: Simulink vs. PINN vs. Physics')
plt.xlabel('Sample')
plt.ylabel('β (rad)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Total Loss')
plt.plot(data_loss_history, label='Data Loss')
plt.plot(physics_loss_history, label='Physics Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

alpha_v_tensor = alpha_v_tensor.unsqueeze(1)       
v_tensor = v_tensor.unsqueeze(1)
psi_dot_tensor = psi_dot_tensor.unsqueeze(1)
delta_tensor = delta_tensor.unsqueeze(1)

# l_v_tensor = torch.full((v_tensor.shape[0], 1), l_v, dtype=torch.float32)

# x = torch.cat([v_tensor, psi_dot_tensor, alpha_v_tensor, delta_tensor, l_v_tensor], dim=1)
x = torch.cat([v_tensor, psi_dot_tensor, alpha_v_tensor, delta_tensor], dim=1)
# x = X_tensor[:1]

torch.onnx.export(
    model,
    x,
    "Schramm_FrontAxle_simpl.onnx",
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

from scipy.io import savemat
savemat("const_DSch_FA_sim.mat", {
    "v_mean": v_mean, "v_std": v_std,
    "psi_mean": psi_dot_mean, "psi_std": psi_dot_std,
    "alphav_mean": alpha_v_mean, "alphav_std": alpha_v_std,
    "delta_mean": delta_mean, "delta_std": delta_std
})
traced = torch.jit.trace(model, x)
traced.save("Schramm_FrontAxle_simpl.pt")
print("Model exported successfully.")
print(next(model.parameters()).shape)