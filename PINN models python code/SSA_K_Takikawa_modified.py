import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load Simulink data
data = scipy.io.loadmat('C:\\CM_Projects\\SSA_SM\\src_cm4sl\\SM_PY_import\\Simulink simulation output for PINN training\\Track 1_Base Track\\KTakikawa_modified.mat')

# Extract variables
v = data['v'].squeeze().astype(np.float32)
psi_dot = data['psi_dot'].squeeze().astype(np.float32)
K_r = data['K_r'].squeeze().astype(np.float32)
beta_sm = data['beta_sm'].squeeze().astype(np.float32)

# Extract constants
m = float(np.ravel(data['m'])[0])
L_f = float(np.ravel(data['L_f'])[0])
L_r = float(np.ravel(data['L_r'])[0])
L = L_f + (-L_r)

# Filter valid data
valid_indices = (v < -1e-3) | (v > 1e-3)
v = v[valid_indices]
psi_dot = psi_dot[valid_indices]
K_r = K_r[valid_indices]
beta_sm = beta_sm[valid_indices]

# Normalize data
v_mean = v.mean()
psi_dot_mean = psi_dot.mean()
K_r_mean = K_r.mean()

v_std = v.std()
psi_dot_std = psi_dot.std()
K_r_std = K_r.std()

v_norm = (v - v_mean) / v_std
psi_dot_norm = (psi_dot - psi_dot_mean) / psi_dot_std
K_r_norm = (K_r - K_r_mean) / K_r_std

# Prepare inputs and outputs
X = np.stack([v_norm, psi_dot_norm, K_r_norm], axis=1)
y = beta_sm

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
v_tensor = torch.tensor(v, dtype=torch.float32)  # Unnormalized
psi_dot_tensor = torch.tensor(psi_dot, dtype=torch.float32)  # Unnormalized
K_r_tensor = torch.tensor(K_r, dtype=torch.float32)  # Unnormalized

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
        return self.network(x).squeeze()

# Xavier initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# Physics loss
def physics_loss(pred, v, psi_dot, K_r):
    v_safe = torch.sign(v) * torch.clamp(torch.abs(v), min=1e-3)
    beta_th = (psi_dot * ((-L_r / v_safe) - (m * L_f / (2 * L * K_r)) * v_safe))
    return torch.mean((pred - beta_th) ** 2)

# Initialize model and optimizer
model = PINN()
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
lambda_phy = 0.1  # Adjusted for better balance

# Training
epochs = 1000
loss_history = []
data_loss_history = []
physics_loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(X_tensor)
    loss_data = loss_fn(pred, y_tensor)
    loss_phy = physics_loss(pred, v_tensor, psi_dot_tensor, K_r_tensor)
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
    beta_pred = model(X_tensor).numpy()
#    beta_th = psi_dot * ((L_r / (v + 1e-6)) - (m * L_f / (2 * L * K_r)) * v)
    v_safe = np.sign(v) * np.clip(np.abs(v), a_min=1e-3, a_max=None)
    beta_th = (psi_dot * ((-L_r / v_safe) - (m * L_f / (2 * L * K_r)) * v_safe))


# Plot results
plt.figure(figsize=(12, 6))
plt.plot(beta_sm, label='Measured β (beta_sm)', linewidth=2)
plt.plot(beta_pred, label='Predicted β (PINN)', linestyle='--')
plt.plot(beta_th, label='Theoretical β (Physics)', linestyle=':')
plt.legend()
plt.title('Comparison of β: Simulink vs. PINN vs. Physics (K Takikawa)')
plt.xlabel('Sample')
plt.ylabel('β (rad)')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Total Loss')
plt.plot(data_loss_history, label='Data Loss')
plt.plot(physics_loss_history, label='Physics Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves: K Takikawa')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

K_r_tensor = K_r_tensor.unsqueeze(1)       # shape: [batch_size, 1]
v_tensor = v_tensor.unsqueeze(1)
psi_dot_tensor = psi_dot_tensor.unsqueeze(1)

x = torch.cat([v_tensor, psi_dot_tensor, K_r_tensor], dim=1)

torch.onnx.export(
    model,
    x,
    "K_Takikawa_modified.onnx",
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
savemat("const_KT_mod.mat", {
    "v_mean": v_mean, "v_std": v_std,
    "psi_mean": psi_dot_mean, "psi_std": psi_dot_std,
    "Kr_mean": K_r_mean, "Kr_std": K_r_std
})
traced = torch.jit.trace(model, x)
traced.save("K_Takikawa_modified.pt")
print("Model exported successfully.")