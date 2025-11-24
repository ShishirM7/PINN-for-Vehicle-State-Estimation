import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.onnx

# Load Simulink data
data = scipy.io.loadmat('C:\\CM_Projects\\SSA_SM\\src_cm4sl\\SM_PY_import\\Simulink simulation output for PINN training\\Track 1_Base Track\\DSchramm_simpl.mat')

# Extract variables
a_y = data['a_y'].squeeze().astype(np.float32)
v = data['v'].squeeze().astype(np.float32)
psi_dot = data['psi_dot'].squeeze().astype(np.float32)
beta_sm = data['beta_sm'].squeeze().astype(np.float32)

'''
fs = 100  # sampling frequency in Hz
start_index = int(10 * fs)

# Trim first 5 seconds of data
a_y = a_y[start_index:]
v = v[start_index:]
psi_dot = psi_dot[start_index:]
beta_sm = beta_sm[start_index:]
'''

# Filter valid data
valid_indices = (v < -1e-3) | (v > 1e-3)
a_y = a_y[valid_indices]
v = v[valid_indices]
psi_dot = psi_dot[valid_indices]
beta_sm = beta_sm[valid_indices]

# Normalize data
v_mean = v.mean()
psi_dot_mean = psi_dot.mean()
a_y_mean = a_y.mean()

v_std = v.std()
psi_dot_std = psi_dot.std()
a_y_std = a_y.std()

v_norm = (v - v_mean) / v_std
psi_dot_norm = (psi_dot - psi_dot_mean) / psi_dot_std
a_y_norm = (a_y - a_y_mean) / a_y_std

# Prepare inputs and outputs
X = np.stack([a_y_norm, v_norm, psi_dot_norm], axis=1)
y = beta_sm

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
a_y_tensor = torch.tensor(a_y, dtype=torch.float32)
v_tensor = torch.tensor(v, dtype=torch.float32)
psi_dot_tensor = torch.tensor(psi_dot, dtype=torch.float32)

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
'''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze() 
'''
    
# Xavier initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# Physics loss function
def physics_loss(pred, a_y, v, psi_dot):
    beta_th = (a_y / v) - psi_dot
    return torch.mean((pred - beta_th) ** 2)

# Initialize model and optimizer
# model = PINN(num_inputs=3, num_layers=3, num_neurons=64, act=nn.Tanh())
model = PINN()
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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

    pred = model(X_tensor)
    loss_data = loss_fn(pred, y_tensor)
    loss_phy = physics_loss(pred, a_y_tensor, v_tensor, psi_dot_tensor)
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
    beta_th = (a_y / v) - psi_dot 

# Plot predicted vs measured vs theoretical
plt.figure(figsize=(12, 6))
plt.plot(beta_sm, label='Measured β̇ (beta_sm)', linewidth=2)
plt.plot(beta_pred, label='Predicted β̇ (PINN)', linestyle='--')
plt.plot(beta_th, label='Theoretical β̇ (physics)', linestyle=':')
plt.legend()
plt.title('Comparison of β̇: Simulink vs. PINN vs. Physics')
plt.xlabel('Sample')
plt.ylabel('β̇ (rad/s)')
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


# scipy.io.savemat('beta_preds.mat', {'beta_pred': beta_pred})

'''
torch.onnx.export(
    model,
    X_tensor,
    "model.onnx",
    export_params=True,
    opset_version=11,  # or 12, check MATLAB compatibility
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
'''

PINN_input = X_tensor
a_y_tensor = a_y_tensor.unsqueeze(1)       
v_tensor = v_tensor.unsqueeze(1)
psi_dot_tensor = psi_dot_tensor.unsqueeze(1)

x = torch.cat([a_y_tensor, v_tensor, psi_dot_tensor], dim=1)

# Export to ONNX
'''
torch.onnx.export(
    model,
    x,
    "pinn_model.onnx",
    input_names=['input'],
    output_names=['beta_pred'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'beta_pred': {0: 'batch_size'}
    },
    opset_version=11,
    verbose=True
)
'''

torch.onnx.export(
    model,
    x,
    "D_Schramm_simpl.onnx",
    input_names=['input'],          
    output_names=['beta_pred'],     
    dynamic_axes={
        'input': {0: 'batch_size'},
        'beta_pred': {0: 'batch_size'}
    },
    opset_version=17,               
    do_constant_folding=False,       
    export_params=True,             
#    dynamic_axes=None,              
)

from scipy.io import savemat
savemat("const_DSch_sim.mat", {
    "v_mean": v_mean, "v_std": v_std,
    "psi_mean": psi_dot_mean, "psi_std": psi_dot_std,
    "ay_mean": a_y_mean, "ay_std": a_y_std
})
traced = torch.jit.trace(model, x)
traced.save("D_Schramm_simpl.pt")
print("Model exported successfully.")