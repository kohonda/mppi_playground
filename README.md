# MPPI Playground

A PyTorch implementation of [Model Predictive Path Integral Control (MPPI)](https://arxiv.org/abs/1707.02342).

**Key Features:**

- 🚀 GPU-accelerated parallel computation
- 🧠 Differentiable implementation in PyTorch
- ⚙️ Flexible definitions of dynamics and cost functions
- 🔧 Additional useful features, including automatic temperature tuning and Savitzky–Golay trajectory smoothing
- 🤖 Example applications

## Tested Environment

- Ubuntu 20.04 or higher
- [Optional] GPU with NVIDIA Driver (510 or later, tested with 550)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Run Examples

To run the example applications, install with additional dependencies:

```bash
# Clone repository
git clone https://github.com/kohonda/mppi_playground.git
cd mppi_playground
uv sync --extra example # Install with extra dependencies for examples
# uv run pre-commit install  # (optional) install pre-commit hooks for code formatting
# uv run pre-commit run --all-files  # (optional) format code
```

### Navigation 2D

```bash
uv run python example/navigation2d.py
```

<p align="center">
  <img src="./media/navigation_2d.gif" width="500" alt="navigation2d">
</p>

### Racing

```bash
uv run python example/racing.py
```

<p align="center">
  <img src="./media/racing.gif" width="500" alt="racing">
</p>

[circuit course information (Japan Automotive AI Challenge 2024)](https://github.com/AutomotiveAIChallenge/aichallenge-2024)

### Pendulum

```bash
uv run python example/pendulum.py
```

<p align="center">
  <img src="./media/pendulum.gif" width="500" alt="pendulum">
</p>

### Cartpole

```bash
uv run python example/cartpole.py
```

<p align="center">
  <img src="./media/cartpole.gif" width="500" alt="cartpole">
</p>

### Mountain car

```bash
uv run python example/mountaincar.py
```

<p align="center">
  <img src="./media/mountaincar.gif" width="500" alt="mountaincar">
</p>

<!-- ## Docker

<details>
<summary>Docker Setup</summary>

### Install Docker

[Installation guide](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

```bash
# Install from get.docker.com
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo groupadd docker
sudo usermod -aG docker $USER
```

### Setup GPU for Docker

[Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker
```

</details>

### Build and Run

```bash
# Build container with GPU support
make build-gpu

# Open remote container via VSCode (Recommended)
# 1. Open the folder using VSCode
# 2. Ctrl+P and select 'devcontainer rebuild and reopen in container'

# Or run container via terminal
make bash-gpu
```  -->

### Usage in Your Project

Add dependency to your `pyproject.toml` and `uv sync`:

```toml
dependencies = [
    "pi-mpc"
]

[tool.uv.sources]
pi-mpc = {git = "https://github.com/kohonda/mppi_playground.git"}
```

Or install via pip:

```bash
pip install git+https://github.com/kohonda/mppi_playground
```

<details>

<summary>Minimal code example</summary>

```python
import torch
from pi_mpc.mppi import MPPI

# Define dynamics function (batch processing: state [N, dim_state], action [N, dim_control])
def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    # Example: simple integrator
    next_state = state + action
    return next_state

# Define cost function (batch processing)
def cost_func(state: torch.Tensor, action: torch.Tensor, info: dict) -> torch.Tensor:
    # Example: distance to goal
    goal = torch.tensor([1.0, 1.0], device=state.device)
    return torch.sum((state - goal) ** 2, dim=1)

# Create MPPI controller
controller = MPPI(
    horizon=50,
    num_samples=1000,
    dim_state=2,
    dim_control=2,
    dynamics=dynamics,
    cost_func=cost_func,
    u_min=torch.tensor([-1.0, -1.0]),
    u_max=torch.tensor([1.0, 1.0]),
    sigmas=torch.tensor([0.5, 0.5]),
    lambda_=1.0,
)

# Solve for optimal action sequence
current_state = torch.tensor([0.0, 0.0])
action_seq, state_seq = controller(current_state)
print(f"Optimal first action: {action_seq[0]}")
```
</details>

<details>

<summary>Configurations</summary>

### MPPI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `horizon` | int | required | Prediction horizon length |
| `num_samples` | int | required | Number of trajectory samples |
| `dim_state` | int | required | State dimension |
| `dim_control` | int | required | Control dimension |
| `dynamics` | Callable | required | Dynamics model: `(state, action) -> next_state` |
| `cost_func` | Callable | required | Cost function: `(state, action, info) -> cost` |
| `u_min` | Tensor | required | Minimum control bounds, shape `(dim_control,)` |
| `u_max` | Tensor | required | Maximum control bounds, shape `(dim_control,)` |
| `sigmas` | Tensor | required | Noise std for each control dim, shape `(dim_control,)` |
| `lambda_` | float / str | required | Temperature or auto-tuning method (`"MPO"`, `"LBPS"`, `"ESSPS"`) |

### Auto-Lambda Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lbps_delta` | float | 0.01 | Confidence parameter for LBPS (0 < δ < 1) |
| `essps_target_ess` | float | num_samples/10 | Target effective sample size for ESSPS |
| `lambda_min` | float | 0.01 | Minimum lambda bound for optimization |
| `lambda_max` | float | 10.0 | Maximum lambda bound for optimization |

### Sampling & Filtering Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exploration` | float | 0.0 | Fraction of purely random samples (0 to 1) |
| `use_sg_filter` | bool | False | Enable Savitzky-Golay smoothing |
| `sg_window_size` | int | 5 | Window size for SG filter (must be odd) |
| `sg_poly_order` | int | 3 | Polynomial order for SG filter |


### Automatic Temperature Tuning (Auto-Lambda)

The temperature parameter λ is a hyperparameter that controls the exploration-exploitation trade-off. Three auto-tuning methods are available:


-  `MPO`: Gradient-based optimization: https://arxiv.org/abs/1806.06920
-  `LBPS`: Lower-Bound Policy Search: https://openreview.net/forum?id=HbGgF93Ppoy
- `ESSPS` : Effective Sample Size Policy Search: https://openreview.net/forum?id=HbGgF93Ppoy

```python
# Example: Enable auto-lambda with LBPS
controller = MPPI(..., lambda_="LBPS", lbps_delta=0.01)
```

### Trajectory Smoothing

- **Savitzky-Golay Filter**: Optional smoothing to reduce control jitter (https://arxiv.org/abs/1707.02342)

```python
# Example: Enable smoothing
controller = MPPI(..., use_sg_filter=True, sg_window_size=5, sg_poly_order=3)
```
</details>

## Citation

If you use this repo in your work, please cite this paper (https://arxiv.org/abs/2511.08019) as:

```
@article{honda2025model,
  title={Model Predictive Control via Probabilistic Inference: A Tutorial},
  author={Honda, Kohei},
  journal={arXiv preprint arXiv:2511.08019},
  year={2025}
}
```
