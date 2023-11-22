import torch
import time


# torch complie is not supported for python 3.11 yet
# @torch.compile
@torch.jit.script
def matmul_jit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, y)


if torch.cuda.is_available():
    print(torch.__version__)
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available. CPU is used")


matrix_size = 10000

# Calculate on CPU
start_time = time.time()
input_matrix = torch.randn(matrix_size, matrix_size).to("cpu")
result_cpu = torch.matmul(input_matrix, input_matrix)
end_time = time.time()
cpu_time = end_time - start_time

# Calculate on GPU
input_matrix = input_matrix.to(device)
start_time = time.time()
result_gpu = torch.matmul(input_matrix, input_matrix)
end_time = time.time()
gpu_time = end_time - start_time

# Calculate on GPU with Torch compile
# input_matrix = input_matrix.to(device)
# start_time = time.time()
# result_gpu = matmul_compile(input_matrix, input_matrix)
# end_time = time.time()
# gpu_time = end_time - start_time

# Calculate on GPU with jit
input_matrix = input_matrix.to(device)
start_time = time.time()
result_gpu_jit = matmul_jit(input_matrix, input_matrix)
end_time = time.time()
gpu_time_jit = end_time - start_time

print("CPU time: ", cpu_time)
print("GPU time: ", gpu_time)
print("GPU time with jit: ", gpu_time_jit)
print("Speed up w/o jit: ", cpu_time / gpu_time)
print("Speed up with jit: ", cpu_time / gpu_time_jit)
assert torch.allclose(result_cpu[:2, :2], result_gpu[:2, :2].to("cpu"), atol=1e-3)
assert torch.allclose(result_cpu[:2, :2], result_gpu_jit[:2, :2].to("cpu"), atol=1e-3)
