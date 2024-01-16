import torch
import subprocess
import time

def get_cuda_version():
    return torch.version.cuda

def get_gpu_memory_usage(device_id=0):
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    gpu_memory = [int(x) for x in gpu_info.strip().decode('utf-8').split('\n')]
    return gpu_memory[device_id]

def display_gpu_info():
    print("********** GPU Information **********")
    print("CUDA Version:", get_cuda_version())
    
    if torch.cuda.is_available():
        print("GPU is available.")
        print("Number of GPU devices:", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            print("\nGPU Device", i)
            print("\tDevice Name:", torch.cuda.get_device_name(i))
            print("\tMemory Usage:", get_gpu_memory_usage(i), "MB")

        print("\nIs GPU supporting CUDNN:", torch.backends.cudnn.enabled)
    else:
        print("GPU is not available. :(")

if __name__ == "__main__":
    display_gpu_info()