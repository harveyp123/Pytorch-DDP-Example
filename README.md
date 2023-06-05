A minimum example for pytorch DDP based single node multi-GPU usage on MNIST dataset. This example includes basic DDP usage, gradient compression, and additional handling. 

## 1. Environment setup
Create a environment with pytorch in it
```sh
conda create --name torchenv python=3.9
conda activate torchenv
conda install -y pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```

## 2. Launching distributed data parallel example

```sh
bash run_mnist_ddp.sh
```
You may change the available gpu with ```CUDA_VISIBLE_DEVICES```. 

The gradient compression can be adjusted through ```--compression powersgd_fp16```, by default it uses ```powersgd_fp16``` compression. You have 5 choices for gradient compression: ```['none', 'fp16', 'powersgd', 'powersgd_fp16', 'batched_powersgd_fp16']```
Details can be found later section. 

## 3. Important explanation of code

### 3.1 SyncBN
The batchnorm in the model will be transformed into synchronous batch norm ([SyncBN](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html)):

```py
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
```
The running mean and variance in the batch norm is not pytorch parameter object and will not be aggregated the same as gradient across multiple process. Without synchronous batch norm, the batch norm will only take effects on the main GPU which is in charge of syncing global data. As such, the effective batch size for calculating the batch norm is the single GPU batch size. 

With SyncBatchNorm, the running mean and variance will be shared among multiple GPUs, and the effective batch size will be (single GPU batch size * number of GPUs)

### 3.2 Gradient compression for communication reduction
We have 4 types of gradient compression. The tutorial can be found in official [Pytorch](https://pytorch.org/docs/stable/ddp_comm_hooks.html) document or [Medium](https://medium.com/pytorch/accelerating-pytorch-ddp-by-10x-with-powersgd-585aef12881d). 

#### Method 1, FP16 gradient compression: 
```py
model.register_comm_hook(state=None, hook=torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook) 
```
The method compressed the gradient into FP16 format for communication, and will be uncompressed during gradient accumulation

#### Method 2, PowerSGD compression: 
```py
state = powerSGD.PowerSGDState(
process_group=None, 
matrix_approximation_rank=2,
start_powerSGD_iter=1_000,
)
model.register_comm_hook(state, powerSGD.powerSGD_hook)
```
Original powewSGD algorithm is proposed in this [Paper](https://arxiv.org/abs/1905.13727). Pytorch wrapped the algorithm for easy usage. PowewSGD is similar to SVD compression but with power iterations and has lower overhead. 

#### Method 3, PowerSGD + FP16 compression: 
```py
state = powerSGD.PowerSGDState(
process_group=None, 
matrix_approximation_rank=2,
start_powerSGD_iter=1_000,
)
model.register_comm_hook(state, torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper(powerSGD.powerSGD_hook))
```
Besides naive powerSGD, you can also wrap the powerSGD with additional FP16 compression to further reduce the communication by half.

#### Method 4, batchedPowerSGD + FP16 compression: 
```py
state = powerSGD.PowerSGDState(
process_group=None, 
matrix_approximation_rank=2,
start_powerSGD_iter=1_000,
)
model.register_comm_hook(state, torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper(powerSGD.batched_powerSGD_hook))  
```
The batched_powerSGD_hook flattenes all gradient tensors and conduct compression together, such method may lead to a lower accuracy. 

## 4 More code details explanation

### 4.1 Accuracy
This defines the function for calculating top-1 and top-5 accuracy
```py
def accuracy(output, target, topk=(1,5)):
```

### 4.2 Model
This defines the function for model
```py
class Net(nn.Module):
```

### 4.3. Logger:
```py
logger = get_logger(os.path.join(args.path, "mnist_train.log"))
```
This set the logger output path, by default ```args.path``` is ```./logging/```

### 4.4 Distributed communication package
This defines the communication backend and protocol. 
```py
torch.distributed.init_process_group(backend="nccl", init_method="tcp://localhost:12345", world_size=nums_gpus, rank=gpu) 
```
Detailed can be found in [Pytorch](https://pytorch.org/docs/stable/distributed.html) official document, which compares difference between ```gloo, mpi, nccl``` communication backend. 

### 4.5 Keyboard interrupt handling
This defines the handler for keyboard interrupt. Upon the keyboard interrupt received, all process will be terminated and joined. It ensures the correct keyboard interrupt behavior, otherwise some process will be dangling on system and requires manual kill. 
```py
try:
    for gpu in range(num_gpus):
        p = Process(target=train, args=(gpu, args, barrier))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

except KeyboardInterrupt:
    print("KeyboardInterrupt received. Terminating processes...")
    for p in processes:
        p.terminate()
        p.join()
```
