import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD

def accuracy(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output



#### Refer to https://pytorch.org/docs/stable/_modules/torch/distributed/algorithms/ddp_comm_hooks/default_hooks.html#fp16_compress_hook

def fp16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    compressed_tensor = bucket.buffer().to(torch.float16).div_(world_size)

    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.buffer()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        decompressed_tensor.copy_(fut.value()[0])
        return decompressed_tensor

    return fut.then(decompress)

def train(gpu, args):
    torch.cuda.set_device(gpu)
    nums_gpus = torch.cuda.device_count()
    torch.distributed.init_process_group(backend="nccl", init_method="tcp://localhost:12345", world_size=nums_gpus, rank=gpu) # Add this line

    model = Net().to(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # Add this line for synchronization of batch normalization
    torch.backends.cudnn.benchmark = True
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)


####### Refer to tutorial in https://pytorch.org/docs/stable/ddp_comm_hooks.html for gradient compression #######

    if args.compression != 'none':
        if args.compression == 'fp16':
            ##### Option 1, fp16 gradient compression
            # model.register_comm_hook(state=None, hook=fp16_compress_hook) 
            model.register_comm_hook(state=None, hook=torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook) 

        if args.compression == 'powersgd':
            #### Option 2, PowerSGD gradient compression
            state = powerSGD.PowerSGDState(
            process_group=None, 
            matrix_approximation_rank=2,
            start_powerSGD_iter=1_000,
            )
            model.register_comm_hook(state, powerSGD.powerSGD_hook)

        if args.compression == 'powersgd_fp16':
            #### Option 3, PowerSGD + fp16 gradient compression
            state = powerSGD.PowerSGDState(
            process_group=None, 
            matrix_approximation_rank=2,
            start_powerSGD_iter=1_000,
            )
            model.register_comm_hook(state, torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper(powerSGD.powerSGD_hook))

        if args.compression == 'batched_powersgd_fp16':
            #### Option 4, BatchedPowerSGD + fp16 gradient compression
            state = powerSGD.PowerSGDState(
            process_group=None, 
            matrix_approximation_rank=2,
            start_powerSGD_iter=1_000,
            )
            model.register_comm_hook(state, torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper(powerSGD.batched_powerSGD_hook))  
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                    num_replicas=nums_gpus,
                    rank=gpu)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=nums_gpus,
        rank=gpu
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True, 
        sampler=train_sampler)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,        # Set to False when using DistributedSampler
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler  # Pass the custom sampler here
    )
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    
    for epoch in range(args.epochs):
        ###### Training ######
        train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(gpu, non_blocking=True), target.cuda(gpu, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


            reduced_loss = loss.data.clone()
            reduced_correct_top1, reduced_correct_top5 = accuracy(output, target, topk=(1, 5))
            dist.reduce(reduced_loss, dst=0)
            dist.reduce(reduced_correct_top1, dst=0)
            dist.reduce(reduced_correct_top5, dst=0)

            if gpu == 0:
                if batch_idx % args.log_interval == 0:
                    reduced_loss /= nums_gpus
                    reduced_correct_top1 /= nums_gpus
                    reduced_correct_top5 /= nums_gpus
                    print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}\t"
                        f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t"
                        f"Loss: {reduced_loss.item():.6f}\t"
                        f"Top 1 Acc: {reduced_correct_top1.item():.2f}%\t"
                        f"Top 5 Acc: {reduced_correct_top5.item():.2f}%")
                    
        ###### Test ######
        model.eval()
        test_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_item = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(gpu), target.to(gpu)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * target.size(0)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                total_item += target.size(0)
                correct_top1 += acc1.item() * target.size(0)
                correct_top5 += acc5.item() * target.size(0)

                loss = loss
                acc1 = acc1.data.clone()
                acc5 = acc5.data.clone()
                dist.reduce(loss, dst=0)
                dist.reduce(acc1, dst=0)
                dist.reduce(acc5, dst=0)
                if gpu == 0:
                    if batch_idx % args.log_interval == 0:
                        loss /= nums_gpus
                        acc1 /= nums_gpus
                        acc5 /= nums_gpus
                        print(f"Test Epoch: {epoch} [{batch_idx}/{len(test_loader)}\t"
                            f"({100.0 * batch_idx / len(test_loader):.0f}%)]\t"
                            f"Loss: {loss.item():.6f}\t"
                            f"Top 1 Acc: {acc1.item():.2f}%\t"
                            f"Top 5 Acc: {acc5.item():.2f}%")

        correct_top1 = correct_top1/total_item
        correct_top5 = correct_top5/total_item
        test_loss = test_loss/total_item
        reduced_test_loss = torch.tensor(test_loss, dtype=torch.float, device=gpu)
        reduced_correct_top1 = torch.tensor(correct_top1, dtype=torch.float, device=gpu)
        reduced_correct_top5 = torch.tensor(correct_top5, dtype=torch.float, device=gpu)
        dist.reduce(reduced_test_loss, dst=0)
        dist.reduce(reduced_correct_top1, dst=0)
        dist.reduce(reduced_correct_top5, dst=0)

        if gpu == 0:
            num_gpus = torch.cuda.device_count()
            reduced_test_loss /= num_gpus #* len(test_loader.dataset)
            reduced_correct_top1 = reduced_correct_top1.item() / num_gpus
            reduced_correct_top5 = reduced_correct_top5.item() / num_gpus
            print(f"\nTest Epoch: {epoch}: Average loss: {reduced_test_loss:.4f}, "
                    f"Top 1 Acc: {reduced_correct_top1:.2f}%, "
                    f"Top 5 Acc: {reduced_correct_top5:.2f}%\n")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--compression', type=str, default='none', \
                        choices=['none', 'fp16', 'powersgd', 'powersgd_fp16', 'batched_powersgd_fp16'])
    parser.add_argument('--log_interval', type=int, default=10)
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()

    # mp.spawn(train, nprocs=num_gpus, args=(args,), join=True)


    mp.set_start_method("spawn")
    processes = []

    try:
        for gpu in range(num_gpus):
            p = mp.Process(target=train, args=(gpu, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating processes...")
        for p in processes:
            p.terminate()
            p.join()
            

    print("Exiting main process.")

if __name__ == '__main__':
    main()