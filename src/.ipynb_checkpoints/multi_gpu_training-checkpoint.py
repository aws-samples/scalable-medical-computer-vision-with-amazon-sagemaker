import os
import sys
import time
import json
import logging
import argparse
from glob import glob

import monai
import torch
import numpy as np
import nibabel as nib
from monai.config import print_config
from monai.utils import set_determinism
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import partition_dataset, Dataset, DataLoader 

from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel
import smdistributed.dataparallel.torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
print_config()
    
# initialize the distributed training process, every GPU runs in a process
dist.init_process_group()


class ProcessedDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        file_path = self.data_list[index]
        file = torch.load(file_path)
        image = file['image']
        label = file['label']
        return (image, label)

def get_data_loaders(args):
    """
    This function loads input/output file names, builds a MONAI SmartCacheDataset
    (child of torch dataset) and a DistributedSampler for them. It returns a DataLoader 
    """

    # glob learning dataset dir
    data_list = sorted(glob(os.path.join(args.train, '*.pt')))
    if dist.get_rank() == 0:
        logger.info('Total file pairs: %d' % (len(data_list)))
    
    # split the data_list into train and val
    train_files, val_files = partition_dataset(
        data_list,
        ratios = [0.99, 0.01],
        shuffle = True,
        seed = args.seed
        )
    if dist.get_rank() == 0:
        logger.info('# of train and val: %d/%d' % (len(train_files), len(val_files)))
    
    # create training/validation data loaders
    if dist.get_rank() == 0:
        logger.info('Defining Train Datasets')
    train_ds = ProcessedDataset(data_list=train_files)
    if dist.get_rank() == 0:
        logger.info('Defining Validation Dataset')
    val_ds = ProcessedDataset(data_list=val_files)

    # create a training data sampler
    if dist.get_rank() == 0:
        logger.info('DistributedSampler')
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=args.world_size,
        rank=args.host_rank
        )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=args.world_size,
        rank=args.host_rank
        )
    
    if dist.get_rank() == 0:
        logger.info('Defining DataLoader')
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True, # was True
        sampler=train_sampler
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True, # was True
        sampler=val_sampler
        )

    return train_loader, val_loader

def get_model(args):

    if dist.get_rank() == 0:
        logger.info('Defining network')    
    device = torch.device("cuda") # WARNING: do not assign local rank here, i.e. cuda:0
    
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=4, # original brats data has 4 channel
        out_channels=3, 
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # wrap the model with sm DistributedDataParallel module
    model = DistributedDataParallel(model)
    # set the model to local GPUs
    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)

    return model, optimizer, device

def train(args, train_loader, val_loader, model, optimizer, device):
    loss_function = monai.losses.DiceLoss(
        to_onehot_y=False,
        sigmoid=True,
        squared_pred=True
    ).to(device)

    # Training epochs
    val_interval = 5
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []
    for epoch in range(args.epochs):
        if dist.get_rank() == 0:
            logger.info("-" * 10)
        if dist.get_rank() == 0:
            logger.info(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        tic = time.time()
        
        train_loader.sampler.set_epoch(epoch)
        #epoch_len = len(train_ds) // (train_loader.batch_size * args.world_size)
        
            
        for batch_data in train_loader:
            step += 1
            
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            #inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            #logger.info(f"train_loss: {loss.item():.4f}")
            #logger.info(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        if dist.get_rank() == 0:
            print("-" * 10)
            print("step is" + str(step))
            print("-" * 10)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        if dist.get_rank() == 0:
            logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            logger.info("secs_time_per_epoch: {}".format(time.time() - tic))
        
        # validation loop
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean")
                post_trans = Compose(
                    [Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
                )
                metric_sum = metric_sum_tc = metric_sum_wt = metric_sum_et = 0.0
                metric_count = (
                    metric_count_tc
                ) = metric_count_wt = metric_count_et = 0
                for val_data in val_loader:
                    #val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device))
                    val_inputs, val_labels = val_data
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)


                    
                    val_outputs = model(val_inputs)
                    val_outputs = post_trans(val_outputs)
                    # compute overall mean dice
                    value, not_nans = dice_metric(y_pred=val_outputs, y=val_labels)
                    not_nans = not_nans.item()
                    metric_count += not_nans
                    metric_sum += value.item() * not_nans
                    # compute mean dice for TC
                    value_tc, not_nans = dice_metric(y_pred=val_outputs[:, 0:1], 
                                                     y=val_labels[:, 0:1])
                    not_nans = not_nans.item()
                    metric_count_tc += not_nans
                    metric_sum_tc += value_tc.item() * not_nans
                    # compute mean dice for WT
                    value_wt, not_nans = dice_metric(y_pred=val_outputs[:, 1:2], 
                                                     y=val_labels[:, 1:2])
                    not_nans = not_nans.item()
                    metric_count_wt += not_nans
                    metric_sum_wt += value_wt.item() * not_nans
                    # compute mean dice for ET
                    value_et, not_nans = dice_metric(y_pred=val_outputs[:, 2:3], 
                                                     y=val_labels[:, 2:3])
                    not_nans = not_nans.item()
                    metric_count_et += not_nans
                    metric_sum_et += value_et.item() * not_nans

                metric = metric_sum / metric_count
                metric_values.append(metric)
                metric_tc = metric_sum_tc / metric_count_tc
                metric_values_tc.append(metric_tc)
                metric_wt = metric_sum_wt / metric_count_wt
                metric_values_wt.append(metric_wt)
                metric_et = metric_sum_et / metric_count_et
                metric_values_et.append(metric_et)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(),
                                   os.path.join(args.model_dir, "best_metric_model.pth"))
                        logger.info("saved new best metric model")
                logger.info(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )

    logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    #if dist.get_rank() == 0:
    #    # all processes should see same parameters as they all start from same
    #    # random parameters and gradients are synchronized in backward passes,
    #    # therefore, saving it in one process is sufficient
    #    torch.save(model.state_dict(), os.path.join(args.model_dir, "final_model.pth"))


def main():
    #parser = get_parser()
    parser = argparse.ArgumentParser(description='Job Args')        
    parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num cpu workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()
    
    args.world_size = dist.get_world_size()
    args.host_rank = rank = dist.get_rank()
    args.local_rank = local_rank = dist.get_local_rank()
    args.batch_size //= args.world_size // 8
    args.batch_size = max(args.batch_size, 1)
    args.data_dir = os.path.join(os.environ['SM_INPUT_DIR'], 'data')
    
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.host_rank)
    logger.info('Arguments: %s' % args)
    if not torch.cuda.is_available():
        raise Exception("Must run this example on CUDA-capable devices.")
    logger.info('Hello from rank %d of local_rank %d in world size of %d' %(rank, local_rank, args.world_size))

    set_determinism(seed=args.seed)
    train_loader, val_loader = get_data_loaders(args=args)
    model, optimizer, device = get_model(args)
    train(args, train_loader, val_loader, model, optimizer, device)

if __name__ == "__main__":
    main()
