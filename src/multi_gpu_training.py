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
from monai.data import partition_dataset, Dataset, DataLoader 
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
)

from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel
import smdistributed.dataparallel.torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
print_config()
    
# initialize the distributed training process, every GPU runs in a process
dist.init_process_group()

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d

def get_data_loaders(args):
    """
    This function loads input/output file paths, builds a MONAI DataLoader
    from a PersistentDataset (child of torch dataset) for them.
    It returns a DataLoader for train and validation splits 
    """    
    images = sorted(glob(os.path.join(args.train, 'imagesTr', 'BRATS*.nii.gz')))
    segs = sorted(glob(os.path.join(args.train, 'labelsTr', 'BRATS*.nii.gz')))
    data_list = [{"image": img, "label": seg} for img, seg in zip(images, segs)]
    logger.info('Total file pairs: %d' % (len(data_list)))
    
    # split the data_list into train and val
    train_files, val_files = partition_dataset(
        data_list,
        ratios=[0.99, 0.01],
        shuffle=True,
        seed=args.seed
    )
    logger.info('# of train and val: %d/%d' % (len(train_files), len(val_files)))

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=[128, 128, 64], random_size=False
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    # create training/validation data loaders
    if dist.get_rank() == 0:
        logger.info('Defining Train Datasets')
        
    train_ds = Dataset(data=train_files, transform=train_transforms)
    
    if dist.get_rank() == 0:
        logger.info('Defining Validation Dataset')
        
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # create a training data sampler
    if dist.get_rank() == 0:
        logger.info('DistributedSampler')
        
    train_sampler = DistributedSampler(
                        train_ds,
                        num_replicas=args.world_size,
                        rank=args.host_rank)
    val_sampler = DistributedSampler(
                        val_ds,
                        num_replicas=args.world_size,
                        rank=args.host_rank)
    
    if dist.get_rank() == 0:
        logger.info('Defining DataLoader')
        
    train_loader = DataLoader(
                        train_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True, # was True
                        sampler=train_sampler)
    
    val_loader = DataLoader(
                        val_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True, # was True
                        sampler=val_sampler)

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
                        squared_pred=True).to(device)

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
            
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

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
                dice_metric_tc = monai.metrics.DiceMetric(include_background=True, reduction="mean")
                dice_metric_wt = monai.metrics.DiceMetric(include_background=True, reduction="mean")
                dice_metric_et = monai.metrics.DiceMetric(include_background=True, reduction="mean")
                post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=True)])
                metric_sum = metric_sum_tc = metric_sum_wt = metric_sum_et = 0.0
                metric_count = (
                    metric_count_tc
                ) = metric_count_wt = metric_count_et = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (val_data["image"].to(device), 
                                              val_data["label"].to(device))
                    
                    val_outputs = model(val_inputs)
                    val_outputs = post_trans(val_outputs)
                    # compute overall mean dice
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    # compute mean dice for TC
                    dice_metric_tc(y_pred=val_outputs[:, 0:1], y=val_labels[:, 0:1])
                    # compute mean dice for WT
                    dice_metric_wt(y_pred=val_outputs[:, 1:2], y=val_labels[:, 1:2])
                    # compute mean dice for ET
                    dice_metric_et(y_pred=val_outputs[:, 2:3], y=val_labels[:, 2:3])

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                dice_metric.reset()
                metric_tc = dice_metric_tc.aggregate().item()
                metric_values_tc.append(metric_tc)
                dice_metric_tc.reset()
                metric_wt = dice_metric_wt.aggregate().item()
                metric_values_wt.append(metric_wt)
                dice_metric_wt.reset()
                metric_et = dice_metric_et.aggregate().item()
                metric_values_et.append(metric_et)
                dice_metric_et.reset()
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
