import os
import sys
import subprocess

def pip_upgrade(package):
    subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", package])

def pip_install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", "--quiet", package])

print('********** Pip Install MONAI **********')
pip_upgrade('setuptools')
pip_install('monai==0.4.0')
pip_install('nibabel == 3.2.1')

import time
import json
import logging
import argparse
from glob import glob
from multiprocessing   import Pool, cpu_count

import monai
import torch
import numpy as np
import nibabel as nib
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

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
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ToTensord(keys=["image", "label"])
    ]
)


def apply_transformations(file):
    output = train_transforms(file)
    output_name = file["image"].split('/')[-2] + '.pt'
    torch.save(output, os.path.join(output_dir, output_name)) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_path', type=str, default="/opt/ml/processing")
    args, _ = parser.parse_known_args()
    
    input_dir = args.local_path + "/input/"
    input_dir_unzipped = input_dir + "unzipped/"
    output_dir = args.local_path + "/train/"
    
    logger.info('Make new folder for unzipped pairs')
    make_unzipped_dir = os.system("mkdir " + input_dir_unzipped)

    
    zipped_pairs = glob(input_dir+"*.zip")
    logger.info('Total file zipped pairs: %d' % (len(zipped_pairs)))

    for archive in zipped_pairs:
        os.system("unzip " + archive + " -d " + input_dir_unzipped)
    
    logger.info('Unzipping finished')

    files = sorted(glob(input_dir_unzipped+"*/*"))
    logger.info('Total unizpped pairs: %d' % (len(files)))

    data_list = [{"image": "{}/{}_image.nii.gz".format(files[i],files[i].split("/")[-1]),
                  "label": "{}/{}_label.nii.gz".format(files[i],files[i].split("/")[-1])
                 }
                 for i in range(len(files))]
    logger.info('Total JSON pairs: %d' % (len(data_list)))
        
    # Creates one process per machine cpu
    pool = Pool(processes = cpu_count() - 2)
    result = pool.map(apply_transformations, data_list)