## Scalable Medical CV with Amazon SageMaker 
Medical imaging, the technique to image and visualize a region of interest for clinical and diagnostic purposes, is under the most influence with the recent trend of applying highly accurate CV models trained with DL. Highly accurate and automated CV models that perform tasks such as pattern classification, abnormality detection, tissue segmentation have been used by clinicians in medical domains such as neuroradiology, cardiology, and pulmonology, to augment their clinical routines for a more reliable, faster and better patient care.

Training a CV model with DL in a medical domain, however, is uniquely challenging for data scientists and ML engineers compared to training a CV in other domains. Training a medical CV model requires a scalable compute and storage infrastructure for the following reasons:

- Medical imaging data presents much more sophisticated and subtle visual cues for most tasks. It requires a complex neural network architecture and a large amount of data.
- Medical imaging data standards are complex in nature as they need to carry much more information for medical and clinical use. ML training on such complex data requires lots of customized coding on top of an existing DL framework.
- Medical imaging data is significantly larger in size, attributed to high resolution and multiple dimensions (3D and 4D are common) beyond flat 2D images.
- The use of multiple imaging modalities in diagnosis and prognosis, and ML modeling is widely adopted in the medical domain.

A major challenge for model training that arises from these factors is the scalability of compute and storage. A critical topic is how data scientists and ML engineers conduct model training in the cloud in a manageable timeframe.

In this lab you will scale a medical semantic segmentation training workload using MONAI's various dataset methods ([Dataset](https://docs.monai.io/en/stable/data.html#dataset), [PersistentDataset](https://docs.monai.io/en/stable/data.html#persistentdataset) and [CacheDataset](https://docs.monai.io/en/stable/data.html#cachedataset)) on single GPU device and using [SageMaker's distributed training library (data parallel)](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html) on multiple GPU devices. The data IO, transformation and network architecture are built using PyTorch and MONAI library.

## Prerequisite
You need the following IAM permission attached to the IAM execution role of the SageMaker user profile in SageMaker Studio.
- [AmazonSageMakerFullAccess](https://us-east-1.console.aws.amazon.com/iam/home#/policies/arn:aws:iam::aws:policy/AmazonSageMakerFullAccess)

To run the training jobs in the notebook, you need to make sure you have sufficient quota for the following training instances.
- 3 `ml.g5.2xlarge`
- 1 `ml.p3.16xlarge`

The notebooks and codes are developed in Amazon SageMaker Studio. To get started with Amazon SageMaker Studio, please visit the [Onboard to Amazon SageMaker Domain](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) page.

## Dataset
We use a multi-modal MRI brain tumor image segmentation benchmark dataset named [BraTS](https://ieeexplore.ieee.org/document/6975210). There are 750 patients (484 training and 266 testing) from BraTS’16 and ’17. This dataset is provided by [Medical Segmentation Decathlon](http://medicaldecathlon.com/), and can be downloaded from the [AWS Open Data Registry](https://registry.opendata.aws/msd/). Individual files are saved as standard GNU zip (gzip) compressed archives of the multi-dimensional neuroimaging [NIfTI](https://nifti.nimh.nih.gov/nifti-1/) format (for example, `imagesTr/BRATS_001.nii.gz`, `labelsTr/BRATS_001.nii.gz`). On average, each image or label pair takes up about 9.6 MB on disk (~135 MB when in decompressed form .nii). The 484 training pairs have a disk footprint of 4.65 GB.

Please open the [00-data_prep.ipynb](https://github.com/aws-samples/scalable-medical-computer-vision-with-amazon-sagemaker/blob/workshop/00-data_prep.ipynb) to download, explore, and visualize the dataset. You should use `ml.t3.medium` instance with `Python 3 (Data Science)` kernel for the notebook.

We can plot the MRI images and the corresponding tumor labels using `nilearn` library. The MRI imaging sequences included in this dataset are T2-FLAIR, T1, T1Gd, and T2-weighted (top to bottom in the left panel in the figure below). The four modalities are concatenated as the fourth dimension (channel) on top of the X, Y, and Z dimensions. The target of this dataset is to segment gliomas tumors and subregions. The tumor can also be partitioned as peritumoral edema (1), GD-enhancing tumor (2), and the necrotic (or non-enhancing) tumor core (3) (overlayed in the right panel in the figure below). The tumor can also be delineated as a whole tumor (WT), tumor core (TC), and enhancing tumor (ET), which is the label definition we used in this work. Human expert annotation is conducted and provided within the dataset.

![brats-mri](https://d2908q01vomqb2.cloudfront.net/c5b76da3e608d34edb07244cd9b875ee86906328/2022/06/29/Figure-1-2-853x1024.png)

![brats-label](https://d2908q01vomqb2.cloudfront.net/c5b76da3e608d34edb07244cd9b875ee86906328/2022/06/29/Figure-2.png)

Once you complete the [00-data_prep.ipynb](https://github.com/aws-samples/scalable-medical-computer-vision-with-amazon-sagemaker/blob/workshop/00-data_prep.ipynb), please open [01-model_training.ipynb](https://github.com/aws-samples/scalable-medical-computer-vision-with-amazon-sagemaker/blob/workshop/01-model_training.ipynb) from the file explorer.

## Model training overview
We will train a neural network with the NIfTI images consists of the following services/resources:

- **Amazon SageMaker distributed training library**, with only a few lines of additional code, can be used to achieve data parallelism or model parallelism to the training script. SageMaker optimizes the distributed training jobs through algorithms that are designed to fully utilize AWS compute and network infrastructure in order to achieve near-linear scaling efficiency, which allows you to complete training faster than manual implementations.
- **MONAI** is an open source project bringing medical imaging ML community together to create best practices of CV for medical imaging. It is built on top of a PyTorch framework and follows a native PyTorch programming paradigm, which is widely adopted. We use the data transformations and UNet model architecture from MONAI library.

Below is an illustration of SageMaker Distributed Training using multiple GPU devices.

![sm-ddp](https://d2908q01vomqb2.cloudfront.net/c5b76da3e608d34edb07244cd9b875ee86906328/2022/06/29/Figure-3.-Distributed-Data-Parallel-Training-by-Amazon-SageMaker.png)

## Model development with MONAI
MONAI is a medical imaging domain-specific library that offers a wide range of features and functionalities for medical imaging-specific data formats. Developers no longer need to write custom data loaders to process and train medical imaging data. You also do not lose any data integrity with unnecessary data conversion to other formats such as JPG or PNG. In addition, MONAI provides medical imaging-specific image processing as transformation and deep learning network architectures, which are proven in the medical imaging community. With this capability, developers don’t need to implement from scratch.

In our model training, we employ the following transformation from MONAI as defined in [src/single_gpu_training.py](https://github.com/aws-samples/scalable-medical-computer-vision-with-amazon-sagemaker/blob/workshop/src/single_gpu_training.py#L88-L109).

```python
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
```
Notably, MONAI’s [transforms](https://docs.monai.io/en/latest/transforms.html) API supports dictionary-based data input and indexing. In medical imaging ML, it is typical to have image data and the label data saved in separate files. MONAI’s dictionary-based transforms (class name ending with -d, such as LoadImage**d** and RandFlip**d**) are suitable for this scenario. We can easily compose a chain of transformation for either the image or the label data with a key. In this transformation for training data we do the following (in the same order as in `transforms`):
- Load the NIfTI image pair as numpy arrays with the NIfTI headers associated.
- Make the image data channel first.
- Reconstruct the labels to create the aggregated labels: tumor core (TC), enhancing tumor (ET), and whole tumor (WT). This is a custom function.
- Resample the image and label data.
- Reorient both image and label data to RAS, which is the neurological convention.
- Randomly crop the image and label to reduce the size of the image and augment the data with randomness.
- Randomly flip the image and label on the first axis.
- Normalize the intensity of the image.
- Randomly scale the intensity of the image.
- Randomly shift the intensity of the image.
- Finally, convert the numpy arrays into tensors (still with the dictionary structure).

For the semantic segmentation model, we use the 3D [UNet](https://docs.monai.io/en/latest/networks.html#unet) implementation from MONAI. Note that the input data is a three-dimensional image with four modalities (channels).

```python
model = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=4, # BraTS data has 4 channels (modalities)
    out_channels=3, 
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
```

Once we understand our model training script in `src/single_gpu_training.py`, we can proceed to train with Amazon SageMaker using the managed training feature.

## Model training with SageMaker managed training
The SageMaker Studio environment is suitable for interactive data exploration, script writing, and prototyping of feature engineering and modeling. We recommend using notebooks with instances that are smaller in compute for interactive building and leaving the heavy lifting to ephemeral training, tuning, and processing jobs with larger instances. This way, you don’t keep a large instance (or a GPU) constantly running with your notebook. This can help you minimize your build costs by selecting the right instance.

When you use fully managed Amazon SageMaker Training, it dispatches all things needed for a training job, such as code, container, and data to a compute infrastructure separate from the SageMaker notebook instance. Therefore, your training jobs aren’t limited by the compute resource of the notebook instance. The SageMaker Training Python SDK lets you to launch the training and tuning jobs at will. 

### Single GPU Device Experiments

We will run 3 experiments for three of MONAI's dataset classes to show you can run a model training script built with MONAI in SageMaker managed training and how the choice of Dataset and data loading strategies affect the model training efficiency with a GPU device. 

1. [`Dataset`](https://docs.monai.io/en/stable/data.html#monai.data.Dataset): standard data loading
2. [`PersistentDataset`](https://docs.monai.io/en/stable/data.html#persistentdataset): persist processed data on disk
3. [`CacheDataset`](https://docs.monai.io/en/stable/data.html#cachedataset): persist processed data in CPU memory

You can find more information about MONAI's dataset classes by following the links above.

We use the PyTorch estimator from SageMaker Python SDK and bring in the training script `single_gpu_training.py` using the [script mode](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#train-a-model-with-pytorch). Note that we provide training code as the `entry_point` with `source_dir` to the PyTorch estimator. The library dependency (MONAI and others) for the training code should be listed with a `requirements.txt` file located in the `source_dir`. 

SageMaker estimators help you choose the compute resource needed for the job. You can choose which compute resource you’d like to use as part of the estimator construct with `instance_count` and `instance_type`. In the preceding code snippet, one `ml.p3.2xlarge` instance was shown as an example, which has one NVIDIA Tesla V100 GPU with 16-GiB GPU memory. You can also run the training with other GPU instances such as `ml.g4dn.xlarge`, and `ml.g5.2xlarge`. For more about the instance types and cost, please visit the [SageMaker Pricing Page](https://aws.amazon.com/sagemaker/pricing/).

The experiments runs should produce 3 training jobs. Visit the SageMaker training jobs for details on each. The `CacheDataset` run should be the fastest, followed by `PersistentDataset` and `Dataset`.

### Multi-GPU Device Experiment
In this section we show how to distribute the previous training job across multiple GPU devices. Each device is assigned a single process. Each process performs the same task (forward and backward passes) on different shards of data. That is, we show how to use [SageMaker’s distributed data parallel (SMDDP)](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html) library. Note that SageMaker also supports [distributed model parallel training](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html) which is suitable for training jobs where the model is too large to fit into a GPU device.

Adopting SMDDP requires only a few changes to our training script. To start, we initialize a distributed processing group and set each GPU to a single process. We also import the SageMaker `DistributedDataParallel` class, which is an implementation of distributed data parallelism (DDP) for PyTorch. You can find the following snippets in [src/multi_gpu_training.py](https://github.com/aws-samples/scalable-medical-computer-vision-with-amazon-sagemaker/blob/workshop/src/multi_gpu_training.py)).

```python
import torch
import smdistributed.dataparallel.torch.distributed as dist
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel

# initialize the distributed training processing group
dist.init_process_group()

# pin each GPU to a single process
torch.cuda.set_device(dist.get_local_rank())
```

Next, we replicate our data loader across our group of processes using PyTorch’s `DistributedSampler`. The number of replicas is set via `num_replicas=arg.world_size` and each replica is assigned a rank within the group.

```python
from torch.utils.data.distributed import DistributedSampler
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    sampler=DistributedSampler(train_ds, 
                              num_replicas=args.world_size, 
                              rank=args.host_rank)
 )
 ```

 The model definition does not need to change. We do, however, need to wrap the model with the SageMaker `DistributedDataParallel` class. This grants each process its own copy of the model so it can perform the forward and backward passes on its subset of each batch of data.

```python
 # wrap the model with sm DistributedDataParallel module
model = DistributedDataParallel(model)
```

Lastly, we save checkpoints only on the leader node (the node that `dist.get_rank()` returns zero) in the `trainer` function. This is done with an if-statement in the training loop:

```python
if dist.get_rank() == 0:
    torch.save(
        model.state_dict(),
        os.path.join(args.model_dir, "best_model.pth")
        )
```

These are the only changes we need to make to our entry point script. Now, back to the `01-model_training.ipynb` notebook, in order to execute this updated entry point, we must activate the distribution option in the SageMaker’s PyTorch estimator:

```python
estimator = PyTorch(
    ...
    entry_point='multi_gpu_training.py',
    instance_count=instance_count,
    instance_type=instance_type,
    distribution={'smdistributed': {'dataparallel': {'enabled': True}}}
)
```

By setting `instance_count=1` and `instance_type='ml.p3.16xlarge'`, which is [one of the instance type supported by SageMaker DDP](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-faq.html), we would be distributing data and compute to all 8 GPU devices on the instance, resulting a faster run-time.

## Clean up
To avoid resources incurring charges, remove the data in the Amazon S3 bucket, and the kernel gateway apps from SageMaker Studio. Instances behind SageMaker training jobs are automatically shut down at the end of the jobs.

## Reference
This example is an adaptation from a full solution presented in the blog post series [Scalable Medical Computer Vision Model Training with Amazon SageMaker part 1](https://aws.amazon.com/blogs/industries/scalable-medical-computer-vision-model-training-with-amazon-sagemaker-part-1/), [part 2](https://aws.amazon.com/blogs/industries/scalable-medical-computer-vision-model-training-with-amazon-sagemaker-part-2/).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

