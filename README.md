## Scalable Medical CV with Amazon SageMaker 
This example shows how we scaled a medical semantic segmentation training workload using MONAI's various dataset methods ([Dataset](https://docs.monai.io/en/stable/data.html#dataset), [PersistentDataset](https://docs.monai.io/en/stable/data.html#persistentdataset) and [CacheDataset](https://docs.monai.io/en/stable/data.html#cachedataset)) on single GPU device and using [SageMaker's distributed training library (data parallel)](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html) on multiple GPU devices. The data IO, transformation and network architecture are built using PyTorch and MONAI library.

## Prerequisite
You need the following IAM permission attached to the IAM execution role of the SageMaker user profile.
- [AmazonSageMakerFullAccess](https://us-east-1.console.aws.amazon.com/iam/home#/policies/arn:aws:iam::aws:policy/AmazonSageMakerFullAccess)

To run the training jobs in the notebook, you need to make sure you have sufficient quota for the following training instances.
- 3 `ml.g5.2xlarge`
- 1 `ml.p3.16xlarge`

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

