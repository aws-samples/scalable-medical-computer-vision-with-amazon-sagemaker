## Scalable Medical CV with Amazon SageMaker 
This example shows how we scaled a medical semantic segmentation training workload using MONAI's various dataset methods ([Dataset](https://docs.monai.io/en/stable/data.html#dataset), [PersistentDataset](https://docs.monai.io/en/stable/data.html#persistentdataset) and [CacheDataset](https://docs.monai.io/en/stable/data.html#cachedataset)) on single GPU device and using SageMaker's distributed training library (data parallel) on multiple GPU devices. The data IO, transformation and network architecture are built using PyTorch and MONAI library.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

