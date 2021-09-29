## Scalable Medical CV with Amazon SageMaker 
This project shows how we scaled a medical semantic segmentation training workload on terabytes of data from four days to four hours. We used Amazon SageMaker Processing for distributed decoding and augmentation of compressed 4D brain scans, Amazon FSx for Lusrtre to provide high performant data transfer from Amazon Simple Storage Service (S3) to compute hosts, and the SageMaker Distributed Data Parallel Training Library for multi-node data parallel neural network training. The data IO, transformation and network architecture are built using PyTorch and MONAI library.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

