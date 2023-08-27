# SmartLite

## Quantization neural networks and the lookup table
Binarized Neural Network is based on https://github.com/eladhoffer/convNet.pytorch

Please install torch and torchvision by following the instructions at: http://pytorch.org/

To run resnet18 for cifar10 dataset use: python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10

**NN_data_storage_in_db.py** is used to process the shape of data from devices and used to store the data in the table.

**im2col.py** is used to flatten the tensor data.

Many **main_x.py** are the processing of quantized model generatoring.

**/data/** includes the dataset.

## Pruned neural networks

**main_binary.py** includes the number of parameter flips used for model pruning.

**bnn_prune.py** is used to adjust the pruning ratio and accuracy.

## Multi-model scheduling and serving

**./exp-load** records the implementation of multi-model serving and scheduling, and tests VGG and AlexNet.

## Test

Many **test_x.py** are the experiments used to measure the performance of SmartLite.




