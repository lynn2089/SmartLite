# SmartLite
Binarized Neural Network is based on https://github.com/eladhoffer/convNet.pytorch

Please install torch and torchvision by following the instructions at: http://pytorch.org/

To run resnet18 for cifar10 dataset use: python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10

bitwise_processing.py includes the example of bitwise processing in DBMS.

data_processing_in_db.py is used to process the data from devices.

im2col.py is used to flatten the tensor data.

Many test_x.py are some experiments used to measure the performance of DBMS.

torch_modelload.py is the example of multi-model running on PyTorch.

Many main_x.py are the processing of quantized model generatoring.

/data/ includes the dataset.

/results/ is used to store the quantized models.



