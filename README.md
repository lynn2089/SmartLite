# SmartLite

SmartLite is a lightweight DBMS that stores the parameters and structural information of neural networks as database tables and implements neural network operators inside the DBMS engine. SmartLite quantizes model parameters as binarized values, applies neural pruning techniques to compress the models, and transforms tensor manipulations into value lookup operations of the DBMS to reduce computation overhead. 

**Notable:**
According to the Alibaba's regulations, the papers published in any conference or journal need to meet the legal and security requirements and pass the risk assessment, which usually will take more than two months. To demonstrate sufficient materials, we provide the core codes related to the SmartLite and omit the unimportant and tedious system-level codes.

## Architecture
<img src="./arch.pdf" width="60%" alt="architecture of SmartLite"/>

## Core Components

### Data

Please save the downloaded data in **. /data/**.


### Neural Network Quantization
In order to reduce the computational complexity for better storage in DBMS, the floating point neural network is transformed into a binarized neural network by quantization.
Binarized Neural Network (BNN) is based on https://github.com/eladhoffer/convNet.pytorch and https://github.com/itayhubara/BinaryNet.pytorch .

Please install torch and torchvision by following the instructions at: http://pytorch.org/

To run resnet18 for cifar10 dataset use: 
```
python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10
```

**Additional Notes:**

**NN_data_storage_in_db.py** is used to process the shape of tensor data and used to store the flattened data in the table.

**im2col.py** is used to flatten the tensor data.

**main_x.py** is the processing of quantized model generatoring.

**/data/** includes the dataset.



* **Code Directory:**
```
/layer_reshape_udf/
```
It is the format conversion function used for transforming the results of calculations from the upper layer of data to the input of the lower layer of data. This function has been implemented as a built-in function within ClickHouse, so it is necessary to compile the code into the ClickHouse source code before using it to ensure proper functionality.

### LookUp Table

LookUp Table is employed to store the results of limited binarization operations, typically stored in contiguous memory space and retrieved based on provided parameters. 

* **Code Directory:**
```
/lookuptable/
```

As LookUp is implemented as a built-in function in ClickHouse, it is essential to compile the code into the ClickHouse source code before using it.

### Pruned neural networks

Neural network pruning is aimed at reducing unnecessary neural network calculations in the database to improve the database's inference speed. The primary idea is to identify parameters that frequently change during the model training process but have minimal impact on accuracy. SmartLite marks them as invalid parameters and avoids their calculations.

**Additional Notes:**

**main_binary.py** tracks the number of parameter changes during iterations at runtime and stores the data in the statistical log.

**bnn_prune.py** is used to adjust the pruning ratio and accuracy. In the code, we provide two pruning strategies for selection. 

### Multi-model scheduling and serving

For the scenario involving multiple model serving, the strategy for resource optimization needs to be provided. In addition to model fine-tuning during the training process, we offer model scheduling based on the minimum-cost strategy. 

* **Code Directory:**
```
/exp-load/
```

**./exp-load/** documents the implementation of multi-model serving and scheduling, and it tests system memory and time consumption using VGGs and AlexNets.

* **Code Directory:**
```
/model_udf/
```

**./model_udf/** contains the code for models compiled into ClickHouse internals.

### Test

All **test_x_in_db.py** files are the experiments used to measure the performance of SmartLite based on different datasets (CIFAR10/ImageNet/MNIST). 
In the code, data generation methods for DL2SQL and SmartLite are provided. When conducting performance tests, please refer to the SQL queries in ./exp-load/.

As an example of the SQL in **basic_model_load.py**:
```
db = Client(host='localhost')
db.execute(q0_1)
```




