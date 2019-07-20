# bert-multi-gpu

## Overview
This code is the **multi-gpu version(data parallelism)** of <a href="https://github.com/google-research/bert">BERT</a>, or Bidirectional Encoder Representations from Transformer. Most of the code is tested on Tensorflow 1.9 with NVIDIA® Tesla® V100 with MSMARCO dataset, a large scale passage re-ranking task containing more than 1 million real Bing user query with corresponding top 1,000 possible documents. You can find more information <a href="http://www.msmarco.org/">here</a>. The default version of this code is for text classification, but you can change it for any kind of data you are interested in. What you need to do is to modify the input and output data.

## Dependencies
- Tensorflow >= 1.9.0 # CPU Version of TensorFlow.
- Tensorflow-gpu >= 1.9.0 # GPU version of TensorFlow.
**You don't need to install other packages! Everything is implemented with tensorflow.**

## Performance
We have tested the performance on 2 and 4 GPU and compared it with single GPU version. With 4 GPU, it can be **2~3 times faster** than single GPU version of the official code.

## Usage
### 1. ```run_multi_gpu.py```
Most of the code related to data parallelism is in this file. The main idea of data parallelism is to copy the exactly same model and compute gradients on seperated GPU, and then update weights on CPU. The reason to design this architecture is that weights are shared between GPU, and we only have to update once after calculating gradients on seperate GPU, therefore, after computing, we basically sum up the gradients and update the weights on single CPU.

For implementation, you can simply open terminal and run ```python run_multi_gpu.py```. There are two more things need to be mentioned, first,  fine-tuning hyperparameter. The following is the highlight of some important parameters:

- ```data_dir```: Directory of dataset. More details for the required dataset can be found in ```create_pretraining_data.py```, the default version is ```tfrecord```.
- ```bert_config_file```: Directory of ```bert_config.json``` (BERT_BASE)
- ```bert_config_file_large```: Directory of ```bert_config.json``` (BERT_LARGE)
- ```init_checkpoint```: Directory of ```bert_model.ckpt``` (BERT_BASE)
- ```init_checkpoint_large``` Directory of```bert_model.ckpt``` (BERT_LARGE)
- ```output_dir```: **Checkpoints** will be saved in this directory.

- ```num_gpu```: The most important parameter, the **number of GPU** you want to use.
- ```is_training```: During training, it should be ```True```, otherwise ```False```.
- ```is_fine_tuning```: If you want to fine-tune BERT model, it should be ```True```, otherwise ```False```.

- ```num_labels```: Number of labels of classification task. Default is 2.
- ```max_seq_length```: Maximum sequence length. Default is 512.
- ```train_batch_size```: Batch size for training data. For example, if ```train_batch_size``` is 16, and ```num_gpu``` is 4, your **GLOBAL** batch size is 16, and the **LOCAL** batch size is 4 (batch size for each GPU).
- ```learning_rate```: Learning rate for Adam optimizer initialization.
- ```num_train_steps```: Number of training epoch.
- ```num_warmup_steps```: The orginal BERT optimizer provide this parameter. Use it if you want to warm up.

```python
# hyperparameters
data_dir = '../MSMARCO_tfrecord/'
bert_config_file = '../uncased_L-12_H-768_A-12/bert_config.json'
bert_config_file_large = '../uncased_L-24_H-1024_A-16/bert_config.json'
init_checkpoint = '../uncased_L-12_H-768_A-12/bert_model.ckpt'
init_checkpoint_large = '../uncased_L-24_H-1024_A-16/bert_model.ckpt'
output_dir = './model_dense_multi_GPU/'

num_gpu = 1
is_training = True
is_fine_tuning = False

num_labels = 2
max_seq_length = 512
query_length = 64
document_length = 448 # 512 - 64 = 448
train_batch_size = 8
learning_rate = 1e-5
num_train_steps = 400000
num_warmup_steps = 0
save_checkpoints_steps = 500
iterations_per_loop = 1000

num_warmup_steps = num_warmup_steps * iterations_per_loop
num_train_steps = num_train_steps * iterations_per_loop
```

### 2. ```create_pretraining_data.py```
Data preprocessing is done in this file. The default data type is ```tfrecord```, most of this code is same as the official one.

### 3. ```bert-multi-gpu.ipynb```
Feel free to use ```Google Colab``` or ```Jupyter Notebook``` with this file. 
