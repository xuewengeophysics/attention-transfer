

搞清楚attention loss是在哪里计算的？



```json
json_stats: {"depth": 16, "width": 1, "dataset": "CIFAR10", "dataroot": ".", "dtype": "float", "nthread": 4, "teacher_id": "resnet_16_2_teacher", "batch_size": 128, "lr": 0.1, "epochs": 200, "weight_decay": 0.0005, "epoch_step": "[60,120,160]", "lr_decay_ratio": 0.2, "resume": "", "randomcrop_pad": 4, "temperature": 4, "alpha": 0, "beta": 1000.0, "cuda": false, "save": "logs/at_16_1_16_2", "ngpu": 1, "gpu_id": "0", "train_loss": 0.649514677274562, "train_acc": 92.194, "test_loss": 0.8538061327572111, "test_acc": 87.63, "epoch": 103, "num_classes": 10, "n_parameters": 175994, "train_time": 37.58035182952881, "test_time": 6.2485761642456055, "at_losses": [[1.0369961391722654e-05, 3.747507565364576e-07], [5.250534164549609e-05, 2.547601211105283e-06], [0.0003698490768116206, 2.7313048405678e-05]]}
```

attention loss的list为什么是3个呢？

```python
meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]
```

attention loss为什么是2个值呢？均值、标准差

```python
def on_end_epoch(state):
    "at_losses": [m.value() for m in meters_at]
```

```python
    def value(self):
        return self.mean, self.std
```



```python
def h(sample):	
    [m.add(v.item()) for m, v in zip(meters_at, loss_groups)]
```

```python
    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))
```



loss_groups是什么？

```python
    def h(sample):
        inputs = utils.cast(sample[0], opt.dtype).detach()
        targets = utils.cast(sample[1], 'long')
        #如果有教师模型，进行TS训练
        if opt.teacher_id != '':
            y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))
            loss_groups = [v.sum() for v in loss_groups]
            #计算meters_at，即at_losses注意力loss
            [m.add(v.item()) for m, v in zip(meters_at, loss_groups)]
            return utils.distillation(y_s, y_t, targets, opt.temperature, opt.alpha) + opt.beta * sum(loss_groups), y_s
        #如果没有教师模型，训练单个模型
        else:
            y = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))[0]
            return F.cross_entropy(y, targets), y
```

```python
def data_parallel(f, input, params, mode, device_ids, output_device=None):
    device_ids = list(device_ids)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)
```



```json
tensor inputs =  torch.Size([128, 3, 32, 32])  #batch_size是128
dict params =  dict_keys([
    'student.conv0', 
    'student.group0.block0.conv0', 'student.group0.block0.conv1', 'student.group0.block0.bn0.weight', 'student.group0.block0.bn0.bias', 'student.group0.block0.bn0.running_mean', 'student.group0.block0.bn0.running_var', 'student.group0.block0.bn1.weight', 'student.group0.block0.bn1.bias', 'student.group0.block0.bn1.running_mean', 'student.group0.block0.bn1.running_var', 
    'student.group0.block1.conv0', 'student.group0.block1.conv1', 'student.group0.block1.bn0.weight', 'student.group0.block1.bn0.bias', 'student.group0.block1.bn0.running_mean', 'student.group0.block1.bn0.running_var', 'student.group0.block1.bn1.weight', 'student.group0.block1.bn1.bias', 'student.group0.block1.bn1.running_mean', 'student.group0.block1.bn1.running_var', 
    'student.group1.block0.conv0', 'student.group1.block0.conv1', 'student.group1.block0.bn0.weight', 'student.group1.block0.bn0.bias', 'student.group1.block0.bn0.running_mean', 'student.group1.block0.bn0.running_var', 'student.group1.block0.bn1.weight', 'student.group1.block0.bn1.bias', 'student.group1.block0.bn1.running_mean', 'student.group1.block0.bn1.running_var', 'student.group1.block0.convdim', 
    'student.group1.block1.conv0', 'student.group1.block1.conv1', 'student.group1.block1.bn0.weight', 'student.group1.block1.bn0.bias', 'student.group1.block1.bn0.running_mean', 'student.group1.block1.bn0.running_var', 'student.group1.block1.bn1.weight', 'student.group1.block1.bn1.bias', 'student.group1.block1.bn1.running_mean', 'student.group1.block1.bn1.running_var', 
    'student.group2.block0.conv0', 'student.group2.block0.conv1', 'student.group2.block0.bn0.weight', 'student.group2.block0.bn0.bias', 'student.group2.block0.bn0.running_mean', 'student.group2.block0.bn0.running_var', 'student.group2.block0.bn1.weight', 'student.group2.block0.bn1.bias', 'student.group2.block0.bn1.running_mean', 'student.group2.block0.bn1.running_var', 'student.group2.block0.convdim', 
    'student.group2.block1.conv0', 'student.group2.block1.conv1', 'student.group2.block1.bn0.weight', 'student.group2.block1.bn0.bias', 'student.group2.block1.bn0.running_mean', 'student.group2.block1.bn0.running_var', 'student.group2.block1.bn1.weight', 'student.group2.block1.bn1.bias', 'student.group2.block1.bn1.running_mean', 'student.group2.block1.bn1.running_var', 
    'student.bn.weight', 'student.bn.bias', 'student.bn.running_mean', 'student.bn.running_var', 
    'student.fc.weight', 'student.fc.bias', 
    
    'teacher.conv0', 
    'teacher.group0.block0.conv0', 'teacher.group0.block0.conv1', 'teacher.group0.block0.bn0.weight', 'teacher.group0.block0.bn0.bias', 'teacher.group0.block0.bn0.running_mean', 'teacher.group0.block0.bn0.running_var', 'teacher.group0.block0.bn1.weight', 'teacher.group0.block0.bn1.bias', 'teacher.group0.block0.bn1.running_mean', 'teacher.group0.block0.bn1.running_var', 'teacher.group0.block0.convdim', 
    'teacher.group0.block1.conv0', 'teacher.group0.block1.conv1', 'teacher.group0.block1.bn0.weight', 'teacher.group0.block1.bn0.bias', 'teacher.group0.block1.bn0.running_mean', 'teacher.group0.block1.bn0.running_var', 'teacher.group0.block1.bn1.weight', 'teacher.group0.block1.bn1.bias', 'teacher.group0.block1.bn1.running_mean', 'teacher.group0.block1.bn1.running_var', 
    'teacher.group1.block0.conv0', 'teacher.group1.block0.conv1', 'teacher.group1.block0.bn0.weight', 'teacher.group1.block0.bn0.bias', 'teacher.group1.block0.bn0.running_mean', 'teacher.group1.block0.bn0.running_var', 'teacher.group1.block0.bn1.weight', 'teacher.group1.block0.bn1.bias', 'teacher.group1.block0.bn1.running_mean', 'teacher.group1.block0.bn1.running_var', 'teacher.group1.block0.convdim', 
    'teacher.group1.block1.conv0', 'teacher.group1.block1.conv1', 'teacher.group1.block1.bn0.weight', 'teacher.group1.block1.bn0.bias', 'teacher.group1.block1.bn0.running_mean', 'teacher.group1.block1.bn0.running_var', 'teacher.group1.block1.bn1.weight', 'teacher.group1.block1.bn1.bias', 'teacher.group1.block1.bn1.running_mean', 'teacher.group1.block1.bn1.running_var', 'teacher.group2.block0.conv0', 'teacher.group2.block0.conv1', 'teacher.group2.block0.bn0.weight', 'teacher.group2.block0.bn0.bias', 'teacher.group2.block0.bn0.running_mean', 'teacher.group2.block0.bn0.running_var', 'teacher.group2.block0.bn1.weight', 'teacher.group2.block0.bn1.bias', 'teacher.group2.block0.bn1.running_mean', 'teacher.group2.block0.bn1.running_var', 'teacher.group2.block0.convdim', 
    'teacher.group2.block1.conv0', 'teacher.group2.block1.conv1', 'teacher.group2.block1.bn0.weight', 'teacher.group2.block1.bn0.bias', 'teacher.group2.block1.bn0.running_mean', 'teacher.group2.block1.bn0.running_var', 'teacher.group2.block1.bn1.weight', 'teacher.group2.block1.bn1.bias', 'teacher.group2.block1.bn1.running_mean', 'teacher.group2.block1.bn1.running_var', 
    'teacher.bn.weight', 'teacher.bn.bias', 'teacher.bn.running_mean', 'teacher.bn.running_var', 
    'teacher.fc.weight', 'teacher.fc.bias'])
sample =  True
opt.ngpu =  range(0, 1)
```



```json
y_s =  torch.Size([128, 10])  #batch_size是128，分类的类别数是10
y_t =  torch.Size([128, 10])  #batch_size是128，分类的类别数是10
loss_groups =  [tensor(0.0004, device='cuda:0', grad_fn=<MeanBackward0>), tensor(0.0017, device='cuda:0', grad_fn=<MeanBackward0>), tensor(0.0054, device='cuda:0', grad_fn=<MeanBackward0>)]  #3个TS对的attention loss
```





```python
def on_start_epoch(state):
    [meter.reset() for meter in meters_at]
```

```python
    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
```





```python
import math
from . import meter
import numpy as np


class AverageValueMeter(meter.Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
```

