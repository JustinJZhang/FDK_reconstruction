# FDK_reconstruction
实现锥束CT正投影和FDK重建，并探讨锥束CT各参数对FDK重建质量影响。

* 基本代码：```fdk_main.py```，其他实验结果参考各文件夹。
* 示例数据：```wugui.mat```，大小为(256, 256, 256)的乌龟数据。

获取完整实验代码、报告、数据或交流请start discussion或E-mail：JustinJZhang@outlook.com。

锥束CT正投影结果：
![](projection.png)

重建结果：
![](reconstruction.png)

## Requirements
* Astra 2.1：```conda install -c astra-toolbox astra-toolbox```
* h5py
* numpy
* torchvision
* matplotlib
* torch
* skimage
* sklearn

## Reference
[1] Feldkamp L A, Davis L C, Kress J W. Practical cone-beam algorithm[J]. Josa a, 1984, 1(6): 612-619.

[2] Van Aarle W, Palenstijn W J, Cant J, et al. Fast and flexible X-ray tomography using the ASTRA toolbox[J]. Optics express, 2016, 24(22): 25129-25147.

[3] Astra Document: www.astra-toolbox.com/index.html
