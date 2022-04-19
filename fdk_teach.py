import h5py
import numpy as np
import astra

## 导入数据
data_path = './wuguiP3D.mat'
data = np.array(h5py.File(data_path)['reference'])

## 正投影
# 投影的结构，定义扫描的参数，造ct
projection_geom = astra.create_proj_geom('cone', 1, 1, 256, 256, [0,12.56], 800, 0)
# 1.锥束 'cone'
# 2.检测器像素 1,1
# 3.检测器的大小 256，256
# 4.扫描的角度，[xxx]
# 5.发射源到乌龟中心的距离
# 6.检测器到乌龟中心的距离

# 物体的结构
vol_geom = astra.create_vol_geom([256, 256, 256])

# 得到正投影
projection = astra.create_sino3d_gpu(data, projection_geom, vol_geom)
print(projection.shape)
# 1. 乌龟
# 2. 扫描的结构
# 3. 物体的结构

## FDK
# 定义重建的形状，理解为一个特定形状的空的容器
recconstruction_shape = astra.data3d.create('-vol', vol_geom)
# 1. 不改
# 2. 重建后你期望的形状，不一致会插值/压缩

# 重建算法配置设置
config_fdk = astra.astra_dict('FDK_CUDA')      # 用的是FDK
config_fdk['ProjectionDataId'] = projection    # 重建的输入：正投影
config_fdk['ReconstructionDataId'] = recconstruction_shape     # 重建的形状

# 根据配置造出这个FDK算法
algorithm = astra.algorithm.create(config_fdk)

# 跑这个算法
astra.algorithm.run(algorithm, 1)
# 1. 你的算法
# 2. 迭代次数

# 取出重建的结果
recconstruction = astra.data3d.get(recconstruction_shape)

## 保存
np.save('zwh.npy', recconstruction)