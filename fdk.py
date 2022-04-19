import h5py
import numpy as np
import astra

## 导入数据
data_path = './wuguiP3D.mat'
data = np.array(h5py.File(data_path)['reference'])

## 正投影
projection_geom = astra.create_proj_geom('cone', 1, 1, 256, 256, np.linspace(0, 3.14, 90), 200, 200)
vol_geom = astra.create_vol_geom([256, 256, 256])
projection, _ = astra.create_sino3d_gpu(data, projection_geom, vol_geom)

## FDK
recconstruction_shape = astra.data3d.create('-vol', vol_geom)
config_fdk = astra.astra_dict('FDK_CUDA')
config_fdk['ProjectionDataId'] = projection
config_fdk['ReconstructionDataId'] = recconstruction_shape
algorithm = astra.algorithm.create(config_fdk)
astra.algorithm.run(algorithm, 1)
recconstruction = astra.data3d.get(recconstruction_shape)