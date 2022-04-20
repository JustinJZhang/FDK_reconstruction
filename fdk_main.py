import h5py
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch
import astra
from skimage.measure import compare_ssim, compare_psnr
from sklearn.metrics import mean_squared_error

# Load data
data_path = '../wuguiP3D.mat'
data = h5py.File(data_path)['reference']
data_np = np.array(data)

# Visualize reference
plt.figure(figsize=(82, 82))
data_tensor = torch.from_numpy(data_np[:, np.newaxis, :, :])
ori_grid = np.transpose(vutils.make_grid(data_tensor, nrow=16), (1, 2, 0))
plt.imshow(ori_grid)
plt.axis("off")
plt.savefig('img_ori.png')

# Set scanning parameter
angle_num = 1000                # number of projection angles
angle_scale = 2 * np.pi         # scale of projection angles
detector_size = 0.5             # size of each probe element of the flat plate detector
detector_scale = 256 * 3        # flat panel detector detection scale
source_distance = 256 * 4       # distance from the ray source to the center of the object
dector_distance = 256 * 4       # distance from the detector to the center of the object

# Create cone-beam geometry and volume geometry
detector_num = int(detector_scale/detector_size)
angles = np.linspace(0, angle_scale, angle_num, False)
proj_geom = astra.create_proj_geom('cone', detector_size, detector_size, detector_num,
                                   detector_num, angles, source_distance, dector_distance)
vol_geom = astra.create_vol_geom(data_np.shape)

# Create cone-beam projection
proj_id, projdata = astra.create_sino3d_gpu(data=data_np, proj_geom=proj_geom, vol_geom=vol_geom)

# visualize projection
proj = np.transpose(projdata, (0, 2, 1))
proj = proj[np.arange(0, detector_num, int(detector_num / 100)), np.newaxis, :, :]
proj_tensor = torch.from_numpy(proj)
proj_grid = np.transpose(vutils.make_grid(proj_tensor, nrow=10, normalize=True), (1, 2, 0))
plt.imshow(proj_grid)
plt.axis("off")
plt.savefig('projection.png')
np.save('projection.npy', proj)

# Create a data object for the reconstruction
rec_id = astra.data3d.create('-vol', vol_geom)

# Set up the parameters and create FDK algorithm
cfg_fdk = astra.astra_dict('FDK_CUDA')
cfg_fdk['ProjectionDataId'] = proj_id
cfg_fdk['ReconstructionDataId'] = rec_id
alg_id = astra.algorithm.create(cfg_fdk)

# Run FDK algorithm and get results
astra.algorithm.run(alg_id, 1)
rec = astra.data3d.get(rec_id)
np.save('reconstruction.npy', rec)

# visualize reconstruction
rec_tensor = torch.from_numpy(rec[:, np.newaxis, :, :])
rec_grid = np.transpose(vutils.make_grid(rec_tensor, nrow=16), (1, 2, 0))
plt.figure(figsize=(82, 82))
plt.imshow(rec_grid)
plt.axis("off")
plt.savefig('reconstruction.png')

# compute matrix
rec_grid0 = np.array(rec_grid[:, :, 0])
ori_grid0 = np.array(ori_grid[:, :, 0])
rec_grid0 = rec_grid0 * (rec_grid0 <= 1) + (rec_grid0 > 1)
ori_grid0 = ori_grid0 * (ori_grid0 <= 1) + (ori_grid0 > 1)
ssim = compare_ssim(rec_grid0, ori_grid0)
psnr = compare_psnr(rec_grid0, ori_grid0)
mse = mean_squared_error(ori_grid0, rec_grid0)
print('[Evaluation] SSIM:{:.5f} PSNR:{:.3f} MSE:{:.3e}'.format(ssim, psnr, mse))

# Clean up memory
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)