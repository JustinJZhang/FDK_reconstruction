import h5py
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch
import astra
from skimage.measure import compare_ssim, compare_psnr
from sklearn.metrics import mean_squared_error

data_path = '../wuguiP3D.mat'
data = h5py.File(data_path)['reference']
data_np = np.array(data)
data_tensor = torch.from_numpy(data_np)
ori_grid = np.transpose(vutils.make_grid(data_tensor.resize(256, 1, 256, 256), nrow=16), (1, 2, 0))

# Create a 3D cone beam geometry structure.
angle_num = 500
detector_num = 356 * 2
angles = np.linspace(0, 2 * np.pi, angle_num, False)
proj_geom = astra.create_proj_geom('cone', 1, 1, detector_num, detector_num, angles, 800, 800)

# Create a volume geometry structure
vol_sz = data_np.shape
vol_geom = astra.create_vol_geom(vol_sz)

# Create cone-beam projection
proj_id, projdata = astra.create_sino3d_gpu(data=data_np, proj_geom=proj_geom, vol_geom=vol_geom)

# Create object for the reconstruction
rec_id = astra.data3d.create('-vol', vol_geom)
rec_stack1 = np.zeros([20, 256, 256])
rec_stack2 = np.zeros([20, 256, 256])
SSIM_list = []
PSNR_list = []
MSE_list = []

# FDK algorithm
cfg_fdk = astra.astra_dict('FDK_CUDA')
cfg_fdk['ProjectionDataId'] = proj_id
cfg_fdk['ReconstructionDataId'] = rec_id
alg_id = astra.algorithm.create(cfg_fdk)

astra.algorithm.run(alg_id, 150)
rec = astra.data3d.get(rec_id)
rec_stack1[0, :, :] = rec[:, :, 128]
rec_stack2[0, :, :] = rec[190, :, :]

rec_tensor = torch.from_numpy(rec[:, np.newaxis, :, :])
rec_grid = np.transpose(vutils.make_grid(rec_tensor, nrow=16), (1, 2, 0))
rec_grid0 = np.array(rec_grid[:, :, 0])
ori_grid0 = np.array(ori_grid[:, :, 0])
rec_grid0 = rec_grid0 * (rec_grid0 <= 1) + (rec_grid0 > 1)
ori_grid0 = ori_grid0 * (ori_grid0 <= 1) + (ori_grid0 > 1)
SSIM_list.append(compare_ssim(ori_grid0, rec_grid0))
PSNR_list.append(compare_psnr(ori_grid0, rec_grid0))
MSE_list.append(mean_squared_error(ori_grid0, rec_grid0))

# SIRT algorithm
cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id
alg_id = astra.algorithm.create(cfg)

astra.algorithm.run(alg_id, 150)
rec = astra.data3d.get(rec_id)
rec_stack1[1, :, :] = rec[:, :, 128]
rec_stack2[1, :, :] = rec[190, :, :]

rec_tensor = torch.from_numpy(rec[:, np.newaxis, :, :])
rec_grid = np.transpose(vutils.make_grid(rec_tensor, nrow=16), (1, 2, 0))
rec_grid0 = np.array(rec_grid[:, :, 0])
ori_grid0 = np.array(ori_grid[:, :, 0])
rec_grid0 = rec_grid0 * (rec_grid0 <= 1) + (rec_grid0 > 1)
ori_grid0 = ori_grid0 * (ori_grid0 <= 1) + (ori_grid0 > 1)
SSIM_list.append(compare_ssim(ori_grid0, rec_grid0))
PSNR_list.append(compare_psnr(ori_grid0, rec_grid0))
MSE_list.append(mean_squared_error(ori_grid0, rec_grid0))

# CGLS algorithm
cfg = astra.astra_dict('CGLS3D_CUDA')
cfg['ProjectionDataId'] = proj_id
cfg['ReconstructionDataId'] = rec_id
alg_id = astra.algorithm.create(cfg)

astra.algorithm.run(alg_id, 50)
rec = astra.data3d.get(rec_id)
rec_stack1[2, :, :] = rec[:, :, 128]
rec_stack2[2, :, :] = rec[190, :, :]

rec_tensor = torch.from_numpy(rec[:, np.newaxis, :, :])
rec_grid = np.transpose(vutils.make_grid(rec_tensor, nrow=16), (1, 2, 0))
rec_grid0 = np.array(rec_grid[:, :, 0])
ori_grid0 = np.array(ori_grid[:, :, 0])
rec_grid0 = rec_grid0 * (rec_grid0 <= 1) + (rec_grid0 > 1)
ori_grid0 = ori_grid0 * (ori_grid0 <= 1) + (ori_grid0 > 1)
SSIM_list.append(compare_ssim(ori_grid0, rec_grid0))
PSNR_list.append(compare_psnr(ori_grid0, rec_grid0))
MSE_list.append(mean_squared_error(ori_grid0, rec_grid0))

astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)

# visualize result
fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(rec_stack1[0, :, :])
ax[0, 0].set_title('FDK')
ax[0, 0].axis("off")
ax[1, 0].imshow(rec_stack2[0, :, :])
ax[1, 0].axis("off")
ax[0, 1].imshow(rec_stack1[1, :, :])
ax[0, 1].set_title('SIRT')
ax[0, 1].axis("off")
ax[1, 1].imshow(rec_stack2[1, :, :])
ax[1, 1].axis("off")
ax[0, 2].imshow(rec_stack1[2, :, :])
ax[0, 2].set_title('CGLS')
ax[0, 2].axis("off")
ax[1, 2].imshow(rec_stack2[2, :, :])
ax[1, 2].axis("off")
fig.tight_layout()
plt.savefig('compare_rec.png', dpi=800)


print('SSIM:', SSIM_list)
print('PSNR:', PSNR_list)
print('MSE:', MSE_list)

# Result：
# SSIM: [FDK：0.985964341242978, SIRT：0.9989802517952734, CGLS：0.8786340828459018]
# PSNR: [FDK：51.96287055620448, SIRT：58.32893682764286, CGLS：37.47546879884896]
# MSE: [FDK：2.5454990289945224e-05, SIRT：5.877143685599554e-06, CGLS：0.0007153409900464519]

# Draw bar charts for matrix
# labels = ['SSIM', 'PSNR', 'MSE']
# x = np.arange(len(labels))
# FDK = [SSIM_list[0], PSNR_list[0], MSE_list[0] + 1e-5]
# SIRT = [SSIM_list[1], PSNR_list[1], MSE_list[1] + 1e-5]
# CGLS = [SSIM_list[2], PSNR_list[2], MSE_list[2] + 1e-5]
# plt.figure(figsize=(10, 10))
# width = 0.25
# plt.bar(x - width, FDK, width, label='FDK')
# plt.bar(x, SIRT, width, label='SIRT')
# plt.bar(x + width, CGLS, width, label='CGLS')
# for x1, y1 in zip(x, FDK):
#     plt.text(x1 - width, y1 + 0.05, '%.2f' % y1, ha='center', va='bottom')
# for x2, y2 in zip(x, SIRT):
#     plt.text(x2, y2 + 0.05, '%.2f' % y2, ha='center', va='bottom')
# for x3, y3 in zip(x, CGLS):
#     plt.text(x3 + width, y3 + 0.05, '%.2f' % y3, ha='center', va='bottom')
# plt.ylabel('Scores')
# plt.xticks(x, labels=labels)
# plt.legend()
# plt.savefig('compare_matrix.png')
