import h5py
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch
import astra
from skimage.measure import compare_ssim, compare_psnr
from sklearn.metrics import mean_squared_error

# Load data prepare
data_path = '../wuguiP3D.mat'
data = h5py.File(data_path)['reference']
data_np = np.array(data)
data_tensor = torch.from_numpy(data_np)
ori_grid = np.transpose(vutils.make_grid(data_tensor.resize(256, 1, 256, 256), nrow=16), (1, 2, 0))
# Prepare
detector_num = 1424
det_stack = np.zeros([20, detector_num, detector_num])
proj_stack = np.zeros([20, detector_num, detector_num])
rec_stack = np.zeros([20, 256, 256])
SSIM_list = []
PSNR_list = []
MSE_list = []
i = 0

for angle_num in range(1, 1000, 50):
    # Create cone beam geometry structure.
    angles = np.linspace(0, 2 * np.pi, angle_num, False)
    proj_geom = astra.create_proj_geom('cone', 0.5, 0.5, detector_num, detector_num, angles, 800, 800)

    # Create volume geometry structure
    vol_sz = data_np.shape
    vol_geom = astra.create_vol_geom(vol_sz)

    # Create cone-beam projection
    proj_id, projdata = astra.create_sino3d_gpu(data=data_np, proj_geom=proj_geom, vol_geom=vol_geom)

    # store detector data and projection
    det = np.transpose(projdata, (1, 0, 2))
    det_stack[i, :, :] = det[int(angle_num / 2), :, :]
    proj = np.transpose(projdata, (0, 2, 1))
    proj_stack[i, :, :angle_num] = proj[int(detector_num / 2), :, :]

    # Create data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up the parameters and create FDK algorithm
    cfg_fdk = astra.astra_dict('FDK_CUDA')
    cfg_fdk['ProjectionDataId'] = proj_id
    cfg_fdk['ReconstructionDataId'] = rec_id
    alg_id = astra.algorithm.create(cfg_fdk)

    # Run algorithm and store results
    astra.algorithm.run(alg_id, 150)
    rec = astra.data3d.get(rec_id)
    rec_tensor = torch.from_numpy(rec)
    rec_grid = np.transpose(vutils.make_grid(rec_tensor.resize(256, 1, 256, 256), nrow=16), (1, 2, 0))
    rec_stack[i, :, :] = rec[128, :, :]

    # compute matrix
    rec_grid0 = np.array(rec_grid[:, :, 0])
    ori_grid0 = np.array(ori_grid[:, :, 0])
    rec_grid0 = rec_grid0 * (rec_grid0 <= 1) + (rec_grid0 > 1)
    ori_grid0 = ori_grid0 * (ori_grid0 <= 1) + (ori_grid0 > 1)
    SSIM_list.append(compare_ssim(ori_grid0, rec_grid0))
    PSNR_list.append(compare_psnr(ori_grid0, rec_grid0))
    MSE_list.append(mean_squared_error(ori_grid0, rec_grid0))

    # Clean up memory
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
    i = i + 1

# visualize detector data
det_tensor = torch.from_numpy(det_stack[:, np.newaxis, :, :])
det_grid = np.transpose(vutils.make_grid(det_tensor, nrow=5, normalize=True), (1, 2, 0))
plt.imshow(det_grid)
plt.axis("off")
plt.savefig('angle_num_det.png', dpi=500)

# visualize projection
proj_tensor = torch.from_numpy(proj_stack[:, np.newaxis, :, :])
proj_grid = np.transpose(vutils.make_grid(proj_tensor, nrow=5, normalize=True), (1, 2, 0))
plt.imshow(proj_grid)
plt.axis("off")
plt.savefig('angle_num_proj.png', dpi=500)

# visualize reconstruction
rec_tensor = torch.from_numpy(rec_stack[:, np.newaxis, :, :])
rec_grid = np.transpose(vutils.make_grid(rec_tensor, nrow=5), (1, 2, 0))
plt.figure(figsize=(82, 82))
plt.imshow(rec_grid)
plt.axis("off")
plt.savefig('angle_num_rec.png')

# visualize matrix
x = range(0, 1000, 50)
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(x, SSIM_list, linewidth=2, color='blue', marker='P')
plt.ylabel('SSIM')
plt.ylim(ymax=1.2)
for a, b in zip(x, SSIM_list):
    plt.text(a, b, '%.3f' % b, ha='right', va='bottom', fontsize=6)

plt.subplot(3, 1, 2)
plt.plot(x, PSNR_list, linewidth=2, color='orange', marker='P')
plt.ylabel('PSNR')
plt.ylim(ymax=60)
for a, b in zip(x, PSNR_list):
    plt.text(a, b, '%.3f' % b, ha='right', va='bottom', fontsize=6)

plt.subplot(3, 1, 3)
plt.plot(x, MSE_list, linewidth=2, color='limegreen', marker='P')
plt.xlabel('angle_num')
plt.ylabel('MSE')
plt.ylim(ymax=0.14)
for a, b in zip(x, MSE_list):
    plt.text(a, b, '%.1e' % b, ha='right', va='bottom', fontsize=4)
plt.savefig('angle_num_matrix.png', dpi=300)