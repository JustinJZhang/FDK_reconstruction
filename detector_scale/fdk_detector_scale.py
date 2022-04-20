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
data_tensor = torch.from_numpy(data_np)
ori_grid = np.transpose(vutils.make_grid(data_tensor.resize(256, 1, 256, 256), nrow=16), (1, 2, 0))
# Prepare
angle_num = 1000
proj_stack = np.zeros([30, 2000, 2000])
sino_stack = np.zeros([30, 2000, 2000])
rec_stack2 = np.zeros([30, 256, 256])
SSIM_list = []
PSNR_list = []
MSE_list = []
i = 0

for detector_num in range(1, 1500, 50):
    # Create a 3D cone beam geometry structure.
    angles = np.linspace(0, 2 * np.pi, angle_num, False)
    proj_geom = astra.create_proj_geom('cone', 0.5, 0.5, detector_num, detector_num, angles, 800, 800)

    # Create a volume geometry structure
    vol_sz = data_np.shape
    vol_geom = astra.create_vol_geom(vol_sz)

    # Create cone-beam projection
    proj_id, projdata = astra.create_sino3d_gpu(data=data_np, proj_geom=proj_geom, vol_geom=vol_geom)

    # store projection
    proj = np.transpose(projdata, (1, 0, 2))
    proj_stack[i, :detector_num, :detector_num] = proj[int(angle_num / 2), :, :]
    sino = np.transpose(projdata, (0, 2, 1))
    sino_stack[i, :detector_num, :angle_num] = sino[int(detector_num / 2), :, :]

    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up the parameters and create FDK algorithm
    cfg_fdk = astra.astra_dict('FDK_CUDA')
    cfg_fdk['ProjectionDataId'] = proj_id
    cfg_fdk['ReconstructionDataId'] = rec_id
    alg_id = astra.algorithm.create(cfg_fdk)

    # Run algorithm and get results
    astra.algorithm.run(alg_id, 1)
    rec = astra.data3d.get(rec_id)
    rec_tensor = torch.from_numpy(rec)
    rec_grid = np.transpose(vutils.make_grid(rec_tensor.resize(256, 1, 256, 256), nrow=16), (1, 2, 0))

    # store reconstruction
    rec_stack2[i, :, :] = rec[190, :, :]

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
proj_tensor = torch.from_numpy(proj_stack[:, np.newaxis, :, :])
proj_grid = np.transpose(vutils.make_grid(proj_tensor, nrow=5, normalize=True), (1, 2, 0))
plt.imshow(proj_grid)
plt.axis("off")
plt.savefig('detector_scale_det.png', dpi=1000)

# visualize sinogram
sino_tensor = torch.from_numpy(sino_stack[:, np.newaxis, :, :])
sino_grid = np.transpose(vutils.make_grid(sino_tensor, nrow=5, normalize=True), (1, 2, 0))
plt.imshow(sino_grid)
plt.axis("off")
plt.savefig('detector_scale_proj.png', dpi=500)

# visualize reconstruction
rec_tensor2 = torch.from_numpy(rec_stack2[:, np.newaxis, :, :])
rec_grid2 = np.transpose(vutils.make_grid(rec_tensor2, nrow=5), (1, 2, 0))
plt.figure(figsize=(82, 82))
plt.imshow(rec_grid2)
plt.axis("off")
plt.savefig('detector_scale_rec.png')

# visualize matrix
x = range(1, 1500, 50)
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(x, SSIM_list, linewidth=2, color='blue', marker='P')
plt.ylabel('SSIM')
plt.ylim(ymax=1.2)
for a, b in zip(x, SSIM_list):
    plt.text(a, b, '%.3f' % b, ha='right', va='bottom', fontsize=4)

plt.subplot(3, 1, 2)
plt.plot(x, PSNR_list, linewidth=2, color='orange', marker='P')
plt.ylabel('PSNR')
plt.ylim(ymax=60)
for a, b in zip(x, PSNR_list):
    plt.text(a, b, '%.3f' % b, ha='right', va='bottom', fontsize=4)

plt.subplot(3, 1, 3)
plt.plot(x, MSE_list, linewidth=2, color='limegreen', marker='P')
plt.xlabel('detector scale')
plt.ylabel('MSE')
plt.ylim(ymax=0.14)
for a, b in zip(x, MSE_list):
    plt.text(a, b, '%.1e' % b, ha='right', va='bottom', fontsize=3)
plt.savefig('detector_scale_matrix.png', dpi=300)