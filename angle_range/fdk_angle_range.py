import h5py
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch
import astra
from skimage.measure import compare_ssim, compare_psnr
from sklearn.metrics import mean_squared_error

data_path = './wuguiP3D.mat'
data = h5py.File(data_path)['reference']
data_np = np.array(data)
data_tensor = torch.from_numpy(data_np)

# visualize reference
ori_grid = np.transpose(vutils.make_grid(data_tensor.resize(256, 1, 256, 256), nrow=16), (1, 2, 0))

angle_num = 1000
detector_num = 1424
det_stack = np.zeros([21, 1424, 1424])
proj_stack = np.zeros([21, 1424, 1424])
rec_stack2 = np.zeros([21, 256, 256])
SSIM_list = []
PSNR_list = []
MSE_list = []
i = 0

for angle_range in np.arange(0, 10*np.pi+0.1, np.pi/2):
    # Create a 3D cone beam geometry structure.
    angles = np.linspace(0, angle_range, angle_num, False)
    proj_geom = astra.create_proj_geom('cone', 0.5, 0.5, detector_num, detector_num, angles, 800, 800)

    # Create a volume geometry structure
    vol_sz = data_np.shape
    vol_geom = astra.create_vol_geom(vol_sz)

    # Create cone-beam projection
    proj_id, projdata = astra.create_sino3d_gpu(data=data_np, proj_geom=proj_geom, vol_geom=vol_geom)

    # store detector data
    det = np.transpose(projdata, (1, 0, 2))
    det_stack[i, :detector_num, :detector_num] = det[int(angle_num / 2), :, :]

    # store sinogram
    proj = np.transpose(projdata, (0, 2, 1))
    proj_stack[i, :detector_num, :angle_num] = proj[int(detector_num / 2), :, :]

    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up the parameters and create FDK algorithm
    cfg_fdk = astra.astra_dict('FDK_CUDA')
    cfg_fdk['ProjectionDataId'] = proj_id
    cfg_fdk['ReconstructionDataId'] = rec_id
    alg_id = astra.algorithm.create(cfg_fdk)

    # Run and get results
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
proj_tensor = torch.from_numpy(det_stack[:, np.newaxis, :, :])
proj_grid = np.transpose(vutils.make_grid(proj_tensor, nrow=5, normalize=True), (1, 2, 0))
plt.imshow(proj_grid)
plt.axis("off")
plt.savefig('angle_range_det.png', dpi=1000)

# visualize sinogram
sino_tensor = torch.from_numpy(proj_stack[:, np.newaxis, :, :])
sino_grid = np.transpose(vutils.make_grid(sino_tensor, nrow=5, normalize=True), (1, 2, 0))
plt.imshow(sino_grid)
plt.axis("off")
plt.savefig('angle_range_proj', dpi=500)

# visualize reconstruction
rec_tensor2 = torch.from_numpy(rec_stack2[:, np.newaxis, :, :])
rec_grid2 = np.transpose(vutils.make_grid(rec_tensor2, nrow=5), (1, 2, 0))
plt.figure(figsize=(82, 82))
plt.imshow(rec_grid2)
plt.axis("off")
plt.savefig('angle_range_rec.png')

# visualize matrix
x = np.arange(0, 10*np.pi+0.1, np.pi/2)
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
plt.xlabel('angle_num range')
plt.ylabel('MSE')
plt.ylim(ymax=0.14)
for a, b in zip(x, MSE_list):
    plt.text(a, b, '%.1e' % b, ha='right', va='bottom', fontsize=4)
plt.savefig('angle_range_matrix.png', dpi=300)