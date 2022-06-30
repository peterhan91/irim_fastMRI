import numpy as np
import torch.fft as fft


def get_cartesian_mask(shape, n_keep=30):
  # shape [Tuple]: (H, W)
  size = shape[0]
  center_fraction = n_keep / 1000
  acceleration = size / n_keep

  num_rows, num_cols = shape[0], shape[1]
  num_low_freqs = int(round(num_cols * center_fraction))
  mask = np.zeros((num_rows, num_cols), dtype=np.float32)
  pad = (num_cols - num_low_freqs + 1) // 2
  mask[:, pad: pad + num_low_freqs] = True
  adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
      num_low_freqs * acceleration - num_cols
  )
  offset = round(adjusted_accel) // 2
  accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
  accel_samples = np.around(accel_samples).astype(np.uint32)
  mask[:, accel_samples] = True
  return mask
  

def get_kspace(img, axes):
  # img should be a complex tensor
  shape = img.shape[axes[0]]
  return fft.fftshift(
    fft.fftn(fft.ifftshift(
      img, dim=axes
    ), dim=axes),
    dim=axes
  ) / shape


def kspace_to_image(kspace, axes):
  shape = kspace.shape[axes[0]]
  return fft.fftshift(
    fft.ifftn(fft.ifftshift(
      kspace, dim=axes
    ), dim=axes),
    dim=axes
  ) * shape