import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset_loader import ARADDataset


# =====================================================
# FFT utilities
# =====================================================

def fft3(volume):

    volume = volume.astype(np.float32)

    F = np.fft.fftn(volume)

    F = np.fft.fftshift(F)

    mag = np.abs(F)

    log_mag = np.log1p(mag)

    return log_mag


# =====================================================
# Central slices
# =====================================================

def show_fft_slices(fft_mag, title):

    H, W, C = fft_mag.shape

    h = H // 2
    w = W // 2
    c = C // 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(
        fft_mag[h],
        aspect="auto"
    )
    ax[0].set_title("X Slice")

    ax[1].imshow(
        fft_mag[:, w, :],
        aspect="auto"
    )
    ax[1].set_title("Y Slice")

    ax[2].imshow(
        fft_mag[:, :, c]
    )
    ax[2].set_title("Spectral Slice")

    plt.suptitle(title)

    plt.savefig("plot.png")
    plt.close()


# =====================================================
# Radial spectrum
# =====================================================

def radial_power_spectrum(fft_mag):

    H, W, C = fft_mag.shape

    center = np.array(
        [H//2, W//2, C//2]
    )

    coords = np.indices(
        (H, W, C)
    ).transpose(
        1, 2, 3, 0
    )

    dist = np.linalg.norm(
        coords - center,
        axis=-1
    )

    dist = dist.astype(int)

    power = fft_mag ** 2

    max_r = dist.max()

    radial = np.zeros(max_r + 1)

    for r in range(max_r + 1):

        mask = dist == r

        if np.any(mask):

            radial[r] = power[mask].mean()

    return radial


# =====================================================
# Main
# =====================================================

dataset = ARADDataset(
    root_dir="data",
    train=True,
    download=True
)

sample_idx = 0

rgb, hsi = dataset[sample_idx]

print("RGB:", rgb.shape)
print("HSI:", hsi.shape)

# ------------------------------------
# Convert to HWC
# ------------------------------------

rgb_np = rgb.permute(
    1, 2, 0
).numpy()

hsi_np = hsi.permute(
    1, 2, 0
).numpy()

# ------------------------------------
# FFT
# ------------------------------------

rgb_fft = fft3(rgb_np)

hsi_fft = fft3(hsi_np)
print("FFT created")
# ------------------------------------
# Visualize
# ------------------------------------

show_fft_slices(
    rgb_fft,
    "RGB 3D FFT"
)

show_fft_slices(
    hsi_fft,
    "HSI 3D FFT"
)

# ------------------------------------
# Radial power spectrum
# ------------------------------------

rgb_radial = radial_power_spectrum(
    rgb_fft
)

hsi_radial = radial_power_spectrum(
    hsi_fft
)

plt.figure(figsize=(8, 5))

plt.plot(
    rgb_radial / rgb_radial.max(),
    label="RGB"
)

plt.plot(
    hsi_radial / hsi_radial.max(),
    label="HSI"
)

plt.xlabel("Radius")

plt.ylabel("Normalized Power")

plt.title(
    "3D Frequency Distribution"
)

plt.legend()

plt.grid()

plt.show()
