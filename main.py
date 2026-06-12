import numpy as np
import matplotlib.pyplot as plt

from dataset_loader import ARADDataset


# =====================================================
# Spatial FFT per band
# =====================================================

def spatial_fft(volume):
    """
    volume: H x W x C

    returns:
        H x W x C
    """

    H, W, C = volume.shape

    fft_mag = np.zeros(
        (H, W, C),
        dtype=np.float32
    )

    for c in range(C):

        F = np.fft.fft2(
            volume[:, :, c]
        )

        F = np.fft.fftshift(F)

        fft_mag[:, :, c] = np.log1p(
            np.abs(F)
        )

    return log(fft_mag)


# =====================================================
# RGB FFT visualization
# =====================================================

def show_rgb_fft(rgb_fft):

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    names = ["Red", "Green", "Blue"]

    for i in range(3):

        ax[i].imshow(
            rgb_fft[:, :, i]
        )

        ax[i].set_title(
            names[i]
        )

        ax[i].axis("off")

    plt.tight_layout()

    plt.savefig(
        "rgb_fft.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# HSI FFT visualization
# =====================================================

def show_hsi_fft(hsi_fft):

    fig, ax = plt.subplots(
        4,
        8,
        figsize=(20, 10)
    )

    ax = ax.flatten()

    for band in range(31):

        ax[band].imshow(
            hsi_fft[:, :, band]
        )

        ax[band].set_title(
            f"Band {band}"
        )

        ax[band].axis("off")

    for i in range(31, len(ax)):
        ax[i].axis("off")

    plt.tight_layout()

    plt.savefig(
        "hsi_fft.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# FFT Difference Visualization
# =====================================================

def show_fft_difference(rgb_fft, hsi_fft):

    rgb_mean = rgb_fft.mean(
        axis=2
    )

    fig, ax = plt.subplots(
        4,
        8,
        figsize=(20, 10)
    )

    ax = ax.flatten()

    for band in range(31):

        diff = np.abs(
            hsi_fft[:, :, band]
            - rgb_mean
        )

        ax[band].imshow(diff)

        ax[band].set_title(
            f"Band {band}"
        )

        ax[band].axis("off")

    for i in range(31, len(ax)):
        ax[i].axis("off")

    plt.tight_layout()

    plt.savefig(
        "fft_difference.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


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

rgb_np = rgb.permute(
    1,
    2,
    0
).numpy()

hsi_np = hsi.permute(
    1,
    2,
    0
).numpy()

rgb_fft = spatial_fft(
    rgb_np
)

hsi_fft = spatial_fft(
    hsi_np
)

print("FFT created")

show_rgb_fft(
    rgb_fft
)

show_hsi_fft(
    hsi_fft
)

show_fft_difference(
    rgb_fft,
    hsi_fft
)

print("Saved:")
print("  rgb_fft.png")
print("  hsi_fft.png")
print("  fft_difference.png")
