import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Load HSI (.mat)
# ============================================================
def load_hsi(mat_path):

    data = sio.loadmat(mat_path)

    # Change this according to your dataset
    for k in data.keys():
        if not k.startswith("__"):
            hsi = data[k]
            break

    hsi = hsi.astype(np.float32)

    # normalize
    hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min() + 1e-8)

    return hsi


# ============================================================
# Load RGB
# ============================================================
def load_rgb(img_path):

    rgb = np.array(Image.open(img_path).convert("RGB"))

    rgb = rgb.astype(np.float32) / 255.0

    return rgb


# ============================================================
# Compute 3D FFT
# ============================================================
def compute_fft3(volume):

    fft = np.fft.fftn(volume)

    fft_shift = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shift)

    log_mag = np.log1p(magnitude)

    return log_mag


# ============================================================
# Visualize center slices
# ============================================================
def visualize_fft3(fft_mag, title):

    H, W, C = fft_mag.shape

    h_mid = H // 2
    w_mid = W // 2
    c_mid = C // 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(fft_mag[h_mid, :, :], aspect='auto')
    ax[0].set_title("Center H Slice")

    ax[1].imshow(fft_mag[:, w_mid, :], aspect='auto')
    ax[1].set_title("Center W Slice")

    ax[2].imshow(fft_mag[:, :, c_mid])
    ax[2].set_title("Center Spectral Slice")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Frequency energy statistics
# ============================================================
def frequency_statistics(fft_mag, name):

    energy = np.sum(fft_mag ** 2)

    H, W, C = fft_mag.shape

    center = np.array([H//2, W//2, C//2])

    coords = np.indices((H, W, C)).transpose(1,2,3,0)

    dist = np.linalg.norm(coords - center, axis=-1)

    radius = min(H, W, C) // 4

    low_freq_energy = np.sum(
        fft_mag[dist < radius] ** 2
    )

    high_freq_energy = energy - low_freq_energy

    print(f"\n{name}")
    print("-"*50)
    print("Total Energy      :", energy)
    print("Low Frequency %   :", 100*low_freq_energy/energy)
    print("High Frequency %  :", 100*high_freq_energy/energy)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    hsi_path = "sample.mat"
    rgb_path = "sample.png"

    hsi = load_hsi(hsi_path)
    rgb = load_rgb(rgb_path)

    print("HSI Shape :", hsi.shape)
    print("RGB Shape :", rgb.shape)

    fft_hsi = compute_fft3(hsi)
    fft_rgb = compute_fft3(rgb)

    visualize_fft3(fft_hsi, "HSI 3D FFT")

    visualize_fft3(fft_rgb, "RGB 3D FFT")

    frequency_statistics(fft_hsi, "HSI")

    frequency_statistics(fft_rgb, "RGB")
