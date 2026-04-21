import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

# -----------------------------
# Functions
# -----------------------------

# Simple random affine warp
def warp_patch(patch):
    _, H, W = patch.shape

    # random affine parameters
    angle = random.uniform(-30, 30)  # rotation
    translate_x = random.uniform(-0.1, 0.1)
    translate_y = random.uniform(-0.1, 0.1)
    scale = random.uniform(0.8, 1.2)

    theta = torch.tensor([
        [
            scale * torch.cos(torch.tensor(angle)),
            -scale * torch.sin(torch.tensor(angle)),
            translate_x
        ],
        [
            scale * torch.sin(torch.tensor(angle)),
            scale * torch.cos(torch.tensor(angle)),
            translate_y
        ]
    ])

    theta = theta.unsqueeze(0)

    grid = F.affine_grid(theta, patch.unsqueeze(0).size(), align_corners=False)
    warped = F.grid_sample(patch.unsqueeze(0), grid, align_corners=False)

    return warped.squeeze(0)

#Gaussian Blur
def gaussian_blur(patch, kernel_size=2, sigma=0.5):
    channels, H, W = patch.shape

    # create 1D Gaussian kernel
    ax = torch.arange(kernel_size) - kernel_size // 2
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss = gauss / gauss.sum()

    kernel_2d = gauss[:, None] @ gauss[None, :]
    kernel_2d = kernel_2d.to(patch.device)

    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)

    padding = kernel_size // 2

    blurred = F.conv2d(
        patch.unsqueeze(0),
        kernel,
        padding=padding,
        groups=channels
    )

    return blurred.squeeze(0)

#Brightness n Contrast
def adjust_brightness(patch, brightness=1.4, contrast=2.0):
    """
    brightness: multiplies intensity
    contrast: scales deviation from mean
    """
    mean = patch.mean()

    out = (patch - mean) * contrast + mean
    out = out * brightness

    return torch.clamp(out, 0, 1)

#Compression Artifacts
def jpeg_artifacts(patch, downscale=0.2):
    """
    Simulates compression artifacts using downsample/upsample
    """
    C, H, W = patch.shape

    # downsample
    small = F.interpolate(
        patch.unsqueeze(0),
        scale_factor=downscale,
        mode="bilinear",
        align_corners=False
    )

    # upsample back
    restored = F.interpolate(
        small,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    # optional quantization noise (simulates compression loss)
    noise = torch.randn_like(restored) * 0.01

    out = restored + noise

    return torch.clamp(out.squeeze(0), 0, 1)

#smear
def elastic_smear(patch, alpha=5.0, sigma=8.0):
    """
    Smooth curvy distortion using a random displacement field.
    """
    C, H, W = patch.shape

    device = patch.device

    # -----------------------------
    # random displacement field
    # -----------------------------
    dx = torch.randn(1, H, W, device=device)
    dy = torch.randn(1, H, W, device=device)

    # smooth it (Gaussian-like blur via conv)
    def blur(x, k=9):
        kernel = torch.ones(1, 1, k, k, device=device) / (k * k)
        return F.conv2d(x.unsqueeze(0), kernel, padding=k//2).squeeze(0)

    dx = blur(dx)
    dy = blur(dy)

    # scale displacement
    dx = dx * alpha
    dy = dy * alpha

    # -----------------------------
    # create mesh grid
    # -----------------------------
    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)

    # normalize displacement to [-1,1]
    dx = dx.squeeze(0) / W
    dy = dy.squeeze(0) / H

    grid[..., 0] += dx
    grid[..., 1] += dy

    # -----------------------------
    # sample image
    # -----------------------------
    warped = F.grid_sample(
        patch.unsqueeze(0),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )

    return warped.squeeze(0)


# -----------------------------
# Create patch
# -----------------------------
patch_size = 128
patch = torch.rand(3, patch_size, patch_size)

#patch = warp_patch(patch)

patch = gaussian_blur(patch)

patch = adjust_brightness(patch)

patch = jpeg_artifacts(patch)

patch = elastic_smear(patch)



























# -----------------------------
# Convert to display format (H, W, C)
# -----------------------------
patch_img = patch.permute(1, 2, 0).numpy()

# -----------------------------
# Display patch
# -----------------------------
plt.imshow(patch_img)
plt.title("Initial Random Adversarial Patch")
plt.axis("off")
plt.show()