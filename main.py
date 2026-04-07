import numpy as np
from PIL import Image


def load_grayscale(path):
    """Load image and convert to grayscale numpy array."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float64)


def convolve(image, kernel):
    """Element-by-element multiply and sum — the operation we walked through."""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad image with zeros so edges don't cause problems
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    output = np.zeros_like(image)
    for y in range(h):
        for x in range(w):
            # Grab the 3x3 patch (neighborhood)
            patch = padded[y : y + kh, x : x + kw]
            # Multiply element-by-element, sum everything up
            output[y, x] = np.sum(patch * kernel)

    return output


# ============================================================
# PREWITT OPERATOR (slides 14, 16)
# ============================================================

def prewitt(image):
    """
    Prewitt edge detection.

    Mx compares right column vs left column  -> vertical edges
    My compares top row vs bottom row        -> horizontal edges
    G  = sqrt(Gx^2 + Gy^2)                  -> combined edge strength
    """
    # The two kernels — exactly from the slides
    Mx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float64)

    My = np.array([[ 1,  1,  1],
                   [ 0,  0,  0],
                   [-1, -1, -1]], dtype=np.float64)

    Gx = convolve(image, Mx)
    Gy = convolve(image, My)

    # Combine both directions (Pythagorean theorem)
    G = np.sqrt(Gx ** 2 + Gy ** 2)

    return G


# ============================================================
# CANNY ALGORITHM (slides 19-26)
# ============================================================

def gaussian_kernel(size=5, sigma=1.4):
    """Step 1: Build a Gaussian filter to reduce noise (slide 22)."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def canny(image, low_thresh=0.1, high_thresh=0.3):
    """
    Canny edge detection — 4 steps from slide 20:
      1. Noise reduction (Gaussian filter)
      2. Gradient calculation (Sobel operator)
      3. Non-maximum suppression
      4. Hysteresis thresholding
    """
    h, w = image.shape

    # --- Step 1: Smooth with Gaussian filter (slide 22) ---
    gauss = gaussian_kernel(size=5, sigma=1.4)
    smoothed = convolve(image, gauss)

    # --- Step 2: Gradient with Sobel (slide 23) ---
    # Sobel kernels (slide 15)
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)

    Sy = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float64)

    Gx = convolve(smoothed, Sx)
    Gy = convolve(smoothed, Sy)

    # Gradient magnitude and direction (slide 23)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    theta = np.arctan2(Gy, Gx)  # direction of the edge

    # --- Step 3: Non-maximum suppression (slide 24) ---
    # Keep a pixel only if it's the strongest along the gradient direction
    nms = np.zeros_like(G)
    angle = np.rad2deg(theta) % 180  # map to 0-180

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = angle[y, x]

            # Pick two neighbors along the gradient direction
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                n1, n2 = G[y, x - 1], G[y, x + 1]        # horizontal
            elif 22.5 <= a < 67.5:
                n1, n2 = G[y - 1, x + 1], G[y + 1, x - 1]  # diagonal /
            elif 67.5 <= a < 112.5:
                n1, n2 = G[y - 1, x], G[y + 1, x]        # vertical
            else:
                n1, n2 = G[y - 1, x - 1], G[y + 1, x + 1]  # diagonal \

            # Keep only if it's the local maximum
            if G[y, x] >= n1 and G[y, x] >= n2:
                nms[y, x] = G[y, x]

    # --- Step 4: Hysteresis thresholding (slides 25-26) ---
    # Normalize to 0-1 range for thresholding
    if nms.max() > 0:
        nms = nms / nms.max()

    strong = nms >= high_thresh       # definitely an edge
    weak = (nms >= low_thresh) & (~strong)  # maybe an edge

    # A weak pixel becomes an edge if it's connected to a strong pixel
    result = np.zeros_like(nms, dtype=np.uint8)
    result[strong] = 255

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if weak[y, x]:
                # Check 8 neighbors for a strong pixel
                neighborhood = result[y - 1 : y + 2, x - 1 : x + 2]
                if neighborhood.max() == 255:
                    result[y, x] = 255

    return result

if __name__ == "__main__":
    for name in ["a", "a_noise", "b"]:
        image = load_grayscale(f"{name}.jpg")

        prewitt_result = prewitt(image)
        prewitt_norm = (prewitt_result / prewitt_result.max() * 255).astype(np.uint8)
        Image.fromarray(prewitt_norm).save(f"{name}_prewitt.jpg")

        canny_result = canny(image, low_thresh=0.1, high_thresh=0.3)
        Image.fromarray(canny_result).save(f"{name}_canny.jpg")

        print(f"{name} done")