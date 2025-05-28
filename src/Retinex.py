import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import match_histograms


def single_scale_retinex(img, sigma):
    #blur = cv2.GaussianBlur(img, (0, 0), sigma)
    #return np.log1p(img) - np.log1p(blur)
    return np.log10(img + 1e-6) - np.log10(gaussian_filter(img, sigma=sigma) + 1e-6)

def multi_scale_retinex(img, scales):
    #retinex = np.zeros_like(img, dtype=np.float32)
    #for sigma in scales:
    #    retinex += single_scale_retinex(img, sigma)
    #return retinex / len(scales)
    return np.mean([single_scale_retinex(img, sigma) for sigma in scales], axis=0)

def simplest_color_balance(img, low_clip=1, high_clip=99):
    result = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[2]):
        low_val = np.percentile(img[:, :, i], low_clip)
        high_val = np.percentile(img[:, :, i], high_clip)
        result[:, :, i] = np.clip((img[:, :, i] - low_val) * 255.0 / (high_val - low_val), 0, 255)
    return result

def msrcr(img, scales=[15, 80, 250], gain=1.0, offset=0):
    img = img.astype(np.float32) + 1.0
    retinex = multi_scale_retinex(img, scales)
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = np.log10(125.0 * img / img_sum)
    msrcr_result = gain * (retinex * color_restoration + offset)
    
    for i in range(msrcr_result.shape[2]):
        msrcr_result[:, :, i] = cv2.normalize(msrcr_result[:, :, i], None, 0, 255, cv2.NORM_MINMAX)
    msrcr_result = np.uint8(msrcr_result)
    return simplest_color_balance(msrcr_result)

def msrcr_face_safe(img, scales=[15, 80], gain=0.8, offset=0, gamma=1.2):
    img = img.astype(np.float32) + 1.0
    retinex = multi_scale_retinex(img, scales)
    
    img_sum = np.sum(img, axis=2, keepdims=True)
    img_sum[img_sum == 0] = 1.0  # Avoid divide-by-zero
    color_restoration = np.log10(125.0 * img / img_sum)

    # Less aggressive Retinex combination
    msrcr_result = gain * (retinex * color_restoration + offset)

    # Normalize to [0, 1] before gamma correction
    msrcr_result -= msrcr_result.min()
    msrcr_result /= (msrcr_result.max() + 1e-6)

    # Gamma correction for perceptual brightness
    msrcr_result = np.power(msrcr_result, 1.0 / gamma)

    # Scale to 0-255 and apply color balance
    msrcr_result = (msrcr_result * 255).astype(np.uint8)
    msrcr_result = simplest_color_balance(msrcr_result, low_clip=1, high_clip=99)

    return msrcr_result

def msrcr_face_safe2(img, scales=[15, 80], gain=0.5, offset=0, gamma=1.6, blend_factor=0.4):
    """
    Perform MSRCR with brightness sharing and natural skin tone preservation.

    Args:
        img (np.ndarray): Input image (H, W, 3), dtype=np.uint8.
        scales (list): Gaussian blur scales for Retinex.
        gain (float): Contrast gain.
        offset (float): Contrast offset.
        gamma (float): Gamma correction factor.
        blend_factor (float): How much to blend the result with the original image to restore color. Range [0, 1].

    Returns:
        np.ndarray: Processed image with preserved skin tone.
    """
    original_img = img.copy()
    img = img.astype(np.float32) + 1.0

    # Retinex
    retinex = multi_scale_retinex(img, scales)
    retinex = np.clip(retinex, -2.0, 2.0)
    retinex = cv2.GaussianBlur(retinex, (3, 3), 0)

    # Color restoration
    img_sum = np.sum(img, axis=2, keepdims=True)
    img_sum[img_sum == 0] = 1.0
    color_restoration = np.log10(125.0 * img / img_sum)

    msrcr = gain * (retinex * color_restoration + offset)

    # Normalize to [0,1] and gamma correct
    msrcr -= msrcr.min()
    msrcr /= (msrcr.max() + 1e-6)
    msrcr = np.power(msrcr, 1.0 / gamma)

    # Convert to 8-bit
    msrcr = (msrcr * 255).astype(np.uint8)

    # Color balance (optional)
    msrcr = simplest_color_balance(msrcr, low_clip=1, high_clip=99)

    # === Blend with original to restore skin tone ===
    msrcr = msrcr.astype(np.float32)
    original_img = original_img.astype(np.float32)
    blended = (1.0 - blend_factor) * msrcr + blend_factor * original_img
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended

def msrcr_skin_preserving(img, scales=[15, 80, 250], gain=0.4, offset=0, gamma=1.6, blend_factor=0.3, smooth_kernel=61):
    """
    MSRCR with better preservation of skin tone and shadows, ideal for side-lighting or uneven lighting.

    Args:
        img (np.ndarray): Input BGR image (H, W, 3), dtype=uint8.
        scales (list): Gaussian blur scales for Retinex.
        gain (float): Gain for contrast.
        offset (float): Offset after MSRCR.
        gamma (float): Gamma correction.
        blend_factor (float): Amount of original image retained in final blend.
        smooth_kernel (int): Kernel size for edge-aware smoothing.

    Returns:
        np.ndarray: Output image (H, W, 3), dtype=uint8.
    """
    original_img = img.copy().astype(np.float32)
    img = original_img + 1.0

    # --- Step 1: Multi-Scale Retinex ---
    retinex = multi_scale_retinex(img, scales)
    retinex = np.clip(retinex, -2.0, 2.0)
    
    # --- Step 2: Adaptive smoothing to share brightness in face ---
    # Use a larger kernel to better spread the illumination and reduce the bright spot
    illumination = cv2.GaussianBlur(retinex.mean(axis=2), (smooth_kernel, smooth_kernel), 0)
    illumination = np.expand_dims(illumination, axis=2)
    retinex = retinex - illumination  # Normalize to local average
    retinex = np.clip(retinex, -1.5, 1.5)  # Tighter clamping to avoid over-correction in bright areas

    # --- Step 3: Color Restoration ---
    img_sum = np.sum(img, axis=2, keepdims=True)
    img_sum[img_sum == 0] = 1.0
    color_restoration = np.log10(125.0 * img / img_sum)
    msrcr = gain * (retinex * color_restoration + offset)

    # --- Step 4: Local Contrast Adjustment for Bright Spots ---
    # Reduce contrast in overly bright areas
    brightness = msrcr.mean(axis=2, keepdims=True)
    brightness_mask = np.clip(brightness - 0.7 * brightness.max(), 0, None)  # Focus on bright areas
    brightness_mask /= (brightness_mask.max() + 1e-6)  # Normalize mask
    msrcr = msrcr * (1 - 0.3 * brightness_mask)  # Reduce intensity in bright spots

    # --- Step 5: Normalize & Gamma correction ---
    msrcr -= msrcr.min()
    msrcr /= (msrcr.max() + 1e-6)
    msrcr = np.power(msrcr, 1.0 / gamma)
    msrcr = (msrcr * 255).astype(np.uint8)

    # --- Step 6: Color balance ---
    msrcr = simplest_color_balance(msrcr, low_clip=1, high_clip=99)

    # --- Step 7: Blend with original to preserve skin tone & shadows ---
    msrcr = msrcr.astype(np.float32)
    blended = (1 - blend_factor) * msrcr + blend_factor * original_img
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended

def retinex_on_luminance(img_bgr, sigma=80):
    # Convert BGR (or RGB) to LAB
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(img_lab)

    # Normalize L
    L /= 255.0

    # Apply SSR on L channel
    L_retinex = single_scale_retinex(L, sigma)

    # Normalize back
    L_retinex -= L_retinex.min()
    L_retinex /= L_retinex.max() + 1e-6
    L_retinex = (L_retinex * 255).astype(np.uint8)

    # Merge back with original A/B
    result_lab = cv2.merge([L_retinex, A.astype(np.uint8), B.astype(np.uint8)])

    # Convert back to RGB
    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    return result_rgb

def msrcr_face_preserving(img, original_img, scales=[15, 80], gain=0.6, gamma=1.2):
    img = img.astype(np.float32) + 1.0  # Prevent log(0)
    retinex = multi_scale_retinex(img, scales)

    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = np.log10(125 * img / img_sum)

    result = gain * (retinex * color_restoration)
    result = result - result.min()
    result = result / (result.max() + 1e-6)
    result = np.power(result, 1.0 / gamma)

    result = (result * 255).astype(np.uint8)

    # Color correct toward original skin tone (match histogram)
    result = match_histograms(result, original_img, channel_axis=-1)

    # Reduce edge brightening using a Gaussian mask
    h, w = result.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    gaussian_mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2.0 * (0.35 * min(h, w))**2))
    gaussian_mask = gaussian_mask[:, :, np.newaxis]
    result = result.astype(np.float32) * gaussian_mask + original_img.astype(np.float32) * (1 - gaussian_mask)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def apply_clahe_on_face(image, bbox):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    face_roi = image[y1:y2, x1:x2]

    if face_roi.size == 0:
        return image  # Skip if region is invalid

    # Convert to LAB color space
    lab = cv2.cvtColor(face_roi, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge and convert back
    lab = cv2.merge((l_clahe, a, b))
    face_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Put the enhanced face region back
    enhanced = image.copy()
    enhanced[y1:y2, x1:x2] = face_eq
    return enhanced