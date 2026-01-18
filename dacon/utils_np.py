"""
Image Processing Utilities - Pure NumPy/PIL Implementation
===========================================================

All functions converted from PyTorch to NumPy for CPU-only inference.
"""

import json
import numpy as np
from PIL import Image
import os
import cv2
import yaml
from skimage import measure
from skimage.morphology import footprint_rectangle, binary_dilation


def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_krita_inference_data_list(data_root, line_name, color_name, is_ref):

    data_list = []

    if is_ref:
        frame_names = get_file_names(os.path.join(data_root, line_name, "ref"))
    else:
        frame_names = get_file_names(os.path.join(data_root, line_name, "target"))

    for frame_name in frame_names:
        data_list.append([line_name, color_name, frame_name])

    return data_list

def get_file_names(path):
    file_names = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    return file_names

def check_seg_and_color(data_root, line_name, color_name):

    # all frames have corresponding lines - get their basenames from lines
    frame_names = get_file_names(os.path.join(data_root, line_name, "target"))
    ref_frame_names = get_file_names(os.path.join(data_root, line_name, "ref"))

    for frame_name in frame_names:
        line_image_path = os.path.join(data_root, line_name, "target", frame_name)
        seg_path = os.path.join(data_root, line_name, "seg", frame_name)
        os.makedirs(os.path.join(data_root, line_name, "seg"), exist_ok=True)

        if not(os.path.isfile(seg_path)):
            extract_segment(line_image_path, seg_path)

    for frame_name in ref_frame_names:
        color_image_path = os.path.join(data_root, color_name, "ref", frame_name)
        line_image_path = os.path.join(data_root, line_name, "ref", frame_name)
        seg_path = os.path.join(data_root, line_name, "seg", "ref", frame_name)
        color_json_path = os.path.join(data_root, line_name, "seg", f"{frame_name.split('.')[0]}.json")
        os.makedirs(os.path.join(data_root, line_name, "seg", "ref"), exist_ok=True)

        if not(os.path.isfile(seg_path)):
            extract_segment(line_image_path, seg_path)
        if not(os.path.isfile(color_json_path)):
            extract_color(color_image_path, seg_path, color_json_path)


def max_pool2d_numpy(input_array, kernel_size):
    """
    NumPy implementation of max pooling
    
    Args:
        input_array: [B, C, H, W]
        kernel_size: tuple (kh, kw)
    
    Returns:
        pooled array [B, C, H//kh, W//kw]
    """
    kh, kw = kernel_size
    B, C, H, W = input_array.shape
    
    H_out = H // kh
    W_out = W // kw
    
    # Reshape to expose pooling windows
    reshaped = input_array.reshape(B, C, H_out, kh, W_out, kw)
    
    # Max over the kernel dimensions
    pooled = reshaped.max(axis=3).max(axis=4)
    
    return pooled

def segment_pooling(feats_maps, seg_images, seg_num, pool_size):
    
    B, S, C, H_feat, W_feat = feats_maps.shape
    _, _, H_seg, W_seg = seg_images.shape
    L = int(seg_num.max())
    
    H, W = pool_size
    
    pooled_seg_feats = np.zeros((B, S, L, C), dtype=feats_maps.dtype)
    pool_k_h = max(1, H_seg // H)
    pool_k_w = max(1, W_seg // W)
    H_out = H_seg // pool_k_h
    W_out = W_seg // pool_k_w
    
    skip_upscale = (H_feat == H_out) and (W_feat == W_out)
    
    for s in range(S):
        seg_image = np.zeros((1, H_out, W_out))
        seg_image[0] = cv2.resize(src = seg_images[:, s].transpose(1,2,0), dsize = (W_out, H_out), interpolation = cv2.INTER_NEAREST)
        feats_map = feats_maps[:, s]
        max_seg_num = int(seg_num[:, s].max())
        
        masks = np.stack([(seg_image == (i + 1)) for i in range(max_seg_num)])
        masks = masks.transpose(1, 0, 2, 3).reshape(B * max_seg_num, 1, H_out, W_out)

        # masks = masks.transpose(1, 0, 2, 3).reshape(B * max_seg_num, 1, H_seg, W_seg)
        
        # Max pool using NumPy
        # masks_pooled = max_pool2d_numpy(masks, (pool_k_h, pool_k_w))
        masks_pooled = masks.reshape(B, max_seg_num, H_out, W_out).astype(bool) # prev masks_pooled
        
        # Interpolate using OpenCV
        if not skip_upscale:
            feats_map_resized = np.zeros((B, C, H_out, W_out), dtype=feats_map.dtype)
            for b in range(B):
                for c in range(C):
                    feats_map_resized[b, c] = cv2.resize(
                        feats_map[b, c],
                        (W_out, H_out),
                        interpolation=cv2.INTER_LINEAR
                    )
            feats_map = feats_map_resized
        
        feats_flat = feats_map.reshape(B, C, -1)
        masks_flat = masks_pooled.reshape(B, max_seg_num, -1).astype(np.float32)
        
        mask_sum = np.clip(masks_flat.sum(axis=2, keepdims=True), 1e-6, None)
        pooled = np.matmul(masks_flat, feats_flat.transpose(0, 2, 1)) / mask_sum
        
        pooled_seg_feats[:, s, :max_seg_num] = pooled
    
    return pooled_seg_feats

def get_image(image_path):
    """
    Load image as NumPy array
    
    Args:
        image_path: Path to image file
    
    Returns:
        NumPy array [C, H, W] in range [0, 1] as float32
    """
    # Load image with PIL
    image = Image.open(image_path).convert('RGB')
    
    # Convert to numpy array [H, W, C]
    image_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1]
    image_array /= 255.0
    
    # Transpose to [C, H, W]
    image_array = np.transpose(image_array, (2, 0, 1))
    
    return image_array


def get_alpha_line_image(image_path):
    """
    Load alpha channel from RGBA image
    
    Args:
        image_path: Path to RGBA image file
    
    Returns:
        NumPy array [1, H, W] with alpha channel in range [0, 1] as float32
    """
    # Load image with alpha channel
    image = Image.open(image_path).convert('RGBA')
    
    # Convert to numpy array [H, W, C]
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Extract alpha channel [H, W]
    alpha_channel = image_array[:, :, 3]
    
    # Add channel dimension [1, H, W]
    line_image = np.expand_dims(alpha_channel, axis=0)
    
    return line_image


def get_seg_idx_image(seg_image_path):
    """
    Load segmentation image and convert RGB to single index
    
    Args:
        seg_image_path: Path to RGB segmentation image
    
    Returns:
        seg_idx_image: NumPy array [H, W] with segment indices
        seg_list: NumPy array [N] with unique non-zero segment IDs
    """
    # Load image
    seg_image = Image.open(seg_image_path).convert('RGB')
    seg_image = np.array(seg_image)  # [H, W, 3]
    
    # Transpose to [3, H, W] to match original
    seg_image = np.transpose(seg_image, (2, 0, 1))  # [3, H, W]
    
    # Convert RGB to single index using bit shifting
    # Index = R << 16 | G << 8 | B
    seg_idx_image = (seg_image[0].astype(np.int32) << 16) + \
                    (seg_image[1].astype(np.int32) << 8) + \
                    seg_image[2].astype(np.int32)
    
    # Get unique non-zero segment IDs
    seg_list = np.unique(seg_idx_image[seg_idx_image != 0])
    
    return seg_idx_image, seg_list


def get_seg_info(seg_image_path, json_colors_path, seg_size):
    """
    Extract comprehensive segmentation information
    
    Args:
        seg_image_path: Path to segmentation image
        json_colors_path: Path to JSON with color information (or None)
        seg_size: Target size (H, W) for resizing, or None to keep original
    
    Returns:
        seg_num: int, number of segments
        seg_sizes: NumPy array [N] with segment sizes
        seg_colors: NumPy array [N, 4] with RGBA colors
        seg_coordinates: NumPy array [N, 4] with [center_y, center_x, height, width]
        new_seg_image: NumPy array [H, W] with segment IDs (1-indexed)
    """
    # Load image
    seg_image = Image.open(seg_image_path).convert('RGB')
    seg_image = np.array(seg_image)  # [H, W, 3]
    
    # Transpose to [3, H, W]
    seg_image = np.transpose(seg_image, (2, 0, 1))
    
    # Resize if needed
    if seg_size is not None:
        C, H_orig, W_orig = seg_image.shape
        H_new, W_new = seg_size
        
        # Resize each channel with nearest neighbor interpolation
        seg_image_resized = np.zeros((C, H_new, W_new), dtype=seg_image.dtype)
        for c in range(C):
            # cv2.resize expects (width, height)
            seg_image_resized[c] = cv2.resize(
                seg_image[c], 
                (W_new, H_new), 
                interpolation=cv2.INTER_NEAREST
            )
        seg_image = seg_image_resized
    
    # Load color data if provided
    if json_colors_path is not None:
        with open(json_colors_path, 'r') as file:
            color_data = json.load(file)
    else:
        color_data = None
    
    C, H, W = seg_image.shape
    
    # Convert RGB to single index
    seg_idx_image = (seg_image[0].astype(np.int32) << 16) + \
                    (seg_image[1].astype(np.int32) << 8) + \
                    seg_image[2].astype(np.int32)
    
    # Get unique non-zero segments
    seg_list = np.unique(seg_idx_image[seg_idx_image != 0])
    
    seg_num = len(seg_list)
    seg_colors = np.zeros((seg_num, 4), dtype=np.float32)
    seg_coordinates = np.zeros((seg_num, 4), dtype=np.int64)
    seg_sizes = np.zeros(seg_num, dtype=np.float32)
    
    # Create coordinate grids
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Initialize new segmentation image with consecutive IDs
    new_seg_image = np.zeros_like(seg_idx_image, dtype=np.int32)
    
    # Process each segment
    for idx, seg_idx in enumerate(seg_list):
        # Create mask for this segment
        mask = seg_idx_image == seg_idx
        
        # Get color from JSON if available
        if color_data is not None:
            rgba_value = color_data.get(str(int(seg_idx)), [-1, -1, -1, -1])
            seg_colors[idx] = np.array(rgba_value, dtype=np.float32)
        
        # Compute segment size
        seg_sizes[idx] = np.sum(mask)
        
        # Get coordinates of segment pixels
        xs = xx[mask]
        ys = yy[mask]
        
        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()
        
        # Compute center and bounding box
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        seg_coordinates[idx] = np.array([center_y, center_x, height, width], dtype=np.int64)
        
        # Assign consecutive ID (1-indexed)
        new_seg_image = np.where(mask, idx + 1, new_seg_image)
    
    return seg_num, seg_sizes, seg_colors, seg_coordinates, new_seg_image


def save_seg_label(label_image, save_path):
    """
    Save label image as RGB image (already NumPy)
    
    Args:
        label_image: NumPy array [H, W] with integer labels
        save_path: Path to save image
    """
    if save_path is None:
        return

    h, w = label_image.shape
    output_img = np.zeros((h, w, 3), dtype=np.uint8)

    label_uint = label_image.astype(np.int32)
    output_img[:, :, 0] = (label_uint >> 16) & 255
    output_img[:, :, 1] = (label_uint >> 8) & 255
    output_img[:, :, 2] = label_uint & 255

    Image.fromarray(output_img).save(save_path)


def merge_color_line_to_region(label_image, color_line_label_image):
    """
    Merge colored line labels into region labels (already NumPy)
    """
    updated_label_image = label_image.copy()
    current_max_label = updated_label_image.max()

    color_labels = np.unique(color_line_label_image)
    color_labels = color_labels[color_labels != 0]

    for lbl in color_labels:
        line_mask = color_line_label_image == lbl
        dilated_mask = binary_dilation(line_mask, footprint_rectangle((3, 3)))

        surrounding_labels = updated_label_image[dilated_mask]
        surrounding_labels = surrounding_labels[surrounding_labels != 0]

        if len(surrounding_labels) > 0:
            target_label = np.bincount(surrounding_labels).argmax()
        else:
            current_max_label += 1
            target_label = current_max_label

        updated_label_image[line_mask] = target_label

    return updated_label_image


def label_color_regions(filtered_image):
    """
    Label regions with same color (already NumPy)
    """
    label_img = np.zeros(filtered_image.shape[:2], dtype=np.int32)
    current_label = 1

    mask_alpha = filtered_image[:, :, 3] > 0
    colored_pixels = filtered_image[mask_alpha][:, :3]

    unique_colors = np.unique(colored_pixels, axis=0)

    for color in unique_colors:
        color_mask = np.all(filtered_image[:, :, :3] == color, axis=-1) & mask_alpha
        labeled = measure.label(color_mask, connectivity=2)

        labeled_nonzero = labeled > 0
        label_img[labeled_nonzero] = labeled[labeled_nonzero] + current_label - 1

        current_label += labeled.max()

    return label_img


def extract_color_line(line_rgba):
    mask = (line_rgba[:, :, 3] > 0) & (~np.all(line_rgba[:, :, :3] == [0, 0, 0], axis=2))

    filtered_image = np.ones_like(line_rgba, dtype=np.uint8) * 255
    filtered_image[:, :, 3] = 0

    filtered_image[mask] = line_rgba[mask]
    filtered_image[mask, 3] = 255

    label_image = np.zeros(filtered_image.shape[:2], dtype=np.int32)
    current_label = 1

    mask_alpha = filtered_image[:, :, 3] > 0
    colored_pixels = filtered_image[mask_alpha][:, :3]

    unique_colors = np.unique(colored_pixels, axis=0)

    for color in unique_colors:
        color_mask = np.all(filtered_image[:, :, :3] == color, axis=-1) & mask_alpha
        labeled = measure.label(color_mask, connectivity=2)
        labeled_nonzero = labeled > 0
        label_image[labeled_nonzero] = labeled[labeled_nonzero] + current_label - 1

        current_label += labeled.max()

    return label_image


def label_closed_regions(line_rgba):
    """
    Label closed regions in line art (already NumPy)
    """
    line_mask = line_rgba[:, :, 3] > 0

    region_mask = ~line_mask
    label_image = measure.label(region_mask, connectivity=1)

    return label_image


def convert_to_line_rgba(image_path):
    """
    Convert image to RGBA format for line art (already NumPy)
    """
    image = Image.open(image_path)
    image_np = np.array(image)

    h, w = image_np.shape[:2]
    line_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    if image.mode == "RGBA":
        line_rgba[:, :, :3] = image_np[:, :, :3]
        line_rgba[:, :, 3] = np.where(image_np[:, :, 3] > 0, 255, 0)
    
    elif image.mode == "RGB":
        line_rgba[:, :, :3] = image_np
        mask = np.any(image_np[:, :, :3] < 255, axis=2)
        line_rgba[:, :, 3] = np.where(mask, 255, 0)

    return line_rgba


def extract_segment(line_image_path, save_path):
    line_rgba = convert_to_line_rgba(line_image_path)
    color_line_label_image = extract_color_line(line_rgba)
    label_image = label_closed_regions(line_rgba)
    label_image = merge_color_line_to_region(label_image, color_line_label_image)
    save_seg_label(label_image, save_path)

    return


def extract_color(color_image_path, seg_image_path, save_path):
    """
    Extract color information from colored reference image
    
    Args:
        color_image_path: Path to colored reference image
        seg_image_path: Path to segmentation image
        save_path: Path to save JSON with color information
    """
    # Load color image
    color_image = Image.open(color_image_path)
    color_image = np.array(color_image)  # [H, W, C]
    
    # Add alpha channel if RGB
    if color_image.shape[2] == 3:
        alpha = np.full((color_image.shape[0], color_image.shape[1], 1), 255, dtype=np.uint8)
        color_image = np.concatenate([color_image, alpha], axis=-1)
    
    # Load segmentation image
    seg_idx_image, _ = get_seg_idx_image(seg_image_path)
    seg_idx_image = seg_idx_image.astype(np.int32)
    
    # Extract colors for each segment
    color_dict = {}
    props = measure.regionprops(seg_idx_image)

    for i in range(1, seg_idx_image.max() + 1):
        coords = props[i - 1].coords
        region_colors = color_image[coords[:, 0], coords[:, 1], :]
        
        # Find most common color in this segment
        unique_colors, counts = np.unique(region_colors, axis=0, return_counts=True)
        most_common_color = unique_colors[np.argmax(counts)]
        color_dict[str(i)] = most_common_color.tolist()

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(color_dict, f, indent=2)


def colorize_target_image(color_list_pred, image_tgt, seg_image_tgt, colors_only=True):
    """
    Colorize target image using predicted colors
    
    Args:
        color_list_pred: NumPy array [L, 4] with predicted RGBA colors in range [0, 255]
        image_tgt: NumPy array [C, H, W] target line art image in range [0, 1]
        seg_image_tgt: NumPy array [H, W] segmentation image with segment IDs
        colors_only: bool, if True only return colors without preserving line art
    
    Returns:
        NumPy array [H, W, 4] colored image in range [0, 1]
    """
    # Transpose image from [C, H, W] to [H, W, C]
    image_tgt = np.transpose(image_tgt, (1, 2, 0))
    h, w, _ = image_tgt.shape
    alpha = np.full(color_list_pred.shape[:-1] + (1,),
                    255,
                    dtype=color_list_pred.dtype)
    rgba_list_pred = np.concatenate([color_list_pred, alpha], axis=-1)
    
    # Initialize output image
    combined_image = np.zeros((h,w,4), dtype=np.float32)
    
    # Apply each predicted color to its segment
    for i, color in enumerate(rgba_list_pred):
        # Create mask for this segment (1-indexed)
        mask = seg_image_tgt == (i + 1)
        
        # Expand mask to match color channels
        mask = np.expand_dims(mask, axis=-1)  # [H, W, 1]
        
        # Normalize color to [0, 1]
        color_normalized = color / 255.0
        
        # Apply color where mask is True
        combined_image = np.where(mask, color_normalized, combined_image)
    
    if colors_only:
        return combined_image
    
    # Preserve black lines from original image
    # Black line is [0, 0, 0, 1] in RGBA
    black_line = np.array([0, 0, 0, 1], dtype=np.float32)
    black_mask = np.all(image_tgt == black_line, axis=-1)  # [H, W]
    black_mask = np.expand_dims(black_mask, axis=-1)  # [H, W, 1]
    
    combined_image = np.where(black_mask, image_tgt, combined_image)
    
    return combined_image


# ============================================================================
# ADDITIONAL HELPER FUNCTIONS
# ============================================================================

def normalize_color(colors):
    """
    Normalize colors from [0, 255] to [0, 1]
    
    Args:
        colors: NumPy array [..., C] in range [0, 255]
    
    Returns:
        NumPy array [..., C] in range [0, 1] as float32
    """
    return colors.astype(np.float32) / 255.0


def denormalize_color(colors):
    """
    Denormalize colors from [0, 1] to [0, 255]
    
    Args:
        colors: NumPy array [..., C] in range [0, 1]
    
    Returns:
        NumPy array [..., C] in range [0, 255] as uint8
    """
    return np.clip(colors * 255.0, 0, 255).astype(np.uint8)


def normalize_coordinate(coords, image_size):
    """
    Normalize coordinates to [0, 1] range
    
    Args:
        coords: NumPy array [..., 4] with [center_y, center_x, height, width]
        image_size: tuple (H, W)
    
    Returns:
        NumPy array [..., 4] normalized to [0, 1]
    """
    H, W = image_size
    coords = coords.astype(np.float32)
    
    # Normalize coordinates
    coords_normalized = coords.copy()
    coords_normalized[:, 0] = coords[:, 0] / H  # center_y
    coords_normalized[:, 1] = coords[:, 1] / W  # center_x
    coords_normalized[:, 2] = coords[:, 2] / H  # height
    coords_normalized[:, 3] = coords[:, 3] / W  # width
    
    return coords_normalized


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Image processing utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - get_image()")
    print("  - get_alpha_line_image()")
    print("  - get_seg_idx_image()")
    print("  - get_seg_info()")
    print("  - extract_segment()")
    print("  - extract_color()")
    print("  - colorize_target_image()")
    print("  - normalize_color()")
    print("  - denormalize_color()")
    print("  - normalize_coordinate()")
    
    # Example usage
    # image = get_image("path/to/image.png")
    # print(f"Loaded image shape: {image.shape}")