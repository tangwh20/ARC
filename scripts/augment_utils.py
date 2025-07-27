import numpy as np


def rotate90(matrix: np.ndarray) -> np.ndarray:
    """Rotate a 2D matrix 90 degrees clockwise."""
    return np.rot90(matrix, -1)

def rotate180(matrix: np.ndarray) -> np.ndarray:
    """Rotate a 2D matrix 180 degrees."""
    return np.rot90(matrix, 2)

def rotate270(matrix: np.ndarray) -> np.ndarray:
    """Rotate a 2D matrix 270 degrees clockwise."""
    return np.rot90(matrix, 1)

def flip_horizontal(matrix: np.ndarray) -> np.ndarray:
    """Flip a 2D matrix horizontally."""
    return np.flip(matrix, axis=1)

def flip_vertical(matrix: np.ndarray) -> np.ndarray:
    """Flip a 2D matrix vertically."""
    return np.flip(matrix, axis=0)

def reflect_to_the_right(matrix: np.ndarray) -> np.ndarray:
    """Flips a grid horizontally and prepends to the right of the original grid."""
    flipped = flip_horizontal(matrix)
    return np.hstack((matrix, flipped))

def reflect_to_the_bottom(matrix: np.ndarray) -> np.ndarray:
    """Flips a grid vertically and appends to the bottom of the original grid."""
    flipped = flip_vertical(matrix)
    return np.vstack((matrix, flipped))

def reflect_to_the_left(matrix: np.ndarray) -> np.ndarray:
    """Flips a grid horizontally and prepends to the left of the original grid."""
    flipped = flip_horizontal(matrix)
    return np.hstack((flipped, matrix))

def reflect_to_the_top(matrix: np.ndarray) -> np.ndarray:
    """Flips a grid vertically and prepends to the top of the original grid."""
    flipped = flip_vertical(matrix)
    return np.vstack((flipped, matrix))

def transpose(matrix: np.ndarray) -> np.ndarray:
    """Transpose a 2D matrix."""
    return np.transpose(matrix)

def increase_resolution(matrix: np.ndarray, factor: int = 2) -> np.ndarray:
    """Increase the resolution of a 2D matrix by a given factor."""
    if factor <= 1:
        return matrix
    return np.repeat(np.repeat(matrix, factor, axis=0), factor, axis=1)

def increase_height(matrix: np.ndarray, factor: int = 2) -> np.ndarray:
    """Increase the height of a 2D matrix by a given factor."""
    if factor <= 1:
        return matrix
    return np.repeat(matrix, factor, axis=0)

def increase_width(matrix: np.ndarray, factor: int = 2) -> np.ndarray:
    """Increase the width of a 2D matrix by a given factor."""
    if factor <= 1:
        return matrix
    return np.repeat(matrix, factor, axis=1)

def roll_colors(matrix: np.ndarray, shift: int = 1) -> np.ndarray:
    """Roll the colors of a 2D matrix by a given shift."""
    if shift == 0:
        return matrix
    background = matrix == 0
    rolled = (matrix - 1 + shift) % 9 + 1
    rolled[background] = 0
    return rolled

# ====== 组合增强方法 ======

def combo_rotate90_flip_horizontal(matrix: np.ndarray) -> np.ndarray:
    """Combination: rotate 90 degrees then flip horizontally."""
    return flip_horizontal(rotate90(matrix))

def combo_rotate180_flip_vertical(matrix: np.ndarray) -> np.ndarray:
    """Combination: rotate 180 degrees then flip vertically."""
    return flip_vertical(rotate180(matrix))

def combo_rotate270_flip_horizontal(matrix: np.ndarray) -> np.ndarray:
    """Combination: rotate 270 degrees then flip horizontally."""
    return flip_horizontal(rotate270(matrix))

def combo_rotate90_transpose(matrix: np.ndarray) -> np.ndarray:
    """Combination: rotate 90 degrees then transpose."""
    return transpose(rotate90(matrix))

def combo_flip_horizontal_vertical(matrix: np.ndarray) -> np.ndarray:
    """Combination: flip horizontally then vertically."""
    return flip_vertical(flip_horizontal(matrix))

def combo_rotate90_roll_colors(matrix: np.ndarray) -> np.ndarray:
    """Combination: rotate 90 degrees then roll colors."""
    return roll_colors(rotate90(matrix))

def combo_rotate180_roll_colors(matrix: np.ndarray) -> np.ndarray:
    """Combination: rotate 180 degrees then roll colors."""
    return roll_colors(rotate180(matrix))

def combo_flip_horizontal_roll_colors(matrix: np.ndarray) -> np.ndarray:
    """Combination: flip horizontally then roll colors."""
    return roll_colors(flip_horizontal(matrix))

def combo_flip_vertical_roll_colors(matrix: np.ndarray) -> np.ndarray:
    """Combination: flip vertically then roll colors."""
    return roll_colors(flip_vertical(matrix))

def combo_increase_resolution_rotate90(matrix: np.ndarray) -> np.ndarray:
    """Combination: increase resolution then rotate 90 degrees."""
    return rotate90(increase_resolution(matrix))

def combo_increase_resolution_rotate180(matrix: np.ndarray) -> np.ndarray:
    """Combination: increase resolution then rotate 180 degrees."""
    return rotate180(increase_resolution(matrix))

def combo_increase_resolution_flip_horizontal(matrix: np.ndarray) -> np.ndarray:
    """Combination: increase resolution then flip horizontally."""
    return flip_horizontal(increase_resolution(matrix))

def combo_increase_resolution_flip_vertical(matrix: np.ndarray) -> np.ndarray:
    """Combination: increase resolution then flip vertically."""
    return flip_vertical(increase_resolution(matrix))

def combo_transpose_roll_colors(matrix: np.ndarray) -> np.ndarray:
    """Combination: transpose then roll colors."""
    return roll_colors(transpose(matrix))

def combo_increase_resolution_rotate90_roll_colors(matrix: np.ndarray) -> np.ndarray:
    """Combination: increase resolution, rotate 90 degrees, then roll colors."""
    return roll_colors(rotate90(increase_resolution(matrix)))

AUGMENTATIONS = {
    # 原始的15种增强方法
    0: lambda x: x,  # No augmentation
    1: rotate90,
    2: rotate180,
    3: rotate270,
    4: flip_horizontal,
    5: flip_vertical,
    6: reflect_to_the_right,
    7: reflect_to_the_bottom,
    8: reflect_to_the_left,
    9: reflect_to_the_top,
    10: transpose,
    11: increase_resolution,
    12: increase_height,
    13: increase_width,
    14: roll_colors,
    
    # 新增的15种组合增强方法
    15: combo_rotate90_flip_horizontal,         # rotate90 → flip_horizontal
    16: combo_rotate180_flip_vertical,          # rotate180 → flip_vertical
    17: combo_rotate270_flip_horizontal,        # rotate270 → flip_horizontal
    18: combo_rotate90_transpose,               # rotate90 → transpose
    19: combo_flip_horizontal_vertical,         # flip_horizontal → flip_vertical
    20: combo_rotate90_roll_colors,             # rotate90 → roll_colors
    21: combo_rotate180_roll_colors,            # rotate180 → roll_colors
    22: combo_flip_horizontal_roll_colors,      # flip_horizontal → roll_colors
    23: combo_flip_vertical_roll_colors,        # flip_vertical → roll_colors
    24: combo_increase_resolution_rotate90,     # increase_resolution → rotate90
    25: combo_increase_resolution_rotate180,    # increase_resolution → rotate180
    26: combo_increase_resolution_flip_horizontal, # increase_resolution → flip_horizontal
    27: combo_increase_resolution_flip_vertical,   # increase_resolution → flip_vertical
    28: combo_transpose_roll_colors,            # transpose → roll_colors
    29: combo_increase_resolution_rotate90_roll_colors, # 3-step combination
}
