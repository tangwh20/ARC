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


AUGMENTATIONS = {
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
    14: roll_colors
}