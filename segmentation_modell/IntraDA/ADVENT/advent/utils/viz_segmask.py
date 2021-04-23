import numpy as np
from PIL import Image

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156,
           190, 153, 153, 153, 153, 153, 250,
           170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152,
           70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0,
           142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def label_colormap(n_label=256, value=None):
    """Label colormap.
    Parameters
    ----------
    n_labels: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.
    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    if value is not None:
        hsv = color_module.rgb2hsv(cmap.reshape(1, -1, 3))
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = color_module.hsv2rgb(hsv).reshape(-1, 3)
    return cmap

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    Palette = label_colormap().flatten()
    new_mask.putpalette(Palette)
    return new_mask
