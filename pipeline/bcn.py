"""Small vectorized BC1/BC3/BC4 encoders and mip chains for NASA previews.

GPU block compression keeps the static Earth maps at 4-8 bits per texel
instead of 32, with every mip precomputed so the renderer uploads blocks
directly. The encoders use principal-axis (BC1) and min/max (BC4) endpoint
fits; they trade a little quality for being dependency-free and fast.
"""
import numpy as np


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def mip_chain(image, srgb_channels):
    """Box-filtered mips down to 4 texels on the short edge (one BC block).

    ``srgb_channels`` channels are averaged in linear light; the rest (masks,
    data) are averaged directly.
    """
    levels = [image]
    current = image.astype(np.float32) / 255
    # Stop before a level stops tiling into whole 4x4 blocks.
    while current.shape[0] % 8 == 0 and current.shape[1] % 8 == 0:
        linear = current.copy()
        linear[..., srgb_channels] = srgb_to_linear(linear[..., srgb_channels])
        h, w = linear.shape[0] // 2, linear.shape[1] // 2
        linear = linear[:h * 2, :w * 2].reshape(h, 2, w, 2, -1).mean(axis=(1, 3))
        current = linear.copy()
        current[..., srgb_channels] = linear_to_srgb(linear[..., srgb_channels])
        levels.append(np.round(current * 255).astype(np.uint8))
    return levels


def blocks(image):
    """(H, W, C) -> (H/4 * W/4, 16, C), row-major blocks and texels."""
    h, w, c = image.shape
    return image.reshape(h // 4, 4, w // 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(-1, 16, c)


def encode_bc4(channel):
    """channel: (H, W) uint8 -> BC4 UNORM payload."""
    out = []
    source = blocks(channel[..., None])[..., 0].astype(np.int32)
    for start in range(0, len(source), 1 << 18):
        b = source[start:start + (1 << 18)]
        e0 = b.max(axis=1)
        e1 = b.min(axis=1)
        # e0 > e1 selects the 8-value palette; flat blocks keep index 0.
        weights = np.array([0, 7, 1, 2, 3, 4, 5, 6])  # palette position of index i
        palette = ((7 - weights)[None, :] * e0[:, None] + weights[None, :] * e1[:, None]) / 7.0
        index = np.abs(b[:, :, None] - palette[:, None, :]).argmin(axis=2).astype(np.uint64)
        index[e0 == e1] = 0
        bits = np.zeros(len(b), dtype=np.uint64)
        for texel in range(16):
            bits |= index[:, texel] << np.uint64(3 * texel)
        block = np.empty((len(b), 8), dtype=np.uint8)
        block[:, 0] = e0
        block[:, 1] = e1
        for byte in range(6):
            block[:, 2 + byte] = (bits >> np.uint64(8 * byte)) & np.uint64(0xFF)
        out.append(block.tobytes())
    return b"".join(out)


def _to565(rgb):
    r = np.round(rgb[:, 0] * 31 / 255).astype(np.int32)
    g = np.round(rgb[:, 1] * 63 / 255).astype(np.int32)
    b = np.round(rgb[:, 2] * 31 / 255).astype(np.int32)
    return (r << 11) | (g << 5) | b


def _from565(value):
    r = (value >> 11) & 31
    g = (value >> 5) & 63
    b = value & 31
    return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)), axis=-1).astype(np.float32)


def encode_bc1(rgb):
    """rgb: (H, W, 3) uint8 -> BC1 payload (four-colour mode only)."""
    out = []
    source = blocks(rgb).astype(np.float32)
    for start in range(0, len(source), 1 << 18):
        b = source[start:start + (1 << 18)]
        mean = b.mean(axis=1, keepdims=True)
        centred = b - mean
        cov = np.einsum("nki,nkj->nij", centred, centred)
        axis = np.ones((len(b), 3), dtype=np.float32)
        for _ in range(8):
            axis = np.einsum("nij,nj->ni", cov, axis)
            axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-6)
        projection = np.einsum("nki,ni->nk", centred, axis)
        lo = b[np.arange(len(b)), projection.argmin(axis=1)]
        hi = b[np.arange(len(b)), projection.argmax(axis=1)]
        c0 = _to565(hi)
        c1 = _to565(lo)
        swap = c0 < c1
        c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
        p0 = _from565(c0)
        p1 = _from565(c1)
        palette = np.stack((p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3), axis=1)
        distance = ((b[:, :, None, :] - palette[:, None, :, :]) ** 2).sum(axis=3)
        index = distance.argmin(axis=2).astype(np.uint32)
        index[c0 == c1] = 0
        bits = np.zeros(len(b), dtype=np.uint32)
        for texel in range(16):
            bits |= index[:, texel] << np.uint32(2 * texel)
        block = np.empty((len(b), 8), dtype=np.uint8)
        block[:, 0] = c0 & 0xFF
        block[:, 1] = c0 >> 8
        block[:, 2] = c1 & 0xFF
        block[:, 3] = c1 >> 8
        for byte in range(4):
            block[:, 4 + byte] = (bits >> np.uint32(8 * byte)) & np.uint32(0xFF)
        out.append(block.tobytes())
    return b"".join(out)


def encode_bc3(rgba):
    """rgba: (H, W, 4) uint8 -> BC3 payload (BC4 alpha block + BC1 colour block)."""
    alpha = np.frombuffer(encode_bc4(rgba[..., 3]), dtype=np.uint8).reshape(-1, 8)
    colour = np.frombuffer(encode_bc1(rgba[..., :3]), dtype=np.uint8).reshape(-1, 8)
    return np.concatenate((alpha, colour), axis=1).tobytes()


ENCODERS = {"bc1": (encode_bc1, 3), "bc3": (encode_bc3, 4), "bc4": (encode_bc4, 1)}


def encode_mipped(image, pixel_format, srgb_channels):
    """Returns (payload bytes, [{"byte_offset", "width", "height"}, ...])."""
    encode, channels = ENCODERS[pixel_format]
    if image.ndim == 2:
        image = image[..., None]
    image = image[..., :channels] if channels > 1 else image[..., :1]
    payload, mips = [], []
    offset = 0
    for level in mip_chain(image, srgb_channels):
        data = encode(level if channels > 1 else level[..., 0])
        mips.append({"byte_offset": offset, "width": level.shape[1], "height": level.shape[0]})
        payload.append(data)
        offset += len(data)
    return b"".join(payload), mips


def _halve(level, srgb_channels):
    """One box-filtered 2x reduction; a dimension already at 1 stays 1."""
    current = level.astype(np.float32) / 255
    linear = current.copy()
    linear[..., srgb_channels] = srgb_to_linear(linear[..., srgb_channels])
    h, w = linear.shape[:2]
    if h > 1:
        linear = linear[: h // 2 * 2].reshape(h // 2, 2, w, -1).mean(axis=1)
    h = linear.shape[0]
    if w > 1:
        linear = linear[:, : w // 2 * 2].reshape(h, w // 2, 2, -1).mean(axis=2)
    out = linear.copy()
    out[..., srgb_channels] = linear_to_srgb(linear[..., srgb_channels])
    return np.round(out * 255).astype(np.uint8)


def encode_full_chain(image, pixel_format, srgb_channels):
    """Every mip down to 1x1; levels smaller than a block are edge-padded.

    Returns (payload, mips) with the unpadded level sizes, laid out as the
    star-panorama loader expects (ceil(w/4) x ceil(h/4) blocks per level).
    """
    encode, channels = ENCODERS[pixel_format]
    if image.ndim == 2:
        image = image[..., None]
    level = image[..., :channels]
    payload, mips = [], []
    while True:
        h, w = level.shape[:2]
        padded = np.pad(level, ((0, (-h) % 4), (0, (-w) % 4), (0, 0)), mode="edge")
        payload.append(encode(padded if channels > 1 else padded[..., 0]))
        mips.append({"width": w, "height": h})
        if h == 1 and w == 1:
            break
        level = _halve(level, list(srgb_channels))
    return b"".join(payload), mips
