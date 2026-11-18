"""Composite scan/photo degradation (project spec section 6.4).

Every geometric change is a single homography H that is applied to the image AND
to every word / line / region box, so the answer key stays pixel-accurate.
"""
import random

import cv2
import numpy as np

TIERS = {
    "clean": None,
    "mild": dict(rot=(2, 5), noise=(5, 10), blur="gauss3", bc=(0.10, 0.15), persp=(0.02, 0.05), jpeg=70, shadow=False),
    "heavy": dict(rot=(8, 15), noise=(20, 30), blur="motion_or_defocus7", bc=(0.25, 0.35), persp=(0.08, 0.12), jpeg=30, shadow=True),
}


def _homography(w, h, cfg, rng):
    ang = rng.uniform(*cfg["rot"]) * rng.choice([-1, 1])
    cx, cy = w / 2, h / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    t = np.deg2rad(ang)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts = (src - [cx, cy]) @ R.T + [cx, cy]
    disp = []
    for i in range(4):
        mag = rng.uniform(*cfg["persp"])
        dx = rng.uniform(-1, 1) * mag * w
        dy = rng.uniform(-1, 1) * mag * h
        disp.append((round(dx, 1), round(dy, 1)))
    dst = pts + np.float32(disp)
    margin = 0.02 * max(w, h)
    mn = dst.min(0) - margin
    dst = dst - mn
    out_w, out_h = int(np.ceil(dst[:, 0].max() + margin)), int(np.ceil(dst[:, 1].max() + margin))
    H = cv2.getPerspectiveTransform(src, np.float32(dst))
    return H, (out_w, out_h), dict(rotation_deg=round(ang, 2), corner_displacement_px=disp)


def _kernel_motion(k, angle):
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1)
    ker = cv2.warpAffine(ker, M, (k, k))
    return ker / ker.sum()


def _kernel_disk(k):
    ker = np.zeros((k, k), np.float32)
    cv2.circle(ker, (k // 2, k // 2), k // 2, 1, -1)
    return ker / ker.sum()


def degrade(img_bgr, tier, seed):
    """Returns (degraded image, H 3x3 or None, params dict)."""
    cfg = TIERS[tier]
    if cfg is None:
        return img_bgr, None, {"tier": "clean"}
    rng = random.Random(seed)
    h, w = img_bgr.shape[:2]
    H, (ow, oh), gparams = _homography(w, h, cfg, rng)
    bg = rng.randint(205, 240) if tier == "mild" else rng.randint(70, 160)
    tint = [bg + rng.randint(-12, 12) for _ in range(3)]
    out = cv2.warpPerspective(img_bgr, H, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=tint)
    f = out.astype(np.float32)

    # brightness / contrast (+ uneven lighting for heavy)
    lo, hi = cfg["bc"]
    contrast = 1 - rng.uniform(lo, hi) if tier == "heavy" or rng.random() < 0.5 else 1 + rng.uniform(lo, hi) * 0.5
    bshift = rng.uniform(lo, hi) * 255 * rng.choice([-1, 1]) * (0.6 if tier == "heavy" else 1.0)
    f = (f - 128) * contrast + 128 + bshift
    p = dict(tier=tier, **gparams, contrast=round(contrast, 3), brightness_shift=round(bshift, 1), background=tint)
    if cfg["shadow"]:
        gx, gy = np.meshgrid(np.linspace(0, 1, ow, dtype=np.float32), np.linspace(0, 1, oh, dtype=np.float32))
        a = rng.uniform(0, 2 * np.pi)
        g = (np.cos(a) * gx + np.sin(a) * gy)
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)
        strength = rng.uniform(0.25, 0.45)
        f = f * (1 - strength * g)[..., None]
        p["shadow_strength"] = round(strength, 3)
        p["shadow_angle_deg"] = round(float(np.degrees(a)), 1)
    f = np.clip(f, 0, 255)

    # blur
    if cfg["blur"] == "gauss3":
        f = cv2.GaussianBlur(f, (3, 3), 0)
        p["blur"] = "gaussian_3x3"
    else:
        if rng.random() < 0.5:
            ang = rng.uniform(0, 180)
            f = cv2.filter2D(f, -1, _kernel_motion(7, ang))
            p["blur"] = f"motion_7x7_{ang:.0f}deg"
        else:
            f = cv2.filter2D(f, -1, _kernel_disk(7))
            p["blur"] = "defocus_7x7"

    # gaussian noise (mostly luminance, a little colour)
    sigma = rng.uniform(*cfg["noise"])
    nrng = np.random.default_rng(seed)
    lum = nrng.normal(0, sigma, (oh, ow, 1)).astype(np.float32)
    col = nrng.normal(0, sigma * 0.25, (oh, ow, 3)).astype(np.float32)
    f = np.clip(f + lum + col, 0, 255).astype(np.uint8)
    p["noise_sigma"] = round(sigma, 2)

    # JPEG round trip
    ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, cfg["jpeg"]])
    f = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    p["jpeg_quality"] = cfg["jpeg"]
    p["_jpeg_bytes"] = enc.tobytes()  # the final degraded image, exactly as encoded
    return f, H, p


def transform_box(H, box):
    """box [x0,y0,x1,y1] -> (poly [[x,y]*4], aabb [x0,y0,x1,y1])"""
    x0, y0, x1, y1 = box
    pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    if H is None:
        poly = pts.reshape(-1, 2)
    else:
        poly = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    poly = [[round(float(x), 1), round(float(y), 1)] for x, y in poly]
    xs, ys = [q[0] for q in poly], [q[1] for q in poly]
    return poly, [min(xs), min(ys), max(xs), max(ys)]
