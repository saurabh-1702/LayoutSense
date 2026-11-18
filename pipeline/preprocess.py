"""Stage 1 - shared Computer Vision preprocessing (project spec sections 4.1 and 7).

All tiers  : grayscale -> Otsu binarisation -> morphology -> contour detection (text blocks).
Mild/heavy : page detection + perspective rectification and deskew (geometry), denoising, CLAHE.

Every geometric step is accumulated into one 3x3 matrix M (original image -> preprocessed
image), so OCR boxes found on the preprocessed image can be mapped back to the original
image (where the answer-key annotations live) with inv(M).
"""
import cv2
import numpy as np


# ---------------------------------------------------------------- geometry ----
def _order_quad(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])  # tl, tr, br, bl


def detect_page(img_bgr):
    """Find the paper sheet on a scanner/desk background.

    The background is segmented by flood-filling from the four image corners on a
    downscaled, median-filtered copy (robust to noise, shadows and dark sidebars that
    are *inside* the page). Returns the 4 page corners (tl, tr, br, bl) in full-resolution
    pixels, or None if no convincing page is found (e.g. clean tier: page fills the image).
    """
    h, w = img_bgr.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)
    small = cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)
    sh, sw = small.shape[:2]
    # 1) model the background from a thin border strip: a plane per channel
    #    (captures uneven lighting / shadow gradients across the desk)
    b = 6
    yy, xx = np.mgrid[0:sh, 0:sw]
    strip = np.zeros((sh, sw), bool)
    strip[:b], strip[-b:], strip[:, :b], strip[:, -b:] = True, True, True, True
    A = np.c_[np.ones(strip.sum()), xx[strip], yy[strip]]
    Afull = np.c_[np.ones(sh * sw), xx.ravel(), yy.ravel()]
    diff = np.zeros((sh, sw), np.float32)
    for ch in range(3):
        v = small[..., ch][strip]
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        resid = v - A @ coef
        if resid.std() > 18:  # border is not a uniform background -> page fills the frame (clean tier)
            return None
        plane = (Afull @ coef).reshape(sh, sw)
        diff = np.maximum(diff, np.abs(small[..., ch] - plane) / max(4.0, 3.5 * resid.std()))
    # 2) page = anything that differs from the background model (bright paper, dark sidebars, text)
    page = (diff > 1.0).astype(np.uint8) * 255
    page[:2], page[-2:], page[:, :2], page[:, -2:] = 0, 0, 0, 0
    page = cv2.morphologyEx(page, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    page = cv2.morphologyEx(page, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(page, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.3 * sh * sw:
        return None
    hull = cv2.convexHull(c)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.05):
        ap = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(ap) == 4:
            quad = ap
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
    q = _order_quad(quad)
    # sanity checks: a sheet of paper is a big, convex, roughly A4/Letter-shaped quadrilateral
    tl, tr, br, bl = q
    wq = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    hq = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)) or cv2.contourArea(q) < 0.45 * sh * sw:
        return None
    if not (1.15 < hq / max(wq, 1) < 1.6):
        return None
    return q / scale


def rectify(img_bgr, quad, page_size=None):
    """Perspective-correct the detected page to an upright rectangle."""
    tl, tr, br, bl = quad
    wa = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    ha = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    if page_size:
        W, H = page_size
    else:
        W, H = int(round(max(wa))), int(round(max(ha)))
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    P = cv2.getPerspectiveTransform(np.float32(quad), dst)
    out = cv2.warpPerspective(img_bgr, P, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, P


def skew_angle(gray):
    """Residual text skew in degrees (Hough lines on the binarised text)."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    lines = cv2.HoughLinesP(bw, 1, np.pi / 720, threshold=200, minLineLength=small.shape[1] // 8, maxLineGap=10)
    if lines is None:
        return 0.0
    angs = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 20:
            angs.append(a)
    return float(np.median(angs)) if angs else 0.0


def rotate(img, angle):
    h, w = img.shape[:2]
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, np.vstack([R, [0, 0, 1]])


# ------------------------------------------------------------- photometric ----
def denoise(gray, tier):
    if tier == "heavy":
        g = cv2.fastNlMeansDenoising(gray, None, h=17, templateWindowSize=7, searchWindowSize=21)
    else:
        g = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    return g


def clahe(gray, tier):
    return cv2.createCLAHE(clipLimit=2.5 if tier == "heavy" else 2.0, tileGridSize=(8, 8)).apply(gray)


def flatten_illumination(gray):
    """Divide by a heavily blurred background estimate to remove shadows/gradients."""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    bg = cv2.GaussianBlur(bg, (0, 0), 25)
    return cv2.divide(gray, bg, scale=255)


# ------------------------------------------------------- always-on stages ----
def otsu(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def text_blocks(binary, min_area=400):
    """Morphology + contours: merge characters into words/lines, return block boxes."""
    inv = 255 - binary
    h, w = inv.shape
    k = max(3, w // 160)
    dil = cv2.dilate(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (k * 3, k)), iterations=2)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area and bw < 0.98 * w:
            boxes.append([int(x), int(y), int(x + bw), int(y + bh)])
    return sorted(boxes, key=lambda b: (b[1], b[0]))


# ------------------------------------------------------------------ driver ----
def preprocess(img_bgr, tier):
    """Returns dict(gray=<image for OCR>, binary=<Otsu image>, M=<3x3 original->preprocessed>, info=...)."""
    info = {"tier": tier, "steps": ["grayscale"]}
    M = np.eye(3)
    img = img_bgr
    if tier != "clean":
        quad = detect_page(img)
        if quad is not None:
            img, P = rectify(img, quad)
            M = P @ M
            info["steps"].append("page_detection+perspective_rectify")
            info["page_quad"] = np.round(quad, 1).tolist()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if tier != "clean":
        ang = skew_angle(gray)
        if abs(ang) > 0.25:
            gray, R = rotate(gray, ang)
            M = R @ M
            info["steps"].append(f"deskew({ang:.2f}deg)")
        gray = denoise(gray, tier)
        info["steps"].append("denoise(nl-means)")
        if tier == "heavy":
            gray = flatten_illumination(gray)
            info["steps"].append("illumination_flattening")
        gray = clahe(gray, tier)
        info["steps"].append("CLAHE")
    binary = otsu(gray)
    info["steps"] += ["otsu", "morphology+contours"]
    return dict(gray=gray, binary=binary, M=M, info=info)
