import os
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk, ImageEnhance, ImageDraw

APP_TITLE = "Photo Color Matcher Pro"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def make_toolbar_icon(kind, size=34, fg="#f7f9fb"):
    """Render crisp Photoshop-style monochrome icons using 4x supersampling.

    The icon is drawn at high resolution and reduced with LANCZOS, so the EXE
    stays self-contained while curves/diagonals remain smooth on Windows.
    """
    S = 8
    N = size * S
    im = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = fg
    w = max(5, round(size * 0.065 * S))

    def P(v): return round(v * S)
    def box(x0, y0, x1, y1): return tuple(P(v) for v in (x0, y0, x1, y1))
    def line(points, width=w, fill=c):
        pts=[(P(x),P(y)) for x,y in points]
        d.line(pts, fill=fill, width=width, joint="curve")
        r=width//2
        for x,y in (pts[0], pts[-1]): d.ellipse((x-r,y-r,x+r,y+r), fill=fill)
    def ellipse(x0,y0,x1,y1,width=w,fill=None): d.ellipse(box(x0,y0,x1,y1), outline=c if fill is None else None, fill=fill, width=width)
    def rect(x0,y0,x1,y1,width=w,r=0,fill=None):
        kw={"outline":c if fill is None else None,"fill":fill,"width":width}
        d.rounded_rectangle(box(x0,y0,x1,y1), radius=P(r), **kw) if r else d.rectangle(box(x0,y0,x1,y1), **kw)
    def poly(points, fill=c): d.polygon([(P(x),P(y)) for x,y in points], fill=fill)

    if kind == "open":
        # folder
        poly([(4.5,10),(10,10),(12,7.5),(22.5,7.5),(24.5,10.5),(24.5,23),(4.5,23)], None)
        line([(5,10.5),(10,10.5),(12,8),(22,8),(24,10.5)], width=w)
        rect(4.5,10.5,24.5,23,width=w,r=1.2)
        line([(7,14),(22,14)], width=max(3,w-2))
    elif kind == "save":
        rect(5,4,24,25,width=w,r=1.2); rect(8,5.5,20,11.5,width=max(3,w-2))
        rect(9,17,20,24,width=max(3,w-2),r=.8)
    elif kind == "rect":
        # marching-ants style rectangle
        segs=[((5,5),(11,5)),((15,5),(23,5)),((5,23),(11,23)),((15,23),(23,23)),
              ((5,5),(5,11)),((5,15),(5,23)),((23,5),(23,11)),((23,15),(23,23))]
        for a,b in segs: line([a,b],width=max(4,w-1))
    elif kind == "ellipse":
        # Photoshop-style marching ants ellipse
        bb = box(4, 6, 25, 23)
        for a in range(0, 360, 28):
            d.arc(bb, start=a, end=min(a + 15, 359), fill=c, width=w)
    elif kind == "lasso":
        # freehand lasso + tail
        d.ellipse(box(4,5,24,20), outline=c, width=w)
        line([(15,19.5),(12,24.5),(17,22.5)],width=max(4,w-1))
    elif kind == "subtract_lasso":
        # lasso with a small minus badge: remove a polygon from current selection
        d.ellipse(box(4,5,22,19), outline=c, width=w)
        line([(14,18.5),(11.5,24),(16.5,21.5)],width=max(4,w-1))
        d.ellipse(box(17,16,28,27), fill="#343b42", outline=c, width=max(3,w-2))
        line([(19.5,21.5),(25.5,21.5)], width=max(4,w-1))
    elif kind == "brush":
        # tapered Photoshop-like brush
        line([(8,22),(18.5,11.5)],width=w+3)
        poly([(17.2,12.8),(21.4,4.2),(24.5,2.8),(22.6,10.8)])
        d.ellipse(box(4,19,11.5,25), fill=c)
    elif kind == "eraser":
        poly([(5,18.5),(15,7),(24,14.5),(13.5,25)], None)
        line([(5.5,18.5),(15,7),(24,14.5),(13.5,25),(5.5,18.5)],width=w)
        line([(9,20.8),(18.5,20.8)],width=max(4,w-1))
    elif kind == "eyedrop":
        # pipette
        line([(7,23),(20,10)],width=w+2)
        d.ellipse(box(18,4,25,11), outline=c, width=w)
        line([(5,25),(10,20)],width=w+1)
        line([(16.5,12.5),(20.5,16.5)],width=max(4,w-1))
    elif kind == "remove":
        # clean X/delete symbol
        line([(7,7),(23,23)],width=w+2); line([(23,7),(7,23)],width=w+2)
    elif kind == "color":
        # artist palette
        d.ellipse(box(4,5,25,24), outline=c, width=w)
        d.ellipse(box(17.5,17,24.5,24), fill=(0,0,0,0))
        for x,y in [(10,10),(16.5,9),(20.5,13.5),(11.5,18)]: d.ellipse(box(x-1.6,y-1.6,x+1.6,y+1.6),fill=c)
    elif kind == "blur":
        # Photoshop-style blur tool: clean water droplet silhouette
        # pointed top with a rounded lower body
        poly([(15,3.2),(9.0,12.0),(6.7,16.2),(6.6,18.7),(7.2,21.4),
              (8.7,23.8),(10.8,25.5),(13.5,26.5),(16.4,26.4),(19.1,25.3),
              (21.3,23.2),(22.7,20.5),(23.1,17.8),(22.5,15.1),(20.8,12.1)])
        # subtle highlight cut-out for a more polished icon
        d.arc(box(9.0,14.0,16.2,22.5), start=P(105), end=P(205), fill="#343b42", width=max(3,w-2))
    elif kind == "clone":
        ellipse(11,4,19,12,fill=c); rect(8,11,22,17,fill=c,r=1)
        rect(5,17,25,25,width=w,r=2)
    elif kind == "move":
        # Four-direction move tool, similar to a professional image editor.
        lw = max(4, w-1)
        line([(15,4),(15,26)], width=lw); line([(4,15),(26,15)], width=lw)
        poly([(15,2.5),(10.5,8),(19.5,8)])
        poly([(15,27.5),(10.5,22),(19.5,22)])
        poly([(2.5,15),(8,10.5),(8,19.5)])
        poly([(27.5,15),(22,10.5),(22,19.5)])
    elif kind == "crop":
        # Photoshop-style crop corners
        lw = max(4, w-1)
        line([(8,3.5),(8,20.5),(25,20.5)], width=lw)
        line([(3.5,8),(20.5,8),(20.5,25)], width=lw)
        line([(12,3.5),(12,8)], width=max(3,w-2))
        line([(3.5,12),(8,12)], width=max(3,w-2))
        line([(20.5,16),(25,16)], width=max(3,w-2))
        line([(16,20.5),(16,25)], width=max(3,w-2))
    elif kind == "compare":
        # split-frame before/after comparison icon
        rect(4.5,5,25.5,24.5,width=w,r=1.6)
        line([(15,6.5),(15,23)], width=max(3,w-2))
        poly([(7.5,19),(11.2,14.8),(14.2,18.2),(14.2,22),(7.5,22)])
        d.ellipse(box(9,9,12.5,12.5), fill=c)
        poly([(16.5,20.5),(20,16.5),(24,20.5),(24,23),(16.5,23)])
    elif kind == "undo":
        # clean counter-clockwise history arrow
        d.arc(box(6,6,25,25), P(48), P(302), fill=c, width=w)
        poly([(4.2,8.2),(12.1,4.2),(10.6,13.1)])
    elif kind == "reset":
        # full restore arrow, visually distinct from undo
        d.arc(box(5,5,25,25), P(15), P(338), fill=c, width=w)
        poly([(19.4,3.4),(26.2,7.9),(19.1,11.1)])
        d.ellipse(box(13.2,13.2,16.8,16.8), fill=c)
    elif kind == "clear":
        line([(7,7),(23,23)],width=w+2); line([(23,7),(7,23)],width=w+2)
    else:
        rect(6,6,24,24,width=w,r=1)

    return im.resize((size, size), Image.Resampling.LANCZOS)

class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show, add="+"); widget.bind("<Leave>", self.hide, add="+")
    def show(self, _=None):
        if self.tip: return
        x=self.widget.winfo_rootx()+8; y=self.widget.winfo_rooty()+self.widget.winfo_height()+6
        self.tip=tk.Toplevel(self.widget); self.tip.wm_overrideredirect(True); self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip,text=self.text,bg="#1f252b",fg="white",padx=7,pady=4,font=("Segoe UI",9)).pack()
    def hide(self, _=None):
        if self.tip: self.tip.destroy(); self.tip=None


def read_image(path):
    path = str(path)
    with Image.open(path) as im:
        exif = im.getexif()
        icc = im.info.get("icc_profile")
        dpi = im.info.get("dpi")
        im = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.array(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), exif, icc, dpi


def save_image(path, bgr, exif=None, icc=None, dpi=None, jpeg_quality=95):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    ext = Path(path).suffix.lower()
    kwargs = {}
    if exif:
        try:
            kwargs["exif"] = exif.tobytes()
        except Exception:
            pass
    if icc:
        kwargs["icc_profile"] = icc
    if dpi:
        kwargs["dpi"] = dpi
    if ext in {".jpg", ".jpeg"}:
        kwargs.update(quality=jpeg_quality, subsampling=0)
    elif ext == ".png":
        kwargs.update(compress_level=3)
    im.save(path, **kwargs)


def lab_stats(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return cv2.meanStdDev(lab)[0].reshape(3), cv2.meanStdDev(lab)[1].reshape(3)


def color_transfer(source_bgr, ref_mean, ref_std, strength=0.85):
    lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean, src_std = cv2.meanStdDev(lab)
    src_mean, src_std = src_mean.reshape(3), src_std.reshape(3)
    src_std = np.maximum(src_std, 1e-6)

    out = lab.copy()
    for c in range(3):
        out[:, :, c] = (lab[:, :, c] - src_mean[c]) * (ref_std[c] / src_std[c]) + ref_mean[c]
    out = np.clip(out, 0, 255).astype(np.uint8)
    matched = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(source_bgr, 1.0 - strength, matched, strength, 0)


def apply_manual_adjustments(bgr, brightness=0, contrast=0, saturation=0,
                             temperature=0, tint=0, highlights=0, shadows=0,
                             sharpness=0):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)

    if brightness:
        im = ImageEnhance.Brightness(im).enhance(max(0.05, 1.0 + brightness / 100.0))
    if contrast:
        im = ImageEnhance.Contrast(im).enhance(max(0.05, 1.0 + contrast / 100.0))
    if saturation:
        im = ImageEnhance.Color(im).enhance(max(0.0, 1.0 + saturation / 100.0))

    arr = np.asarray(im).astype(np.float32)

    # Temperature: + = warmer, - = cooler
    t = temperature / 100.0
    arr[:, :, 0] *= (1.0 + 0.18 * t)
    arr[:, :, 2] *= (1.0 - 0.18 * t)

    # Tint: + = magenta, - = green
    ti = tint / 100.0
    arr[:, :, 0] *= (1.0 + 0.07 * ti)
    arr[:, :, 2] *= (1.0 + 0.07 * ti)
    arr[:, :, 1] *= (1.0 - 0.12 * ti)

    arr = np.clip(arr, 0, 255)

    # Tonal adjustments with smooth luminance masks
    lum = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]) / 255.0
    if shadows:
        mask = np.clip((0.65 - lum) / 0.65, 0, 1)[..., None]
        amount = shadows / 100.0
        if amount >= 0:
            arr += (255.0 - arr) * mask * (0.35 * amount)
        else:
            arr *= 1.0 + mask * (0.35 * amount)

    if highlights:
        mask = np.clip((lum - 0.35) / 0.65, 0, 1)[..., None]
        amount = highlights / 100.0
        if amount >= 0:
            arr += (255.0 - arr) * mask * (0.25 * amount)
        else:
            arr *= 1.0 + mask * (0.35 * amount)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)

    if sharpness:
        im = ImageEnhance.Sharpness(im).enhance(max(0.0, 1.0 + sharpness / 50.0))

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def remove_masked_object(bgr, mask, radius=5, strength=70):
    """Offline object removal with adjustable strength and local sharpening."""
    if mask is None or not np.any(mask):
        return bgr.copy()

    mask8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    strength = int(np.clip(strength, 0, 100))
    alpha = strength / 100.0

    # 강도가 높을수록 물체 테두리까지 조금 더 넓게 제거
    iterations = 1 if strength < 55 else 2 if strength < 85 else 3
    kernel = np.ones((3, 3), np.uint8)
    work_mask = cv2.dilate(mask8, kernel, iterations=iterations)

    r = max(1.0, float(radius))

    # 서로 다른 두 복원 방식을 혼합하여 단순 번짐을 줄임
    telea = cv2.inpaint(bgr, work_mask, r, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(bgr, work_mask, max(1.0, r * 0.75), cv2.INPAINT_NS)
    repaired = cv2.addWeighted(telea, 1.0 - 0.55 * alpha, ns, 0.55 * alpha, 0)

    # 복원 영역에만 국부 선명화 적용
    amount = 0.20 + 1.20 * alpha
    blurred = cv2.GaussianBlur(repaired, (0, 0), 1.0)
    sharpened = cv2.addWeighted(repaired, 1.0 + amount, blurred, -amount, 0)

    # 경계만 자연스럽게 연결하고 내부는 선명하게 유지
    feather = cv2.GaussianBlur(work_mask, (0, 0), 0.65).astype(np.float32) / 255.0
    feather = feather[..., None]
    out = bgr.astype(np.float32) * (1.0 - feather) + sharpened.astype(np.float32) * feather
    return np.clip(out, 0, 255).astype(np.uint8)


def clone_texture_into_mask(bgr, mask, source_point, strength=70):
    """
    선택한 source_point 주변의 색/질감을 마스크 영역으로 평행 이동하여 복제.
    마스크 중심과 source_point 사이의 오프셋을 유지하므로 Clone Stamp처럼 동작한다.
    """
    if mask is None or not np.any(mask) or source_point is None:
        return bgr.copy()

    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx = int(round((x0 + x1) / 2))
    cy = int(round((y0 + y1) / 2))
    sx, sy = map(int, source_point)

    h, w = bgr.shape[:2]
    dx = sx - cx
    dy = sy - cy

    # 목적지 각 픽셀에 대응하는 원본 좌표를 만든다.
    yy, xx = np.indices((h, w), dtype=np.float32)
    map_x = np.clip(xx + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + dy, 0, h - 1).astype(np.float32)
    cloned = cv2.remap(bgr, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    mask8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    # 강도가 높을수록 가장자리까지 더 확실히 덮고, 낮으면 더 부드럽게 혼합
    s = float(np.clip(strength, 0, 100)) / 100.0
    sigma = 2.5 - 1.8 * s
    feather = cv2.GaussianBlur(mask8, (0, 0), max(0.55, sigma)).astype(np.float32) / 255.0
    feather = feather[..., None]

    # 복제 질감의 미세 선명도 강화
    blur = cv2.GaussianBlur(cloned, (0, 0), 0.9)
    amount = 0.15 + 0.85 * s
    cloned_sharp = cv2.addWeighted(cloned, 1.0 + amount, blur, -amount, 0)

    out = bgr.astype(np.float32) * (1.0 - feather) + cloned_sharp.astype(np.float32) * feather
    return np.clip(out, 0, 255).astype(np.uint8)


class PhotoColorMatcherApp(tk.Tk):
    def __init__(self):

        # V8.3 selection tool state
        self.shape_drag_start = None
        self.shape_preview_id = None
        super().__init__()
        self.active_remove_tool = "brush"
        self.clone_source_mode = False
        self.selection_tool = tk.StringVar(master=self, value="free")
        self.title(APP_TITLE)
        self.geometry("1080x760")
        self.minsize(900, 680)

        self.ref_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "matched_output"))
        self.keep_subfolders = tk.BooleanVar(value=True)
        self.files = []

        self.match_strength = tk.DoubleVar(value=85)
        self.brightness = tk.DoubleVar(value=0)
        self.contrast = tk.DoubleVar(value=0)
        self.saturation = tk.DoubleVar(value=0)
        self.temperature = tk.DoubleVar(value=0)
        self.tint = tk.DoubleVar(value=0)
        self.highlights = tk.DoubleVar(value=0)
        self.shadows = tk.DoubleVar(value=0)
        self.sharpness = tk.DoubleVar(value=0)

        self.preview_src = None
        self.preview_ref = None
        self.preview_photo_before = None
        self.preview_photo_after = None
        self._preview_job = None

        # Offline object-removal state
        self.object_image_path = None
        self.object_bgr = None
        self.object_result = None
        self.object_mask = None
        self.object_display_photo = None
        self.object_display_scale = 1.0
        self.object_display_offset = (0, 0)
        # V8.1 zoom/pan state: Ctrl+mouse wheel zooms around the cursor.
        self.object_zoom = 1.0
        self.object_zoom_center = None
        self.object_history = []
        self.object_compare_original = False
        self.object_full_original = None
        self.crop_drag_start = None
        self.crop_preview_id = None
        # V15 selected-pixel move state
        self.move_drag_start = None
        self.move_base_result = None
        self.move_layer = None
        self.move_mask = None
        self.move_offset = (0, 0)
        # V18 non-destructive object layers. The background image is never cut/blanked.
        self.object_layers = []
        self.active_layer_index = None
        self.layer_background = None
        self._last_brush_point = None
        self.brush_size = tk.DoubleVar(value=35)
        self.inpaint_radius = tk.DoubleVar(value=5)
        self.remove_strength = tk.DoubleVar(value=70)
        self.blur_strength = tk.DoubleVar(value=35)
        self.brush_preview_id = None
        self.clone_source_mode = False
        self.clone_source_point = None
        self.clone_source_marker_id = None

        # V8 manual polygon selection / editing state
        self.object_select_mode = False
        self.object_select_points = []
        self.object_edit_mask = None
        self.edit_tool = "brush"
        self.object_hue = tk.DoubleVar(value=0)
        self.object_saturation = tk.DoubleVar(value=0)
        self.object_brightness = tk.DoubleVar(value=0)

        self._build_ui()
        # V8.2: Ctrl +/- changes brush size anywhere in the app.
        self.bind_all("<Escape>", lambda e: self.cancel_v83_selection())

        self.bind_all("<Control-plus>", lambda e: self.adjust_brush_size_shortcut(+1))
        self.bind_all("<Control-equal>", lambda e: self.adjust_brush_size_shortcut(+1))
        self.bind_all("<Control-KP_Add>", lambda e: self.adjust_brush_size_shortcut(+1))
        self.bind_all("<Control-minus>", lambda e: self.adjust_brush_size_shortcut(-1))
        self.bind_all("<Control-KP_Subtract>", lambda e: self.adjust_brush_size_shortcut(-1))
        self.bind_all("<Control-s>", self._ctrl_save)
        self.bind_all("<Control-S>", self._ctrl_save)

        self.bind_all("<Control-d>", self.activate_clone_source_mode)
        self.bind_all("<Control-D>", self.activate_clone_source_mode)

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        color_tab = ttk.Frame(notebook)
        remove_tab = ttk.Frame(notebook)
        notebook.add(color_tab, text="색감 보정")
        notebook.add(remove_tab, text="물체 삭제")
        pad = 10
        top = ttk.Frame(color_tab, padding=pad)
        top.pack(fill="both", expand=True)

        ref = ttk.LabelFrame(top, text="기준 사진", padding=8)
        ref.pack(fill="x")
        ttk.Entry(ref, textvariable=self.ref_path).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(ref, text="기준 사진 선택", command=self.choose_reference).pack(side="left")

        targets = ttk.LabelFrame(top, text="보정할 사진", padding=8)
        targets.pack(fill="x", pady=(8, 0))
        btns = ttk.Frame(targets)
        btns.pack(fill="x")
        ttk.Button(btns, text="파일 추가", command=self.add_files).pack(side="left")
        ttk.Button(btns, text="폴더 추가", command=self.add_folder).pack(side="left", padx=6)
        ttk.Button(btns, text="목록 비우기", command=self.clear_files).pack(side="left")
        self.listbox = tk.Listbox(targets, height=5)
        self.listbox.pack(fill="x", pady=(6, 0))
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.schedule_preview())

        out = ttk.LabelFrame(top, text="출력 폴더", padding=8)
        out.pack(fill="x", pady=(8, 0))
        ttk.Entry(out, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(out, text="폴더 선택", command=self.choose_output).pack(side="left")

        body = ttk.Frame(top)
        body.pack(fill="both", expand=True, pady=(8, 0))

        controls = ttk.LabelFrame(body, text="세부 보정", padding=8)
        controls.pack(side="left", fill="y")

        self.add_slider(controls, "색감 매칭 강도", self.match_strength, 0, 100, "%")
        ttk.Separator(controls).pack(fill="x", pady=5)
        self.add_slider(controls, "밝기", self.brightness, -100, 100)
        self.add_slider(controls, "대비", self.contrast, -100, 100)
        self.add_slider(controls, "채도", self.saturation, -100, 100)
        self.add_slider(controls, "색온도", self.temperature, -100, 100)
        self.add_slider(controls, "틴트", self.tint, -100, 100)
        self.add_slider(controls, "하이라이트", self.highlights, -100, 100)
        self.add_slider(controls, "그림자", self.shadows, -100, 100)
        self.add_slider(controls, "선명도", self.sharpness, -100, 100)

        ttk.Button(controls, text="세부값 초기화", command=self.reset_adjustments).pack(fill="x", pady=(8, 4))

        # 저장 버튼은 항상 보이도록 세부 보정 패널 안에 고정 배치
        ttk.Button(
            controls,
            text="현재 사진 저장 (Ctrl+S)",
            command=self.save_current_color_result
        ).pack(fill="x", pady=(6, 3))

        ttk.Button(
            controls,
            text="전체 사진 일괄 저장",
            command=self.start_processing
        ).pack(fill="x", pady=(3, 5))

        ttk.Checkbutton(controls, text="폴더 추가 시 하위 폴더 구조 유지",
                        variable=self.keep_subfolders).pack(anchor="w", pady=(4, 0))

        preview = ttk.LabelFrame(body, text="미리보기", padding=8)
        preview.pack(side="left", fill="both", expand=True, padx=(8, 0))
        pv = ttk.Frame(preview)
        pv.pack(fill="both", expand=True)

        left = ttk.Frame(pv)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="원본").pack()
        self.before_label = ttk.Label(left, anchor="center")
        self.before_label.pack(fill="both", expand=True, padx=4, pady=4)

        right = ttk.Frame(pv)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="보정 결과").pack()
        self.after_label = ttk.Label(right, anchor="center")
        self.after_label.pack(fill="both", expand=True, padx=4, pady=4)

        bottom = ttk.Frame(top)
        bottom.pack(fill="x", pady=(8, 0))
        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.pack(fill="x")
        self.status = ttk.Label(bottom, text="준비됨")
        self.status.pack(anchor="w", pady=(4, 4))

        self._build_remove_tab(remove_tab)

    def _build_remove_tab(self, parent):
        wrap = ttk.Frame(parent, padding=10)
        wrap.pack(fill="both", expand=True)

        # Compact icon toolbar: Photoshop-like layout, but all icons are drawn internally.
        bar = tk.Frame(wrap, bg="#20262c", padx=6, pady=6)
        bar.pack(fill="x")
        self._toolbar_images = {}
        self._tool_buttons = {}

        def add_icon(kind, tip, command, tool_key=None, gap=2):
            img = ImageTk.PhotoImage(make_toolbar_icon(kind, 34))
            self._toolbar_images[tip] = img
            b = tk.Button(bar, image=img, command=command, width=46, height=46,
                          bg="#343b42", activebackground="#2387f3", relief="flat", bd=0,
                          highlightthickness=1, highlightbackground="#4a535c", cursor="hand2")
            b.pack(side="left", padx=gap)
            ToolTip(b, tip)
            if tool_key: self._tool_buttons[tool_key] = b
            return b

        add_icon("open", "사진 열기", self.open_object_image)
        add_icon("crop", "영역 잘라서 이동", self.activate_crop_mode, "crop")
        add_icon("move", "선택 영역 잘라서 이동", self.activate_move_mode, "move")
        tk.Frame(bar, width=1, bg="#66717b").pack(side="left", fill="y", padx=6)
        add_icon("rect", "사각형 선택", lambda: self.set_selection_tool("rect"), "rect")
        add_icon("ellipse", "타원형 선택", lambda: self.set_selection_tool("ellipse"), "ellipse")
        add_icon("lasso", "직접 선택", lambda: self.set_selection_tool("free"), "free")
        add_icon("brush", "브러시 · 마스크 추가", self.activate_brush_mode, "brush")
        add_icon("eraser", "브러시 마킹 지우기", self.activate_mask_eraser_mode, "eraser")
        add_icon("subtract_lasso", "직접 선택하여 선택영역 해제", self.activate_selection_subtract_mode, "free_subtract")
        tk.Frame(bar, width=1, bg="#66717b").pack(side="left", fill="y", padx=6)
        add_icon("color", "선택 색상 변경", self.apply_object_color)
        add_icon("blur", "브러시 블러", self.activate_blur_brush_mode, "blur_brush")
        add_icon("remove", "선택 영역 삭제", self.run_object_removal)
        add_icon("eyedrop", "질감 원본 선택 · Ctrl+D", self.activate_clone_source_mode, "clone_source")
        add_icon("clear", "질감 선택 해제", self.clear_clone_source)
        tk.Frame(bar, width=1, bg="#66717b").pack(side="left", fill="y", padx=6)
        add_icon("compare", "원본/결과 비교", self.toggle_object_compare, "compare")
        add_icon("undo", "실행 취소", self.undo_object_removal)
        add_icon("reset", "원본 복원", self.restore_object_original)
        save_btn = add_icon("save", "결과 저장", self.save_object_result)
        save_btn.pack_configure(side="right")
        self._sync_tool_buttons()

        opts = ttk.Frame(wrap)
        opts.pack(fill="x", pady=(8, 5))
        ttk.Label(opts, text="브러시 크기").pack(side="left")
        ttk.Scale(opts, from_=5, to=120, variable=self.brush_size, length=220).pack(side="left", padx=(6, 18))
        ttk.Label(opts, text="복원 범위").pack(side="left")
        ttk.Scale(opts, from_=1, to=15, variable=self.inpaint_radius, length=150).pack(side="left", padx=6)
        ttk.Label(opts, text="삭제 강도").pack(side="left", padx=(12, 0))
        ttk.Scale(opts, from_=0, to=100, variable=self.remove_strength, length=150).pack(side="left", padx=6)
        self.remove_strength_label = ttk.Label(opts, text="70", width=4)
        self.remove_strength_label.pack(side="left")
        self.remove_strength.trace_add("write", lambda *_: self.remove_strength_label.config(text=str(int(self.remove_strength.get()))))
        ttk.Label(opts, text="블러 강도").pack(side="left", padx=(12, 0))
        ttk.Scale(opts, from_=1, to=100, variable=self.blur_strength, length=120).pack(side="left", padx=6)
        self.blur_strength_label = ttk.Label(opts, text="35", width=4)
        self.blur_strength_label.pack(side="left")
        self.blur_strength.trace_add("write", lambda *_: self.blur_strength_label.config(text=str(int(self.blur_strength.get()))))
        self.clone_status_label = ttk.Label(
            opts,
            text="자동 복원 모드 · Ctrl+D → 사진의 질감 원본 클릭 시 복제 모드",
        )
        self.clone_status_label.pack(side="left", padx=12)

        color_opts = ttk.Frame(wrap)
        color_opts.pack(fill="x", pady=(0, 5))
        ttk.Label(color_opts, text="채도").pack(side="left", padx=(8, 0))
        ttk.Label(color_opts, text="밝기").pack(side="left", padx=(8, 0))
        self.object_select_status = ttk.Label(color_opts, text="브러시 모드 · Ctrl+마우스 휠: 확대/축소 · 직접 영역 선택으로 외곽선을 딸 수 있습니다.")
        self.object_select_status.pack(side="left", padx=12)

        work_area = ttk.Frame(wrap)
        work_area.pack(fill="both", expand=True)
        self.object_canvas = tk.Canvas(work_area, bg="#333333", highlightthickness=0, cursor="crosshair")
        self.object_canvas.pack(side="left", fill="both", expand=True)

        # Layer panel: moved objects are independent overlays; background stays untouched.
        layer_panel = ttk.LabelFrame(work_area, text="레이어", padding=6, width=190)
        layer_panel.pack(side="right", fill="y", padx=(8, 0))
        layer_panel.pack_propagate(False)
        self.layer_listbox = tk.Listbox(layer_panel, width=24, exportselection=False)
        self.layer_listbox.pack(fill="both", expand=True)
        self.layer_listbox.bind("<<ListboxSelect>>", self._on_layer_select)
        layer_btns = ttk.Frame(layer_panel)
        layer_btns.pack(fill="x", pady=(6, 0))
        ttk.Button(layer_btns, text="표시/숨김", command=self.toggle_active_layer_visibility).pack(fill="x")
        ttk.Button(layer_btns, text="레이어 삭제", command=self.delete_active_layer).pack(fill="x", pady=(4, 0))
        ttk.Label(layer_panel, text="이동할 영역을 선택 후\n↔ 이동 아이콘을 누르면\n새 객체 레이어가 생성됩니다.", justify="left").pack(anchor="w", pady=(8,0))

        self.object_canvas.bind("<Button-1>", self.object_canvas_click)
        self.object_canvas.bind("<B1-Motion>", self.object_canvas_drag)
        self.object_canvas.bind("<ButtonRelease-1>", self.object_canvas_release)
        self.object_canvas.bind("<Double-Button-1>", lambda e: self.finish_manual_selection() if self.active_remove_tool in ("free", "free_subtract") else None)
        self.bind_all("<Control-d>", self.activate_clone_source_mode)
        self.bind_all("<Control-D>", self.activate_clone_source_mode)
        self.object_canvas.bind("<Motion>", self.show_brush_preview)
        self.object_canvas.bind("<Leave>", self.hide_brush_preview)
        self.object_canvas.bind("<Control-MouseWheel>", self.object_canvas_zoom)
        self.object_canvas.bind("<Control-Button-4>", lambda e: self.object_canvas_zoom(e, 1))
        self.object_canvas.bind("<Control-Button-5>", lambda e: self.object_canvas_zoom(e, -1))
        self.object_canvas.bind("<Configure>", lambda e: self.refresh_object_canvas())

    def open_object_image(self):
        p = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        if not p:
            return
        try:
            bgr, *_ = read_image(p)
            self.object_image_path = p
            self.object_bgr = bgr.copy()
            self.object_full_original = bgr.copy()
            self.object_result = bgr.copy()
            self.object_compare_original = False
            self.object_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            self.object_history = []
            self.object_layers = []
            self.active_layer_index = None
            self.layer_background = bgr.copy()
            self._refresh_layer_list()
            self.clone_source_mode = False
            self.clone_source_point = None
            self.object_select_mode = False
            self.object_select_start = None
            self.object_edit_mask = None
            if hasattr(self, "clone_status_label"):
                self.clone_status_label.config(text="자동 복원 모드 · Ctrl+D → 사진의 질감 원본 클릭 시 복제 모드")
            self.refresh_object_canvas()
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"사진을 열 수 없습니다.\n{e}")

    def adjust_brush_size_shortcut(self, direction):
        """Ctrl + Plus/Minus: change removal brush size."""
        try:
            current = int(float(self.brush_size.get()))
        except Exception:
            current = 20

        step = 2 if current < 30 else 5
        new_value = max(1, min(200, current + step * direction))
        self.brush_size.set(new_value)

        # Update the red brush preview immediately if the pointer is on the canvas.
        try:
            px, py = self.winfo_pointerxy()
            cx = px - self.object_canvas.winfo_rootx()
            cy = py - self.object_canvas.winfo_rooty()
            if 0 <= cx < self.object_canvas.winfo_width() and 0 <= cy < self.object_canvas.winfo_height():
                class _Evt:
                    pass
                evt = _Evt()
                evt.x, evt.y = cx, cy
                self.show_brush_preview(evt)
        except Exception:
            pass
        return "break"

    def cancel_v83_selection(self):
        self.shape_drag_start = None
        if self.shape_preview_id:
            try:
                self.object_canvas.delete(self.shape_preview_id)
            except Exception:
                pass
            self.shape_preview_id = None
        for name in ("selection_mask", "object_mask", "mask"):
            if hasattr(self, name):
                try:
                    old = getattr(self, name)
                    if isinstance(old, np.ndarray):
                        setattr(self, name, np.zeros_like(old))
                except Exception:
                    pass
        for name in ("refresh_object_canvas", "update_object_canvas", "render_object_canvas", "show_object_image"):
            fn = getattr(self, name, None)
            if callable(fn):
                try: fn(); break
                except Exception: pass
        return "break"

    def _sync_tool_buttons(self):
        """Highlight only the currently active selection/paint tool."""
        buttons = getattr(self, "_tool_buttons", {})
        active = getattr(self, "active_remove_tool", "brush")
        for key, btn in buttons.items():
            selected = (key == active) or (key == "compare" and getattr(self, "object_compare_original", False))
            try:
                btn.configure(bg="#2387f3" if selected else "#343b42",
                              activebackground="#2387f3",
                              highlightbackground="#66b3ff" if selected else "#4a535c")
            except Exception:
                pass

    def _set_remove_tool_bindings(self, tool):
        """Keep exactly one active mouse tool."""
        self.active_remove_tool = tool
        self._sync_tool_buttons()
        self.object_select_mode = (tool in ("free", "free_subtract"))
        if tool in ("free", "free_subtract"):
            self.object_select_points = []
        else:
            self.object_select_points = []
        self.hide_brush_preview()
        if hasattr(self, "object_canvas"):
            cursor = "fleur" if tool == "move" else ("pencil" if tool in ("brush", "eraser", "blur_brush") else ("tcross" if tool == "crop" else "crosshair"))
            self.object_canvas.configure(cursor=cursor)
        return "break"


    def _refresh_layer_list(self):
        if not hasattr(self, "layer_listbox"):
            return
        self.layer_listbox.delete(0, tk.END)
        self.layer_listbox.insert(tk.END, "🔒 배경")
        for i, layer in enumerate(self.object_layers):
            eye = "●" if layer.get("visible", True) else "○"
            self.layer_listbox.insert(tk.END, f"{eye} {layer.get('name', f'객체 {i+1}')}" )
        if self.active_layer_index is not None and 0 <= self.active_layer_index < len(self.object_layers):
            self.layer_listbox.selection_set(self.active_layer_index + 1)

    def _on_layer_select(self, event=None):
        if not hasattr(self, "layer_listbox"):
            return
        sel = self.layer_listbox.curselection()
        if not sel:
            return
        idx = sel[0] - 1
        if idx < 0:
            self.active_layer_index = None
            return
        if idx < len(self.object_layers):
            self.active_layer_index = idx
            layer = self.object_layers[idx]
            self.move_mask = self._shift_mask(layer["mask"], layer.get("dx",0), layer.get("dy",0))
            self.object_mask = self.move_mask.copy()
            self.selection_mask = self.move_mask.copy()
            self._set_remove_tool_bindings("move")
            self.refresh_object_canvas()

    def _shift_mask(self, mask, dx, dy):
        h, w = mask.shape[:2]
        out = np.zeros_like(mask)
        ys, xs = np.where(mask > 0)
        nx, ny = xs + int(dx), ys + int(dy)
        valid = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
        out[ny[valid], nx[valid]] = 255
        return out

    def _compose_layers(self):
        if self.layer_background is None:
            if self.object_result is None: return None
            self.layer_background = self.object_result.copy()
        out = self.layer_background.copy()
        h, w = out.shape[:2]
        for layer in self.object_layers:
            if not layer.get("visible", True):
                continue
            mask = layer["mask"]
            pixels = layer["pixels"]
            dx, dy = int(layer.get("dx",0)), int(layer.get("dy",0))
            ys, xs = np.where(mask > 0)
            nx, ny = xs + dx, ys + dy
            valid = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
            out[ny[valid], nx[valid], :3] = pixels[ys[valid], xs[valid], :3]
        return out

    def toggle_active_layer_visibility(self):
        if self.active_layer_index is None or not (0 <= self.active_layer_index < len(self.object_layers)):
            messagebox.showinfo(APP_TITLE, "레이어 목록에서 객체 레이어를 선택해주세요.")
            return
        layer = self.object_layers[self.active_layer_index]
        layer["visible"] = not layer.get("visible", True)
        self.object_result = self._compose_layers()
        self._refresh_layer_list(); self.refresh_object_canvas()

    def delete_active_layer(self):
        if self.active_layer_index is None or not (0 <= self.active_layer_index < len(self.object_layers)):
            messagebox.showinfo(APP_TITLE, "삭제할 객체 레이어를 선택해주세요.")
            return
        self.object_layers.pop(self.active_layer_index)
        self.active_layer_index = None
        self.object_result = self._compose_layers()
        if self.object_mask is not None: self.object_mask[:] = 0
        self._refresh_layer_list(); self.refresh_object_canvas()

    def activate_move_mode(self):
        """Create/select a movable object layer without altering the background."""
        if self.object_result is None:
            messagebox.showwarning(APP_TITLE, "먼저 사진을 열어주세요.")
            return "break"
        mask = self.object_mask
        if mask is None or not np.any(mask):
            # If a layer is selected in the panel, simply reactivate moving it.
            if self.active_layer_index is not None and 0 <= self.active_layer_index < len(self.object_layers):
                layer = self.object_layers[self.active_layer_index]
                self.move_mask = self._shift_mask(layer["mask"], layer.get("dx",0), layer.get("dy",0))
                self._set_remove_tool_bindings("move")
                return "break"
            messagebox.showwarning(APP_TITLE, "먼저 이동할 영역을 선택해주세요.")
            return "break"
        self.clone_source_mode = False
        # Freeze the current edited picture as the untouched background when the first object layer is made.
        if self.layer_background is None or len(self.object_layers) == 0:
            self.layer_background = self.object_result.copy()
        source = self.object_result.copy()
        layer = {
            "name": f"객체 {len(self.object_layers)+1}",
            "pixels": source,
            "mask": (mask > 0).astype(np.uint8) * 255,
            "dx": 0, "dy": 0, "visible": True
        }
        self.object_layers.append(layer)
        self.active_layer_index = len(self.object_layers)-1
        self.move_mask = layer["mask"].copy()
        self.move_drag_start = None
        self.move_offset = (0,0)
        self._set_remove_tool_bindings("move")
        self._refresh_layer_list()
        self.object_result = self._compose_layers()
        self.refresh_object_canvas()
        self.object_select_status.config(text="객체 레이어 생성 완료 · 배경은 그대로 유지됩니다. 선택 객체를 드래그해 이동하세요.")
        return "break"

    def move_selection_start(self, event):
        if self.active_layer_index is None or not (0 <= self.active_layer_index < len(self.object_layers)):
            return
        layer = self.object_layers[self.active_layer_index]
        current_mask = self._shift_mask(layer["mask"], layer.get("dx",0), layer.get("dy",0))
        x, y = self._canvas_to_image(event.x, event.y)
        h, w = current_mask.shape[:2]
        if 0 <= x < w and 0 <= y < h and current_mask[y,x] > 0:
            self.move_drag_start = (x,y)
            self.move_offset = (layer.get("dx",0), layer.get("dy",0))

    def move_selection_drag(self, event):
        if self.move_drag_start is None or self.active_layer_index is None: return
        x, y = self._canvas_to_image(event.x, event.y)
        layer = self.object_layers[self.active_layer_index]
        start_dx, start_dy = self.move_offset
        layer["dx"] = start_dx + (x-self.move_drag_start[0])
        layer["dy"] = start_dy + (y-self.move_drag_start[1])
        self.move_mask = self._shift_mask(layer["mask"], layer["dx"], layer["dy"])
        self.object_mask = self.move_mask.copy()
        self.selection_mask = self.move_mask.copy()
        self.object_result = self._compose_layers()
        self.refresh_object_canvas()

    def move_selection_end(self, event):
        if self.move_drag_start is None: return
        self.move_drag_start = None
        self.move_offset = (0,0)
        self.object_result = self._compose_layers()
        self._refresh_layer_list(); self.refresh_object_canvas()
        self.object_select_status.config(text="이동 완료 · 배경은 변경되지 않았고 객체는 별도 레이어로 유지됩니다.")

    def activate_crop_mode(self):
        """Select a rectangular area to cut and immediately prepare it for moving."""
        self.clone_source_mode = False
        self.crop_drag_start = None
        if self.crop_preview_id:
            try: self.object_canvas.delete(self.crop_preview_id)
            except Exception: pass
            self.crop_preview_id = None
        self._set_remove_tool_bindings("crop")
        if hasattr(self, "object_select_status"):
            self.object_select_status.config(text="잘라서 이동 · 움직일 영역을 사각형으로 드래그해 선택하세요.")
        try: self.status_var.set("잘라서 이동: 움직일 영역을 드래그하여 선택하세요.")
        except Exception: pass
        return "break"

    def crop_select_start(self, event):
        self.crop_drag_start = (event.x, event.y)
        if self.crop_preview_id:
            try: self.object_canvas.delete(self.crop_preview_id)
            except Exception: pass
            self.crop_preview_id = None

    def crop_select_drag(self, event):
        if not self.crop_drag_start:
            return
        x0, y0 = self.crop_drag_start
        if self.crop_preview_id:
            try: self.object_canvas.delete(self.crop_preview_id)
            except Exception: pass
        self.crop_preview_id = self.object_canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="#f7f9fb", width=2, dash=(7,4))

    def crop_select_end(self, event):
        """Turn the dragged crop rectangle into a selection, then switch to Move."""
        if not self.crop_drag_start or self.object_result is None:
            return
        x0, y0 = self.crop_drag_start
        self.crop_drag_start = None
        ix0, iy0 = self._canvas_to_image_xy(x0, y0)
        ix1, iy1 = self._canvas_to_image_xy(event.x, event.y)
        left, right = sorted((ix0, ix1)); top, bottom = sorted((iy0, iy1))
        h, w = self.object_result.shape[:2]
        left=max(0,min(w-1,left)); right=max(0,min(w-1,right))
        top=max(0,min(h-1,top)); bottom=max(0,min(h-1,bottom))
        if right-left < 2 or bottom-top < 2:
            if hasattr(self, "object_select_status"):
                self.object_select_status.config(text="선택 영역이 너무 작습니다. 다시 드래그하세요.")
            return
        mask=np.zeros((h,w),dtype=np.uint8)
        cv2.rectangle(mask,(left,top),(right,bottom),255,-1)
        self.object_mask=mask
        self.selection_mask=mask.copy()
        self.object_edit_mask=mask.copy()
        if self.crop_preview_id:
            try: self.object_canvas.delete(self.crop_preview_id)
            except Exception: pass
            self.crop_preview_id=None
        self.refresh_object_canvas()
        self.activate_move_mode()
        if hasattr(self, "object_select_status"):
            self.object_select_status.config(text="영역 선택 완료 · 선택 물체를 드래그해 이동합니다. 배경은 그대로 유지되고 객체 레이어만 이동합니다.")
        try: self.status_var.set("영역 선택 완료: 선택 영역을 드래그해서 이동하세요.")
        except Exception: pass

    def set_selection_tool(self, tool):
        self.clone_source_mode = False
        self.selection_tool.set(tool)
        self.shape_drag_start = None
        if self.shape_preview_id:
            try: self.object_canvas.delete(self.shape_preview_id)
            except Exception: pass
            self.shape_preview_id = None
        self._set_remove_tool_bindings(tool)
        names={"rect":"사각형 선택","ellipse":"타원형 선택","free":"직접 선택"}
        try: self.status_var.set(f"{names.get(tool, tool)} 도구가 선택되었습니다.")
        except Exception: pass

    def _canvas_to_image_xy(self, x, y):
        scale = max(float(self.object_display_scale), 1e-6)
        ox, oy = self.object_display_offset
        return int((x - ox) / scale), int((y - oy) / scale)

    def shape_select_start(self, event):
        self.shape_drag_start = (event.x, event.y)
        if self.shape_preview_id:
            try: self.object_canvas.delete(self.shape_preview_id)
            except Exception: pass
        self.shape_preview_id = None

    def shape_select_drag(self, event):
        if not self.shape_drag_start:
            return
        x0, y0 = self.shape_drag_start
        if self.shape_preview_id:
            try: self.object_canvas.delete(self.shape_preview_id)
            except Exception: pass
        kwargs = dict(outline="#00ff55", width=2, dash=(5, 3))
        if self.selection_tool.get() == "ellipse":
            self.shape_preview_id = self.object_canvas.create_oval(x0, y0, event.x, event.y, **kwargs)
        else:
            self.shape_preview_id = self.object_canvas.create_rectangle(x0, y0, event.x, event.y, **kwargs)

    def shape_select_end(self, event):
        if not self.shape_drag_start:
            return
        x0, y0 = self.shape_drag_start
        x1, y1 = event.x, event.y
        self.shape_drag_start = None

        ix0, iy0 = self._canvas_to_image_xy(x0, y0)
        ix1, iy1 = self._canvas_to_image_xy(x1, y1)
        left, right = sorted((ix0, ix1))
        top, bottom = sorted((iy0, iy1))
        if right - left < 2 or bottom - top < 2:
            return

        # Determine original image size.
        img = self.object_result
        if img is None:
            return

        h, w = img.shape[:2]
        left, right = max(0, left), min(w - 1, right)
        top, bottom = max(0, top), min(h - 1, bottom)

        mask = np.zeros((h, w), dtype=np.uint8)
        if self.selection_tool.get() == "ellipse":
            cx, cy = (left + right) // 2, (top + bottom) // 2
            ax, ay = max(1, (right-left)//2), max(1, (bottom-top)//2)
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        else:
            cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)

        # One canonical selection mask for every editing action.
        self.object_mask = mask
        self.selection_mask = mask.copy()

        # Refresh overlay using whichever renderer exists.
        for name in ("refresh_object_canvas", "update_object_canvas", "render_object_canvas", "show_object_image"):
            fn = getattr(self, name, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    pass

        try:
            self.status_var.set("선택 완료 — 색상 변경, 영역 삭제 또는 질감 복제를 사용할 수 있습니다.")
        except Exception:
            pass

    def object_canvas_zoom(self, event, linux_direction=None):
        """Ctrl+wheel zoom. Keep the image point under the mouse approximately fixed."""
        if self.object_result is None:
            return "break"

        if linux_direction is not None:
            direction = linux_direction
        else:
            direction = 1 if getattr(event, "delta", 0) > 0 else -1

        old_zoom = self.object_zoom
        factor = 1.20 if direction > 0 else (1 / 1.20)
        new_zoom = max(1.0, min(8.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return "break"

        # Save the image coordinate currently below the mouse cursor.
        try:
            ix, iy = self._canvas_to_image(event.x, event.y)
            self.object_zoom_center = (ix, iy, event.x, event.y)
        except Exception:
            self.object_zoom_center = None

        self.object_zoom = new_zoom
        self.refresh_object_canvas()
        return "break"

    def refresh_object_canvas(self):
        if self.object_result is None or not hasattr(self, "object_canvas"):
            return
        self.object_canvas.update_idletasks()
        cw = max(100, self.object_canvas.winfo_width())
        ch = max(100, self.object_canvas.winfo_height())
        display_bgr = self.object_bgr if (getattr(self, "object_compare_original", False) and self.object_bgr is not None) else self.object_result
        h, w = display_bgr.shape[:2]
        fit_scale = min(cw / w, ch / h, 1.0)
        scale = fit_scale * self.object_zoom
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))

        # At 100%, center the image. While zooming, anchor the image point under the cursor.
        if self.object_zoom_center is not None and self.object_zoom > 1.0:
            ix, iy, cx, cy = self.object_zoom_center
            ox = int(cx - ix * scale)
            oy = int(cy - iy * scale)
            # Prevent losing the image completely outside the viewport.
            if dw > cw:
                ox = min(0, max(cw - dw, ox))
            else:
                ox = (cw - dw) // 2
            if dh > ch:
                oy = min(0, max(ch - dh, oy))
            else:
                oy = (ch - dh) // 2
        else:
            ox, oy = (cw - dw)//2, (ch - dh)//2

        self.object_display_scale = scale
        self.object_display_offset = (ox, oy)

        rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        disp = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        if (not getattr(self, "object_compare_original", False)) and self.object_mask is not None and np.any(self.object_mask):
            m = cv2.resize(self.object_mask, (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
            overlay = disp.copy()
            overlay[m] = [255, 55, 55]
            disp = np.where(m[..., None], (0.55*overlay + 0.45*disp).astype(np.uint8), disp)
            # selected area gets a crisp green outline
            if self.object_edit_mask is not None and np.any(self.object_edit_mask):
                em = cv2.resize(self.object_edit_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours((em > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(disp, contours, -1, (60, 255, 80), 2)

        if self.object_select_mode and self.object_select_points:
            pts_disp = [(int(px * scale), int(py * scale)) for px, py in self.object_select_points]
            if len(pts_disp) >= 2:
                cv2.polylines(disp, [np.array(pts_disp, dtype=np.int32)], False, (60, 255, 80), 2, cv2.LINE_AA)
            for i, (px, py) in enumerate(pts_disp):
                cv2.circle(disp, (px, py), 5 if i == 0 else 3, (60, 255, 80), -1, cv2.LINE_AA)

        im = Image.fromarray(disp)
        self.object_display_photo = ImageTk.PhotoImage(im)
        self.object_canvas.delete("all")
        self.object_canvas.create_image(ox, oy, anchor="nw", image=self.object_display_photo)

        # 선택한 질감 원본 위치를 화면에 십자표시
        if self.clone_source_point is not None:
            sx, sy = self.clone_source_point
            cx = ox + sx * scale
            cy = oy + sy * scale
            r = 10
            self.object_canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#00e5ff", width=2)
            self.object_canvas.create_line(cx-r-5, cy, cx+r+5, cy, fill="#00e5ff", width=2)
            self.object_canvas.create_line(cx, cy-r-5, cx, cy+r+5, fill="#00e5ff", width=2)

    def activate_clone_source_mode(self, event=None):
        self.clone_source_mode = True
        self._set_remove_tool_bindings("clone_source")
        try: self.status_var.set("질감 원본 선택: 복제할 색감/질감의 원본 위치를 클릭하세요.")
        except Exception: pass
        return "break"

    def clear_clone_source(self):
        """Clear clone-source state and leave clone/brush mode completely."""
        self.clone_source_mode = False
        self.clone_source_point = None
        self._last_brush_point = None
        self.hide_brush_preview()
        # Do not fall back to Brush.  A neutral tool prevents the brush cursor
        # and accidental mask painting immediately after '질감 선택 해제'.
        self._set_remove_tool_bindings("none")
        if hasattr(self, "clone_status_label"):
            self.clone_status_label.config(text="질감 선택 해제됨 · Ctrl+D로 새 질감 원본을 선택할 수 있습니다.")
        try:
            self.object_canvas.configure(cursor="arrow")
        except Exception:
            pass
        try:
            self.status_var.set("질감 선택이 해제되었습니다.")
        except Exception:
            pass
        self.refresh_object_canvas()
        return "break"

    def activate_object_select(self):
        self.clone_source_mode = False
        self.selection_tool.set("free")
        self._set_remove_tool_bindings("free")
        for n in ("manual_points", "polygon_points"):
            if hasattr(self, n): setattr(self, n, [])
        try: self.status_var.set("직접 선택: 외곽선을 따라 클릭하고 마지막 점에서 더블클릭하세요.")
        except Exception: pass

    def activate_brush_mode(self):
        self.clone_source_mode = False
        self._set_remove_tool_bindings("brush")
        try: self.status_var.set("브러시 모드: 드래그하여 마스크를 칠하세요.")
        except Exception: pass

    def activate_mask_eraser_mode(self):
        """Erase only the painted/selected mask without changing image pixels."""
        self.clone_source_mode = False
        self._set_remove_tool_bindings("eraser")
        if hasattr(self, "object_select_status"):
            self.object_select_status.config(text="마스크 지우개 모드 · 드래그하여 불필요하게 선택된 부분만 지우세요.")
        try: self.status_var.set("마스크 지우개: 선택 영역 중 불필요한 부분을 드래그하여 지웁니다.")
        except Exception: pass

    def activate_selection_subtract_mode(self):
        """Trace a polygon and subtract it from the current selection mask."""
        self.clone_source_mode = False
        self.selection_tool.set("free")
        self.object_select_points = []
        self._set_remove_tool_bindings("free_subtract")
        if hasattr(self, "object_select_status"):
            self.object_select_status.config(text="선택영역 해제 · 제외할 부분의 외곽선을 따라 클릭하고 시작점을 다시 클릭하거나 더블클릭하세요.")
        try:
            self.status_var.set("선택영역 해제: 제외할 부분을 직접 선택하세요. 기존 사진은 변경되지 않습니다.")
        except Exception:
            pass

    def finish_manual_selection(self):
        if self.object_result is None or len(self.object_select_points) < 3:
            self.object_select_status.config(text="최소 3개 이상의 점을 찍어주세요.")
            return
        pts = np.array(self.object_select_points, dtype=np.int32)
        mask = np.zeros(self.object_result.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        if self.active_remove_tool == "free_subtract":
            # Subtract only from the selection; never alter image pixels here.
            if self.object_mask is None or self.object_mask.shape != mask.shape:
                self.object_mask = np.zeros_like(mask)
            self.object_mask[mask > 0] = 0
            if isinstance(self.object_edit_mask, np.ndarray) and self.object_edit_mask.shape == mask.shape:
                self.object_edit_mask[mask > 0] = 0
            if hasattr(self, "selection_mask") and isinstance(self.selection_mask, np.ndarray) and self.selection_mask.shape == mask.shape:
                self.selection_mask[mask > 0] = 0
            else:
                self.selection_mask = self.object_mask.copy()
            self.object_select_status.config(text="선택영역 해제 완료 · 필요한 부분만 선택 영역에 남았습니다.")
        else:
            self.object_mask = mask
            self.selection_mask = mask.copy()
            self.object_edit_mask = mask.copy()
            self.object_select_status.config(text="직접 선택 완료 · 색상 변경/삭제 가능")
        self.object_select_mode = False
        self.object_select_points = []
        self.refresh_object_canvas()

    def _canvas_to_image(self, cx, cy):
        scale = max(self.object_display_scale, 1e-6)
        ox, oy = self.object_display_offset
        return int((cx - ox) / scale), int((cy - oy) / scale)

    def object_canvas_drag(self, event):
        if self.object_select_mode:
            return
        self.paint_object_mask(event)

    def object_canvas_release(self, event):
        return

    def activate_blur_brush_mode(self):
        """Photoshop-like blur brush: paint only where blur is wanted."""
        if self.object_result is None:
            messagebox.showwarning(APP_TITLE, "먼저 사진을 열어주세요.")
            return "break"
        self.clone_source_mode = False
        self._set_remove_tool_bindings("blur_brush")
        self._blur_stroke_mask = np.zeros(self.object_result.shape[:2], dtype=np.uint8)
        self._blur_stroke_base = None
        if hasattr(self, "object_select_status"):
            self.object_select_status.config(text="블러 브러시 · 원하는 부분을 드래그하세요. 파란 원이 브러시 크기를 표시합니다.")
        try: self.status_var.set("블러 브러시: 원하는 부분만 마우스로 칠하세요.")
        except Exception: pass
        return "break"

    def _paint_blur_stroke(self, event):
        if self.object_result is None:
            return
        if getattr(self, "_blur_stroke_base", None) is None:
            self._blur_stroke_base = self.object_result.copy()
            self._blur_stroke_mask = np.zeros(self.object_result.shape[:2], dtype=np.uint8)
        scale=max(self.object_display_scale,1e-6); ox,oy=self.object_display_offset
        x=int((event.x-ox)/scale); y=int((event.y-oy)/scale)
        h,w=self._blur_stroke_mask.shape
        if not (0<=x<w and 0<=y<h): return
        radius=max(1,int(self.brush_size.get()/scale/2))
        prev=getattr(self,"_last_brush_point",None)
        if prev is not None: cv2.line(self._blur_stroke_mask,prev,(x,y),255,radius*2,cv2.LINE_AA)
        cv2.circle(self._blur_stroke_mask,(x,y),radius,255,-1,cv2.LINE_AA)
        self._last_brush_point=(x,y)
        strength=max(1,min(100,int(round(self.blur_strength.get()))))
        sigma=0.6+(strength/100.0)*24.0
        base=self._blur_stroke_base
        blurred=cv2.GaussianBlur(base,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REFLECT)
        alpha=cv2.GaussianBlur(self._blur_stroke_mask,(0,0),0.8).astype(np.float32)/255.0
        if base.ndim==3: alpha=alpha[...,None]
        self.object_result=(base.astype(np.float32)*(1-alpha)+blurred.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)
        self.refresh_object_canvas(); self.show_brush_preview(event)

    def _finish_blur_stroke(self):
        base=getattr(self,"_blur_stroke_base",None)
        mask=getattr(self,"_blur_stroke_mask",None)
        if base is not None and isinstance(mask,np.ndarray) and np.any(mask):
            self.object_history.append(base.copy())
        self._blur_stroke_base=None
        if self.object_result is not None:
            self._blur_stroke_mask=np.zeros(self.object_result.shape[:2],dtype=np.uint8)
        self._last_brush_point=None
        if hasattr(self,"object_select_status"):
            self.object_select_status.config(text=f"블러 적용 완료 · 강도 {int(self.blur_strength.get())} · 계속 칠하거나 실행 취소할 수 있습니다.")

    def apply_selection_blur(self):
        """Blur only the currently selected mask while preserving a natural edge."""
        mask = getattr(self, "object_mask", None)
        if not (isinstance(mask, np.ndarray) and mask.size and np.any(mask > 0)):
            mask = getattr(self, "selection_mask", None)
        if not (isinstance(mask, np.ndarray) and mask.size and np.any(mask > 0)):
            messagebox.showinfo("선택 필요", "먼저 블러 처리할 영역을 선택해주세요.")
            return
        if self.object_result is None:
            messagebox.showwarning(APP_TITLE, "먼저 사진을 열어주세요.")
            return

        strength = max(1, min(100, int(round(self.blur_strength.get()))))
        # Map 1..100 to a useful Gaussian sigma. Large values create strong privacy blur.
        sigma = 0.6 + (strength / 100.0) * 24.0
        src = self.object_result.copy()
        blurred = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

        # Slight feathering only at the boundary prevents a hard pasted-on edge.
        feather_sigma = max(0.6, min(2.2, sigma * 0.08))
        alpha = cv2.GaussianBlur(mask, (0, 0), feather_sigma).astype(np.float32) / 255.0
        if src.ndim == 3:
            alpha = alpha[..., None]
        result = (src.astype(np.float32) * (1.0 - alpha) +
                  blurred.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

        self.object_history.append(self.object_result.copy())
        self.object_result = result
        self.object_compare_original = False
        self._sync_tool_buttons()
        self.refresh_object_canvas()
        try:
            self.status_var.set(f"선택 영역에 블러를 적용했습니다. 강도 {strength}")
        except Exception:
            pass
        if hasattr(self, "object_select_status"):
            self.object_select_status.config(text=f"블러 적용 완료 · 강도 {strength} · 실행 취소로 되돌릴 수 있습니다.")

    def apply_object_color(self):
        """Open a Photoshop-like system color picker and recolor the selected object."""
        # Use the same mask created by every selection tool.
        mask = getattr(self, "object_mask", None)
        if not (isinstance(mask, np.ndarray) and mask.size and np.any(mask > 0)):
            mask = getattr(self, "selection_mask", None)
        if not (isinstance(mask, np.ndarray) and mask.size and np.any(mask > 0)):
            mask = None

        if mask is None:
            messagebox.showinfo("선택 필요", "먼저 색상을 변경할 영역을 선택해주세요.")
            return

        # Start from the previously selected color when available.
        initial = getattr(self, "selected_object_color_hex", "#ff6600")
        picked = colorchooser.askcolor(color=initial, title="선택 물체 색상")
        if not picked or not picked[1]:
            return
        rgb, hex_color = picked
        self.selected_object_color_hex = hex_color

        img = self.object_result
        if img is None:
            messagebox.showerror("오류", "편집할 사진이 없습니다.")
            return

        # Preserve original luminance/texture while replacing hue/chroma toward chosen color.
        work = img.copy()
        if work.ndim != 3 or work.shape[2] < 3:
            messagebox.showerror("오류", "지원되지 않는 이미지 형식입니다.")
            return

        target_rgb = np.uint8([[list(map(int, rgb))]])
        target_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)[0, 0]
        target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]

        hsv = cv2.cvtColor(work[:, :, :3], cv2.COLOR_BGR2HSV)
        selected = mask > 0

        # Replace hue strongly, blend saturation so surface detail remains natural.
        hsv2 = hsv.copy()
        hsv2[..., 0][selected] = target_hsv[0]
        orig_s = hsv[..., 1].astype(np.float32)
        target_s = float(target_hsv[1])
        hsv2[..., 1][selected] = np.clip(
            orig_s[selected] * 0.35 + target_s * 0.65, 0, 255
        ).astype(np.uint8)
        # Value/brightness is intentionally preserved to retain shadows and texture.

        recolored = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

        # Feather only the mask edge slightly; the interior stays sharp.
        soft = cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32) / 255.0
        alpha = soft[..., None]
        result = (work[:, :, :3].astype(np.float32) * (1.0 - alpha) +
                  recolored.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

        if work.shape[2] > 3:
            result = np.dstack([result, work[:, :, 3:]])

        self.object_history.append(self.object_result.copy())
        self.object_result = result

        for name in ("refresh_object_canvas", "update_object_canvas", "render_object_canvas", "show_object_image"):
            fn = getattr(self, name, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    pass

        try:
            self.status_var.set(f"선택 영역 색상을 {hex_color}로 변경했습니다.")
        except Exception:
            pass

    def object_canvas_click(self, event):
        if self.object_result is None:
            return "break"
        tool = self.active_remove_tool

        if tool == "clone_source":
            x, y = self._canvas_to_image(event.x, event.y)
            h, w = self.object_result.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                self.clone_source_point = (x, y)
                self.clone_source_mode = False
                self.clone_status_label.config(
                    text=f"질감 복제 모드 · 원본 위치 ({x}, {y}) 선택됨"
                )
                self.activate_brush_mode()
                self.refresh_object_canvas()
            return "break"

        if tool == "move":
            self.move_selection_start(event)
            return "break"

        if tool == "crop":
            self.crop_select_start(event)
            return "break"

        if tool in ("rect", "ellipse"):
            self.shape_select_start(event)
            return "break"

        if tool in ("free", "free_subtract"):
            x, y = self._canvas_to_image(event.x, event.y)
            h, w = self.object_result.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                return "break"
            if not self.object_select_mode:
                self.object_select_mode = True
                self.object_select_points = []
            if len(self.object_select_points) >= 3:
                fx, fy = self.object_select_points[0]
                scale = max(self.object_display_scale, 1e-6)
                close_dist = max(6, int(12 / scale))
                if (x-fx)**2 + (y-fy)**2 <= close_dist**2:
                    self.finish_manual_selection()
                    return "break"
            self.object_select_points.append((x, y))
            mode_text = "선택영역 해제 중" if tool == "free_subtract" else "직접 선택 중"
            self.object_select_status.config(
                text=f"{mode_text} · {len(self.object_select_points)}개 점 · 시작점을 다시 클릭하거나 더블클릭하면 완료"
            )
            self.refresh_object_canvas()
            return "break"

        if tool == "brush":
            self._last_brush_point = None
            self.paint_object_mask(event)
        elif tool == "eraser":
            self._last_brush_point = None
            self.erase_object_mask(event)
        elif tool == "blur_brush":
            self._last_brush_point = None
            self._paint_blur_stroke(event)
        return "break"

    def object_canvas_drag(self, event):
        tool = self.active_remove_tool
        if tool == "move":
            self.move_selection_drag(event)
        elif tool == "crop":
            self.crop_select_drag(event)
        elif tool in ("rect", "ellipse"):
            self.shape_select_drag(event)
        elif tool == "brush":
            self.paint_object_mask(event)
        elif tool == "eraser":
            self.erase_object_mask(event)
        elif tool == "blur_brush":
            self._paint_blur_stroke(event)
        return "break"

    def object_canvas_release(self, event):
        if self.active_remove_tool == "move":
            self.move_selection_end(event)
        elif self.active_remove_tool == "crop":
            self.crop_select_end(event)
        elif self.active_remove_tool in ("rect", "ellipse"):
            self.shape_select_end(event)
        elif self.active_remove_tool == "blur_brush":
            self._finish_blur_stroke()
        self._last_brush_point = None
        return "break"

    def show_brush_preview(self, event):
        """현재 브러시 크기를 마우스 위치에 빨간 원으로 표시."""
        if self.object_result is None:
            return
        if self.active_remove_tool not in ("brush", "eraser", "blur_brush"):
            self.hide_brush_preview()
            return
        if self.brush_preview_id is not None:
            try:
                self.object_canvas.delete(self.brush_preview_id)
            except Exception:
                pass
        # brush_size는 화면상 직경 기준
        radius = max(2, int(self.brush_size.get() / 2))
        self.brush_preview_id = self.object_canvas.create_oval(
            event.x - radius, event.y - radius,
            event.x + radius, event.y + radius,
            outline="#00e5ff" if self.active_remove_tool == "eraser" else ("#7dd3fc" if self.active_remove_tool == "blur_brush" else "#ff3b30"), width=2, tags=("brush_preview",)
        )
        self.object_canvas.tag_raise(self.brush_preview_id)

    def hide_brush_preview(self, event=None):
        if self.brush_preview_id is not None:
            try:
                self.object_canvas.delete(self.brush_preview_id)
            except Exception:
                pass
            self.brush_preview_id = None

    def paint_object_mask(self, event):
        if self.object_result is None or self.object_mask is None:
            return
        scale = self.object_display_scale
        ox, oy = self.object_display_offset
        x = int((event.x - ox) / scale)
        y = int((event.y - oy) / scale)
        h, w = self.object_mask.shape
        if 0 <= x < w and 0 <= y < h:
            radius = max(1, int(self.brush_size.get() / max(scale, 1e-6) / 2))
            prev = getattr(self, "_last_brush_point", None)
            if prev is not None:
                cv2.line(self.object_mask, prev, (x, y), 255, radius * 2, cv2.LINE_AA)
            cv2.circle(self.object_mask, (x, y), radius, 255, -1, cv2.LINE_AA)
            self._last_brush_point = (x, y)
            # Keep every legacy mask reference synchronized with the visible mask.
            # Several editing commands still read selection_mask/object_edit_mask.
            self.selection_mask = self.object_mask.copy()
            self.object_edit_mask = self.object_mask.copy()
            self.refresh_object_canvas()
            self.show_brush_preview(event)

    def erase_object_mask(self, event):
        """Erase the visible selection/brush marking only; never alter image pixels."""
        if self.object_result is None:
            return

        # Recover the canonical visible mask even when an older selection tool
        # populated only selection_mask/object_edit_mask.
        base = None
        for candidate in (getattr(self, "object_mask", None),
                          getattr(self, "selection_mask", None),
                          getattr(self, "object_edit_mask", None)):
            if isinstance(candidate, np.ndarray) and candidate.shape[:2] == self.object_result.shape[:2]:
                if base is None:
                    base = candidate.copy()
                else:
                    base = cv2.bitwise_or(base, candidate)
        if base is None:
            base = np.zeros(self.object_result.shape[:2], dtype=np.uint8)
        self.object_mask = base

        scale = max(self.object_display_scale, 1e-6)
        ox, oy = self.object_display_offset
        x = int((event.x - ox) / scale)
        y = int((event.y - oy) / scale)
        h, w = self.object_mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        radius = max(1, int(self.brush_size.get() / scale / 2))
        prev = getattr(self, "_last_brush_point", None)
        if prev is not None:
            cv2.line(self.object_mask, prev, (x, y), 0, radius * 2, cv2.LINE_AA)
        cv2.circle(self.object_mask, (x, y), radius, 0, -1, cv2.LINE_AA)
        self._last_brush_point = (x, y)

        # One source of truth: all selection/edit commands now see exactly the
        # same mask, so no stale green outline can remain after erasing.
        self.selection_mask = self.object_mask.copy()
        self.object_edit_mask = self.object_mask.copy()
        self.refresh_object_canvas()
        self.show_brush_preview(event)

    def clear_object_mask(self):
        if self.object_mask is not None:
            self.object_mask[:] = 0
            self.object_edit_mask = None
            self.object_select_mode = False
            self.object_select_points = []
            self.edit_tool = "brush"
            if hasattr(self, "object_select_status"):
                self.object_select_status.config(text="선택을 지웠습니다.")
            self.refresh_object_canvas()

    def run_object_removal(self):
        if self.object_result is None:
            messagebox.showwarning(APP_TITLE, "먼저 사진을 열어주세요.")
            return
        if self.object_mask is None or not np.any(self.object_mask):
            messagebox.showwarning(APP_TITLE, "삭제할 물체를 먼저 마우스로 칠해주세요.")
            return
        self.object_history.append(self.object_result.copy())

        if self.clone_source_point is not None:
            self.object_result = clone_texture_into_mask(
                self.object_result,
                self.object_mask,
                self.clone_source_point,
                int(round(self.remove_strength.get()))
            )
        else:
            self.object_result = remove_masked_object(
                self.object_result,
                self.object_mask,
                int(round(self.inpaint_radius.get())),
                int(round(self.remove_strength.get()))
            )

        self.object_mask[:] = 0
        self.object_edit_mask = None
        self.refresh_object_canvas()

    def toggle_object_compare(self):
        """Toggle the canvas between untouched original and current edited result."""
        if self.object_bgr is None or self.object_result is None:
            messagebox.showwarning(APP_TITLE, "먼저 사진을 열어주세요.")
            return
        self.object_compare_original = not getattr(self, "object_compare_original", False)
        btn = getattr(self, "_tool_buttons", {}).get("compare")
        if btn is not None:
            try:
                btn.configure(bg="#2387f3" if self.object_compare_original else "#343b42",
                              highlightbackground="#66b3ff" if self.object_compare_original else "#4a535c")
            except Exception:
                pass
        if hasattr(self, "object_select_status"):
            self.object_select_status.config(text="원본 보기" if self.object_compare_original else "편집 결과 보기")
        self.refresh_object_canvas()

    def undo_object_removal(self):
        if getattr(self, "object_layers", None):
            idx = self.active_layer_index if self.active_layer_index is not None else len(self.object_layers)-1
            if 0 <= idx < len(self.object_layers):
                self.object_layers.pop(idx)
                self.active_layer_index = None
                self.object_result = self._compose_layers()
                if self.object_mask is not None: self.object_mask[:] = 0
                self._refresh_layer_list(); self.refresh_object_canvas()
                return
        if self.object_history:
            item = self.object_history.pop()
            if isinstance(item, tuple) and len(item) == 3 and item[0] == "crop":
                self.object_result = item[1]
                self.object_bgr = item[2]
            else:
                self.object_result = item
            if self.object_result is not None:
                h, w = self.object_result.shape[:2]
                self.object_mask = np.zeros((h,w), dtype=np.uint8)
                self.selection_mask = self.object_mask.copy()
            self.object_edit_mask = None
            self.object_compare_original = False
            self.refresh_object_canvas()

    def restore_object_original(self):
        if self.object_bgr is not None:
            base = self.object_full_original if isinstance(getattr(self, "object_full_original", None), np.ndarray) else self.object_bgr
            self.object_bgr = base.copy()
            self.object_result = base.copy()
            self.object_history = []
            self.object_layers = []
            self.active_layer_index = None
            self.layer_background = base.copy()
            self._refresh_layer_list()
            if self.object_mask is not None:
                self.object_mask[:] = 0
            self.object_edit_mask = None
            self.refresh_object_canvas()

    def save_object_result(self):
        if self.object_result is None:
            messagebox.showwarning(APP_TITLE, "저장할 결과가 없습니다.")
            return
        stem = Path(self.object_image_path).stem if self.object_image_path else "edited"
        ext = Path(self.object_image_path).suffix if self.object_image_path else ".jpg"
        p = filedialog.asksaveasfilename(
            initialfile=f"{stem}_object_removed{ext}",
            defaultextension=ext,
            filetypes=[("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        if p:
            try:
                save_image(p, self.object_result)
                messagebox.showinfo(APP_TITLE, "저장되었습니다.")
            except Exception as e:
                messagebox.showerror(APP_TITLE, f"저장 실패\n{e}")

    def add_slider(self, parent, label, var, lo, hi, suffix=""):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=15).pack(side="left")
        scale = ttk.Scale(row, from_=lo, to=hi, variable=var, length=230,
                          command=lambda _=None: self.schedule_preview())
        scale.pack(side="left", padx=5)
        value = ttk.Label(row, width=6, anchor="e")
        value.pack(side="left")

        def update_label(*_):
            value.config(text=f"{int(round(var.get()))}{suffix}")
        var.trace_add("write", update_label)
        update_label()

    def choose_reference(self):
        p = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        if p:
            self.ref_path.set(p)
            self.schedule_preview()

    def add_files(self):
        fs = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        self._append_files(fs)

    def add_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        paths = [str(p) for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
        self._append_files(paths)

    def _append_files(self, paths):
        existing = set(self.files)
        for p in paths:
            if p not in existing:
                self.files.append(p)
                self.listbox.insert("end", p)
                existing.add(p)
        if self.files and not self.listbox.curselection():
            self.listbox.selection_set(0)
        self.schedule_preview()

    def clear_files(self):
        self.files.clear()
        self.listbox.delete(0, "end")
        self.before_label.config(image="")
        self.after_label.config(image="")

    def choose_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output_dir.set(p)

    def reset_adjustments(self):
        self.match_strength.set(85)
        for v in (self.brightness, self.contrast, self.saturation, self.temperature,
                  self.tint, self.highlights, self.shadows, self.sharpness):
            v.set(0)
        self.schedule_preview()

    def schedule_preview(self):
        if self._preview_job:
            try:
                self.after_cancel(self._preview_job)
            except Exception:
                pass
        self._preview_job = self.after(180, self.update_preview)

    def _selected_source(self):
        sel = self.listbox.curselection()
        if sel:
            return self.files[sel[0]]
        return self.files[0] if self.files else None

    def _process_one(self, bgr, ref_mean, ref_std):
        result = color_transfer(bgr, ref_mean, ref_std, self.match_strength.get() / 100.0)
        return apply_manual_adjustments(
            result,
            self.brightness.get(), self.contrast.get(), self.saturation.get(),
            self.temperature.get(), self.tint.get(), self.highlights.get(),
            self.shadows.get(), self.sharpness.get()
        )

    def update_preview(self):
        src = self._selected_source()
        ref = self.ref_path.get()
        if not src or not ref or not Path(src).exists() or not Path(ref).exists():
            return
        try:
            src_bgr, *_ = read_image(src)
            ref_bgr, *_ = read_image(ref)
            ref_mean, ref_std = lab_stats(ref_bgr)

            # Downscale before processing for responsive preview.
            h, w = src_bgr.shape[:2]
            max_side = 700
            s = min(1.0, max_side / max(h, w))
            if s < 1:
                src_small = cv2.resize(src_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
            else:
                src_small = src_bgr
            result = self._process_one(src_small, ref_mean, ref_std)

            self._show_preview(self.before_label, src_small, "before")
            self._show_preview(self.after_label, result, "after")
        except Exception as e:
            self.status.config(text=f"미리보기 오류: {e}")

    def _show_preview(self, label, bgr, which):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail((380, 360), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(im)
        label.config(image=photo)
        if which == "before":
            self.preview_photo_before = photo
        else:
            self.preview_photo_after = photo

    def _ctrl_save(self, event=None):
        self.save_current_color_result()
        return "break"

    def save_current_color_result(self):
        src = self._selected_source()
        ref = self.ref_path.get()
        if not src or not Path(src).exists():
            messagebox.showwarning(APP_TITLE, "저장할 보정 사진을 선택해주세요.")
            return
        if not ref or not Path(ref).exists():
            messagebox.showwarning(APP_TITLE, "기준 사진을 선택해주세요.")
            return
        src_path = Path(src)
        ext = src_path.suffix.lower()
        if ext not in SUPPORTED:
            ext = ".jpg"
        p = filedialog.asksaveasfilename(
            title="현재 보정 결과 저장",
            initialdir=self.output_dir.get() if Path(self.output_dir.get()).exists() else str(src_path.parent),
            initialfile=f"{src_path.stem}_matched{ext}",
            defaultextension=ext,
            filetypes=[("원본 형식", f"*{ext}"), ("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png"), ("모든 파일", "*.*")],
        )
        if not p:
            return
        try:
            bgr, exif, icc, dpi = read_image(src_path)
            ref_bgr, *_ = read_image(ref)
            ref_mean, ref_std = lab_stats(ref_bgr)
            result = self._process_one(bgr, ref_mean, ref_std)
            save_image(p, result, exif=exif, icc=icc, dpi=dpi, jpeg_quality=98)
            self.output_dir.set(str(Path(p).parent))
            self.status.config(text=f"저장 완료: {Path(p).name}")
            messagebox.showinfo(APP_TITLE, f"현재 보정 결과를 저장했습니다.\n\n{p}")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"저장 중 오류가 발생했습니다.\n{e}")

    def start_processing(self):
        if not self.ref_path.get() or not Path(self.ref_path.get()).exists():
            messagebox.showwarning(APP_TITLE, "기준 사진을 선택해주세요.")
            return
        if not self.files:
            messagebox.showwarning(APP_TITLE, "보정할 사진을 추가해주세요.")
            return
        if not self.output_dir.get():
            messagebox.showwarning(APP_TITLE, "출력 폴더를 선택해주세요.")
            return
        threading.Thread(target=self.process_all, daemon=True).start()

    def process_all(self):
        try:
            self.after(0, lambda: self.status.config(text="처리 중..."))
            ref_bgr, *_ = read_image(self.ref_path.get())
            ref_mean, ref_std = lab_stats(ref_bgr)
            output_root = Path(self.output_dir.get())
            output_root.mkdir(parents=True, exist_ok=True)

            common_parent = None
            if self.keep_subfolders.get() and self.files:
                try:
                    common_parent = Path(os.path.commonpath([str(Path(p).parent) for p in self.files]))
                except Exception:
                    pass

            ok, errors = 0, []
            total = len(self.files)
            for i, src in enumerate(self.files, 1):
                try:
                    src_path = Path(src)
                    bgr, exif, icc, dpi = read_image(src_path)
                    result = self._process_one(bgr, ref_mean, ref_std)

                    rel_parent = Path()
                    if self.keep_subfolders.get() and common_parent:
                        try:
                            rel_parent = src_path.parent.relative_to(common_parent)
                        except Exception:
                            pass
                    dest_dir = output_root / rel_parent
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / f"{src_path.stem}_matched{src_path.suffix}"
                    save_image(dest, result, exif=exif, icc=icc, dpi=dpi, jpeg_quality=98)
                    ok += 1
                except Exception as e:
                    errors.append(f"{src}: {e}")

                pct = i * 100 / total
                self.after(0, lambda p=pct, i=i, n=src_path.name:
                           (self.progress.config(value=p), self.status.config(text=f"{i}/{total}  {n}")))

            def done():
                if errors:
                    messagebox.showwarning(APP_TITLE, f"완료: {ok}개\n오류: {len(errors)}개\n\n" + "\n".join(errors[:8]))
                else:
                    messagebox.showinfo(APP_TITLE, f"완료되었습니다.\n{ok}개 사진을 보정했습니다.")
                self.status.config(text=f"완료 - {ok}/{total}")
            self.after(0, done)
        except Exception:
            err = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror(APP_TITLE, err))


if __name__ == "__main__":
    app = PhotoColorMatcherApp()
    app.mainloop()
