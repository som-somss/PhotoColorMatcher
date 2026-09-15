import os
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk, ImageEnhance

APP_TITLE = "Photo Color Matcher Pro"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


def remove_masked_object(bgr, mask, radius=5):
    """Offline object removal using OpenCV inpainting."""
    if mask is None or not np.any(mask):
        return bgr.copy()
    mask8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    # Slight dilation helps remove object edges as well.
    kernel = np.ones((3, 3), np.uint8)
    mask8 = cv2.dilate(mask8, kernel, iterations=1)
    return cv2.inpaint(bgr, mask8, float(radius), cv2.INPAINT_TELEA)


class PhotoColorMatcherApp(tk.Tk):
    def __init__(self):
        super().__init__()
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
        self.object_history = []
        self.brush_size = tk.DoubleVar(value=35)
        self.inpaint_radius = tk.DoubleVar(value=5)

        self._build_ui()
        self.bind_all("<Control-s>", self._ctrl_save)
        self.bind_all("<Control-S>", self._ctrl_save)

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

        bar = ttk.Frame(wrap)
        bar.pack(fill="x")
        ttk.Button(bar, text="사진 열기", command=self.open_object_image).pack(side="left")
        ttk.Button(bar, text="선택 영역 삭제", command=self.run_object_removal).pack(side="left", padx=6)
        ttk.Button(bar, text="마스크 지우기", command=self.clear_object_mask).pack(side="left")
        ttk.Button(bar, text="실행 취소", command=self.undo_object_removal).pack(side="left", padx=6)
        ttk.Button(bar, text="원본 복원", command=self.restore_object_original).pack(side="left")
        ttk.Button(bar, text="결과 저장", command=self.save_object_result).pack(side="right")

        opts = ttk.Frame(wrap)
        opts.pack(fill="x", pady=(8, 5))
        ttk.Label(opts, text="브러시 크기").pack(side="left")
        ttk.Scale(opts, from_=5, to=120, variable=self.brush_size, length=220).pack(side="left", padx=(6, 18))
        ttk.Label(opts, text="복원 범위").pack(side="left")
        ttk.Scale(opts, from_=1, to=15, variable=self.inpaint_radius, length=180).pack(side="left", padx=6)
        ttk.Label(opts, text="삭제할 물체를 마우스로 칠한 뒤 '선택 영역 삭제'를 누르세요.").pack(side="left", padx=14)

        self.object_canvas = tk.Canvas(wrap, bg="#333333", highlightthickness=0, cursor="crosshair")
        self.object_canvas.pack(fill="both", expand=True)
        self.object_canvas.bind("<Button-1>", self.paint_object_mask)
        self.object_canvas.bind("<B1-Motion>", self.paint_object_mask)
        self.object_canvas.bind("<Configure>", lambda e: self.refresh_object_canvas())

    def open_object_image(self):
        p = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        if not p:
            return
        try:
            bgr, *_ = read_image(p)
            self.object_image_path = p
            self.object_bgr = bgr
            self.object_result = bgr.copy()
            self.object_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            self.object_history = []
            self.refresh_object_canvas()
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"사진을 열 수 없습니다.\n{e}")

    def refresh_object_canvas(self):
        if self.object_result is None or not hasattr(self, "object_canvas"):
            return
        self.object_canvas.update_idletasks()
        cw = max(100, self.object_canvas.winfo_width())
        ch = max(100, self.object_canvas.winfo_height())
        h, w = self.object_result.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        ox, oy = (cw - dw)//2, (ch - dh)//2
        self.object_display_scale = scale
        self.object_display_offset = (ox, oy)

        rgb = cv2.cvtColor(self.object_result, cv2.COLOR_BGR2RGB)
        disp = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        if self.object_mask is not None and np.any(self.object_mask):
            m = cv2.resize(self.object_mask, (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
            overlay = disp.copy()
            overlay[m] = [255, 55, 55]
            disp = np.where(m[..., None], (0.55*overlay + 0.45*disp).astype(np.uint8), disp)

        im = Image.fromarray(disp)
        self.object_display_photo = ImageTk.PhotoImage(im)
        self.object_canvas.delete("all")
        self.object_canvas.create_image(ox, oy, anchor="nw", image=self.object_display_photo)

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
            cv2.circle(self.object_mask, (x, y), radius, 255, -1)
            self.refresh_object_canvas()

    def clear_object_mask(self):
        if self.object_mask is not None:
            self.object_mask[:] = 0
            self.refresh_object_canvas()

    def run_object_removal(self):
        if self.object_result is None:
            messagebox.showwarning(APP_TITLE, "먼저 사진을 열어주세요.")
            return
        if self.object_mask is None or not np.any(self.object_mask):
            messagebox.showwarning(APP_TITLE, "삭제할 물체를 먼저 마우스로 칠해주세요.")
            return
        self.object_history.append(self.object_result.copy())
        self.object_result = remove_masked_object(
            self.object_result, self.object_mask, int(round(self.inpaint_radius.get()))
        )
        self.object_mask[:] = 0
        self.refresh_object_canvas()

    def undo_object_removal(self):
        if self.object_history:
            self.object_result = self.object_history.pop()
            if self.object_mask is not None:
                self.object_mask[:] = 0
            self.refresh_object_canvas()

    def restore_object_original(self):
        if self.object_bgr is not None:
            self.object_result = self.object_bgr.copy()
            self.object_history = []
            if self.object_mask is not None:
                self.object_mask[:] = 0
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
