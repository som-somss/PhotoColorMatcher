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

        self._build_ui()

    def _build_ui(self):
        pad = 10
        top = ttk.Frame(self, padding=pad)
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
        ttk.Checkbutton(controls, text="폴더 추가 시 하위 폴더 구조 유지",
                        variable=self.keep_subfolders).pack(anchor="w", pady=(6, 0))

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
        ttk.Button(bottom, text="자동 색감 보정 시작", command=self.start_processing).pack(fill="x", ipady=8)

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
                    save_image(dest, result, exif=exif, icc=icc, dpi=dpi)
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
