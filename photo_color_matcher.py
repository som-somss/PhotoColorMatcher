
import os
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageOps

APP_TITLE = "Photo Color Matcher"

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def read_image(path):
    """Unicode-safe image read with EXIF orientation applied."""
    path = str(path)
    with Image.open(path) as im:
        exif = im.getexif()
        icc = im.info.get("icc_profile")
        dpi = im.info.get("dpi")
        im = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.array(im)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr, exif, icc, dpi

def save_image(path, bgr, exif=None, icc=None, dpi=None, jpeg_quality=95):
    """Save via Pillow for Unicode paths and metadata retention where possible."""
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
    mean, std = cv2.meanStdDev(lab)
    return mean.reshape(3), std.reshape(3)

def color_transfer(source_bgr, ref_mean, ref_std, strength=1.0):
    """
    Reinhard-style color transfer in LAB.
    strength 0..1 blends original and matched result.
    """
    lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)
    src_mean = flat.mean(axis=0)
    src_std = flat.std(axis=0)
    src_std = np.maximum(src_std, 1e-6)

    matched = (lab - src_mean) * (ref_std / src_std) + ref_mean
    matched = np.clip(matched, 0, 255)

    if strength < 1.0:
        matched = lab * (1.0 - strength) + matched * strength

    matched = np.clip(matched, 0, 255).astype(np.uint8)
    return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("760x560")
        self.minsize(700, 520)

        self.reference = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "matched_output"))
        self.strength = tk.DoubleVar(value=85)
        self.keep_subfolders = tk.BooleanVar(value=True)
        self.files = []
        self.running = False

        self._build()

    def _build(self):
        pad = 10

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="기준 사진").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.reference).grid(row=1, column=0, sticky="ew", padx=(0,8))
        ttk.Button(frm, text="기준 사진 선택", command=self.pick_reference).grid(row=1, column=1, sticky="ew")

        ttk.Separator(frm).grid(row=2, column=0, columnspan=2, sticky="ew", pady=12)

        ttk.Label(frm, text="보정할 사진").grid(row=3, column=0, sticky="w")
        buttons = ttk.Frame(frm)
        buttons.grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(buttons, text="파일 추가", command=self.add_files).pack(side="left", padx=(0,6))
        ttk.Button(buttons, text="폴더 추가", command=self.add_folder).pack(side="left", padx=(0,6))
        ttk.Button(buttons, text="목록 비우기", command=self.clear_files).pack(side="left")

        self.listbox = tk.Listbox(frm, height=10, selectmode="extended")
        self.listbox.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(6,10))

        ttk.Label(frm, text="출력 폴더").grid(row=6, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.output_dir).grid(row=7, column=0, sticky="ew", padx=(0,8))
        ttk.Button(frm, text="폴더 선택", command=self.pick_output).grid(row=7, column=1, sticky="ew")

        options = ttk.LabelFrame(frm, text="옵션", padding=10)
        options.grid(row=8, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(options, text="색감 매칭 강도").grid(row=0, column=0, sticky="w")
        ttk.Scale(options, from_=0, to=100, variable=self.strength, orient="horizontal",
                  command=self._update_strength_label).grid(row=0, column=1, sticky="ew", padx=8)
        self.strength_label = ttk.Label(options, text="85%")
        self.strength_label.grid(row=0, column=2, sticky="e")
        ttk.Checkbutton(options, text="폴더 추가 시 하위 폴더 구조 유지", variable=self.keep_subfolders)\
            .grid(row=1, column=0, columnspan=3, sticky="w", pady=(8,0))
        options.columnconfigure(1, weight=1)

        self.progress = ttk.Progressbar(frm, mode="determinate")
        self.progress.grid(row=9, column=0, columnspan=2, sticky="ew")

        self.status = ttk.Label(frm, text="준비됨")
        self.status.grid(row=10, column=0, columnspan=2, sticky="w", pady=(6,8))

        self.run_btn = ttk.Button(frm, text="자동 색감 보정 시작", command=self.start)
        self.run_btn.grid(row=11, column=0, columnspan=2, sticky="ew", ipady=6)

        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(5, weight=1)

    def _update_strength_label(self, _=None):
        self.strength_label.config(text=f"{int(self.strength.get())}%")

    def pick_reference(self):
        p = filedialog.askopenfilename(
            title="기준 사진 선택",
            filetypes=[("이미지", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("모든 파일", "*.*")]
        )
        if p:
            self.reference.set(p)

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="보정할 사진 선택",
            filetypes=[("이미지", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("모든 파일", "*.*")]
        )
        self._append_files(paths)

    def add_folder(self):
        folder = filedialog.askdirectory(title="사진 폴더 선택")
        if not folder:
            return
        folder = Path(folder)
        paths = [str(p) for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
        self._append_files(paths)

    def _append_files(self, paths):
        existing = set(self.files)
        for p in paths:
            if p not in existing and Path(p).suffix.lower() in SUPPORTED:
                self.files.append(p)
                self.listbox.insert("end", p)
                existing.add(p)

    def clear_files(self):
        self.files.clear()
        self.listbox.delete(0, "end")

    def pick_output(self):
        p = filedialog.askdirectory(title="출력 폴더 선택")
        if p:
            self.output_dir.set(p)

    def start(self):
        if self.running:
            return
        ref = Path(self.reference.get())
        if not ref.exists():
            messagebox.showerror("오류", "기준 사진을 선택해 주세요.")
            return
        if not self.files:
            messagebox.showerror("오류", "보정할 사진을 추가해 주세요.")
            return
        out = Path(self.output_dir.get())
        out.mkdir(parents=True, exist_ok=True)

        self.running = True
        self.run_btn.config(state="disabled")
        self.progress["value"] = 0
        self.progress["maximum"] = len(self.files)
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        try:
            ref_bgr, _, _, _ = read_image(self.reference.get())
            ref_mean, ref_std = lab_stats(ref_bgr)
            strength = max(0.0, min(1.0, self.strength.get() / 100.0))
            output_root = Path(self.output_dir.get())

            common_parent = None
            try:
                common_parent = Path(os.path.commonpath([str(Path(p).parent) for p in self.files]))
            except Exception:
                pass

            ok = 0
            errors = []

            for i, src in enumerate(self.files, 1):
                try:
                    src_path = Path(src)
                    bgr, exif, icc, dpi = read_image(src_path)
                    result = color_transfer(bgr, ref_mean, ref_std, strength)

                    if self.keep_subfolders.get() and common_parent:
                        try:
                            rel_parent = src_path.parent.relative_to(common_parent)
                        except Exception:
                            rel_parent = Path()
                    else:
                        rel_parent = Path()

                    dest_dir = output_root / rel_parent
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    dest = dest_dir / f"{src_path.stem}_matched{src_path.suffix}"
                    save_image(dest, result, exif=exif, icc=icc, dpi=dpi)
                    ok += 1
                except Exception as e:
                    errors.append(f"{src}: {e}")

                self.after(0, self._set_progress, i, src_path.name)

            msg = f"완료: {ok}개 사진 보정"
            if errors:
                msg += f"\n오류: {len(errors)}개"
                log = output_root / "errors.txt"
                log.write_text("\n".join(errors), encoding="utf-8")

            self.after(0, self._finish, msg, bool(errors))
        except Exception as e:
            detail = traceback.format_exc()
            self.after(0, self._fatal, str(e), detail)

    def _set_progress(self, i, name):
        self.progress["value"] = i
        self.status.config(text=f"{i}/{len(self.files)} 처리 중: {name}")

    def _finish(self, msg, had_errors):
        self.running = False
        self.run_btn.config(state="normal")
        self.status.config(text="완료")
        if had_errors:
            messagebox.showwarning("완료", msg + "\n출력 폴더의 errors.txt를 확인해 주세요.")
        else:
            messagebox.showinfo("완료", msg)

    def _fatal(self, msg, detail):
        self.running = False
        self.run_btn.config(state="normal")
        self.status.config(text="오류 발생")
        try:
            Path(self.output_dir.get()).mkdir(parents=True, exist_ok=True)
            (Path(self.output_dir.get()) / "fatal_error.txt").write_text(detail, encoding="utf-8")
        except Exception:
            pass
        messagebox.showerror("오류", msg)

if __name__ == "__main__":
    App().mainloop()
