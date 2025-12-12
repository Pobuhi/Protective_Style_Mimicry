import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import lpips
from scipy.fftpack import dctn
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread
import time
import pywt

home_dir = os.path.expanduser("~")
desktop_dir = os.path.join(home_dir, "Desktop")
if not os.path.isdir(desktop_dir):
    desktop_dir = home_dir
base_dir = os.path.join(desktop_dir, "ProtectionStudio")
os.makedirs(base_dir, exist_ok=True)
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

LPIPS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='vgg').to(LPIPS_DEVICE).eval()
_to_tensor_01 = transforms.ToTensor()


def validate_image_file(filepath: str):
    """Return a valid PIL image or raise an exception."""
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError("Selected file does not exist.")

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    if not filepath.lower().endswith(valid_ext):
        raise ValueError("Unsupported file type. Please select a valid image file.")

    try:
        with Image.open(filepath) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Selected file is not a valid image: {e}")

    return Image.open(filepath).convert("RGB")


def pil_to_lpips_tensor(pil_im: Image.Image) -> torch.Tensor:
    t = _to_tensor_01(pil_im.convert("RGB"))
    t = t * 2.0 - 1.0
    return t.unsqueeze(0).to(LPIPS_DEVICE)


def lpips_tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(-1, 1)
    t = (t + 1.0) / 2.0
    t = t.squeeze(0).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


@torch.inference_mode()
def lpips_distance(a_pil: Image.Image, b_pil: Image.Image) -> float:
    a = pil_to_lpips_tensor(a_pil)
    b = pil_to_lpips_tensor(b_pil)
    return float(lpips_model(a, b).squeeze().cpu().item())

def compute_dwt_frequency_map(image_pil: Image.Image, wavelet='haar'):

    img_array = np.array(image_pil.convert("RGB")).astype(np.float32)
    h, w = image_pil.size

    energy = np.zeros((h, w), dtype=np.float32)

    for c in range(3):
        cA, (cH, cV, cD) = pywt.dwt2(img_array[:, :, c], wavelet)
        hf = np.sqrt(cH**2 + cV**2 + cD**2)

        hf_resized = np.array(Image.fromarray(hf).resize((w, h), Image.BILINEAR), dtype=np.float32)

        energy += hf_resized
        
    energy /= 3.0

    energy -= energy.min()
    energy /= (energy.max() + 1e-8)

    return energy



def compute_dct_frequency_map(image_pil: Image.Image, block_size=16) -> np.ndarray:
    img_array = np.array(image_pil.convert('L')).astype(float)
    h, w = img_array.shape

    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    img_padded = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='reflect')

    h_padded, w_padded = img_padded.shape
    energy_map = np.zeros((h_padded, w_padded))

    for i in range(0, h_padded, block_size):
        for j in range(0, w_padded, block_size):
            block = img_padded[i:i + block_size, j:j + block_size]
            dct_block = dctn(block, norm='ortho')

            freq_weight = np.zeros_like(dct_block)
            for y in range(block_size):
                for x in range(block_size):
                    freq_weight[y, x] = np.sqrt(x ** 2 + y ** 2)

            hf_energy = np.sum(np.abs(dct_block) * freq_weight)
            energy_map[i:i + block_size, j:j + block_size] = hf_energy

    energy_map = energy_map[:h, :w]
    energy_map = (energy_map - energy_map.min()) / (energy_map.max() - energy_map.min() + 1e-8)
    energy_map = gaussian_filter(energy_map, sigma=block_size)
    energy_map = np.power(energy_map, 0.6)

    return energy_map


def pgd_glaze_dct_adaptive(
        content_pil: Image.Image,
        target_style_pil: Image.Image,
        steps=150,
        step_size=6 / 255,
        linf_budget_base=18 / 255,
        lpips_budget=0.14,
        style_shift_strength=9.0,
        frequency_weight=0.80,
        dct_block_size=16,
        target_style_shift_percent: float = 37.0,
        progress_callback=None,
        compression_method = "DCT"
        ):
    
    # content_np = np.array(content_pil.convert("RGB")).astype(np.float32)
    if compression_method == "DCT":
        energy_map = compute_dct_frequency_map(content_pil, block_size=dct_block_size)
        
    elif compression_method == "DWT":
        energy_map = compute_dwt_frequency_map(content_pil)
    else:
        raise ValueError("Invalid compression type selected.")
    energy_tensor = torch.from_numpy(energy_map).float().unsqueeze(0).unsqueeze(0)
    energy_tensor = 0.5 + (frequency_weight - 0.5) * energy_tensor

    x0 = pil_to_lpips_tensor(content_pil).detach()
    x_target = pil_to_lpips_tensor(target_style_pil).detach()

    _, _, h, w = x0.shape
    energy_tensor_resized = torch.nn.functional.interpolate(
        energy_tensor, size=(h, w), mode='bilinear', align_corners=False
    ).to(LPIPS_DEVICE)

    with torch.no_grad():
        baseline_target_dist = lpips_model(x0, x_target).mean()

    init_scale = linf_budget_base / 3.0
    delta = torch.empty_like(x0).uniform_(-init_scale, init_scale)
    delta.requires_grad_()

    optim = torch.optim.Adam([delta], lr=step_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps, eta_min=step_size * 0.1)

    for i in range(steps):
        optim.zero_grad()
        x_adv = (x0 + delta).clamp(-1, 1)

        dist_from_orig = lpips_model(x_adv, x0).mean()
        dist_to_target = lpips_model(x_adv, x_target).mean()

        target_frac = target_style_shift_percent / 100.0
        style_shift_frac = (baseline_target_dist - dist_to_target) / (baseline_target_dist + 1e-8)
        shift_loss = (style_shift_frac - target_frac) ** 2
        imperceptibility_penalty = torch.relu(dist_from_orig - lpips_budget)
        loss = style_shift_strength * shift_loss + 5.0 * imperceptibility_penalty

        loss.backward()
        optim.step()
        scheduler.step()

        with torch.no_grad():
            adaptive_budget = linf_budget_base * energy_tensor_resized
            delta.data = torch.clamp(delta.data, -adaptive_budget, adaptive_budget)
            delta.data = torch.clamp(x0 + delta.data, -1, 1) - x0

        if progress_callback and (i % 10 == 0 or i == steps - 1):
            progress_callback(i + 1, steps)

        if i % 40 == 0 and LPIPS_DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    with torch.no_grad():
        adaptive_budget = linf_budget_base * energy_tensor_resized
        delta.data = torch.clamp(delta.data, -adaptive_budget, adaptive_budget)
        delta.data = torch.clamp(x0 + delta.data, -1, 1) - x0
        x_final = (x0 + delta.data).clamp(-1, 1)

    x_final_pil = lpips_tensor_to_pil(x_final)
    final_dist_orig = float(lpips_model(x_final, x0).squeeze().cpu().item())
    final_dist_target = float(lpips_model(x_final, x_target).squeeze().cpu().item())

    if LPIPS_DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return x_final_pil, final_dist_orig, final_dist_target


class GlazeProtectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Mimicry Prevention Mechanism")
        self.root.geometry("900x800")
        self.root.configure(bg='#f8fafc')
        self.tune_hours = 0
        self.is_fine_tuning = False

        self.content_image = None
        self.style_image = None
        self.protected_image = None
        self.is_processing = False

        self.setup_ui()

    def setup_ui(self):
        title_frame = tk.Frame(self.root, bg='#f8fafc')
        title_frame.pack(pady=20)

        title_label = tk.Label(
            title_frame,
            text="AI Mimicry Prevention Mechanism",
            font=("Arial", 24, "bold"),
            bg='#f8fafc',
            fg='#1e293b'
        )
        title_label.pack()

        subtitle_label = tk.Label(
            title_frame,
            text="Protect your artwork from AI style mimicry with adaptive DCT frequency-based cloaking",
            font=("Arial", 10),
            bg='#f8fafc',
            fg='#64748b'
        )
        subtitle_label.pack()

        main_frame = tk.Frame(self.root, bg='#f8fafc')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        strength_frame = tk.LabelFrame(
            main_frame,
            text="Protection Strength Control",
            font=("Arial", 12, "bold"),
            bg='white',
            fg='#1e293b',
            relief=tk.RAISED,
            bd=2
        )
        strength_frame.pack(fill=tk.X, pady=10)

        # Compression Type Option
        compression_label = ttk.Label(strength_frame,
                                       text="Compression Type:",
                                         font=("Arial", 10))
        compression_label.pack()
        
        self.compression_var = tk.StringVar()
        self.compression_var.set("DCT") # Default value

        compression_menu = ttk.Combobox(strength_frame,
                                        textvariable=self.compression_var,
                                        values=["DCT", "DWT"],
                                        state="readonly",
                                        font=("Arial", 10))
        compression_menu.pack()


        # Target Style Shift Slider
        slider_container = tk.Frame(strength_frame, bg='white')
        slider_container.pack(fill=tk.X, padx=20, pady=15)

        slider_label_frame = tk.Frame(slider_container, bg='white')
        slider_label_frame.pack(fill=tk.X)

        self.slider_label = tk.Label(
            slider_label_frame,
            text="Target Style Shift: 40%",
            font=("Arial", 11, "bold"),
            bg='white',
            fg='#1e293b'
        )
        self.slider_label.pack(side=tk.LEFT)

        self.protection_level_label = tk.Label(
            slider_label_frame,
            text="STRONG",
            font=("Arial", 9, "bold"),
            bg='#dcfce7',
            fg='#16a34a',
            padx=10,
            pady=3,
            relief=tk.RAISED
        )
        self.protection_level_label.pack(side=tk.RIGHT)

        self.slider = tk.Scale(
            slider_container,
            from_=5,
            to=85,
            orient=tk.HORIZONTAL,
            command=self.on_slider_change,
            length=500,
            bg='white',
            highlightthickness=0,
            troughcolor='#e2e8f0',
            activebackground='#6366f1'
        )
        self.slider.set(40)
        self.slider.pack(pady=10)

        markers_frame = tk.Frame(slider_container, bg='white')
        markers_frame.pack(fill=tk.X)

        markers = [
            ("5%\nMinimal", 0),
            ("17%\nWeak", 0.15),
            ("26%\nModerate", 0.2625),
            ("38%\nMod-Strong", 0.4125),
            ("46%\nStrong", 0.5125),
            ("59%\nVery Strong", 0.675),
            ("68%\nExtreme", 0.7875),
            ("85%\nMaximum", 1.0)
        ]

        for label, pos in markers:
            marker_label = tk.Label(
                markers_frame,
                text=label,
                font=("Arial", 7),
                bg='white',
                fg='#64748b'
            )
            marker_label.place(relx=pos, anchor='n')

        results_frame = tk.Frame(strength_frame, bg='white')
        results_frame.pack(fill=tk.X, padx=20, pady=10)

        self.lpips_frame = tk.Frame(results_frame, bg='#dbeafe', relief=tk.RAISED, bd=2)
        self.lpips_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(
            self.lpips_frame,
            text="📊 Expected LPIPS",
            font=("Arial", 9, "bold"),
            bg='#dbeafe',
            fg='#1e40af'
        ).pack(pady=5)

        self.lpips_value_label = tk.Label(
            self.lpips_frame,
            text="0.250",
            font=("Arial", 18, "bold"),
            bg='#dbeafe',
            fg='#1e40af'
        )
        self.lpips_value_label.pack()

        tk.Label(
            self.lpips_frame,
            text="Perceptual distance",
            font=("Arial", 7),
            bg='#dbeafe',
            fg='#1e40af'
        ).pack(pady=5)

        self.visual_frame = tk.Frame(results_frame, bg='#dcfce7', relief=tk.RAISED, bd=2)
        self.visual_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(
            self.visual_frame,
            text="🛡️ Visual Impact",
            font=("Arial", 9, "bold"),
            bg='#dcfce7',
            fg='#15803d'
        ).pack(pady=5)

        self.visual_impact_label = tk.Label(
            self.visual_frame,
            text="Clearly visible\nmodifications",
            font=("Arial", 8, "bold"),
            bg='#dcfce7',
            fg='#15803d',
            wraplength=150
        )
        self.visual_impact_label.pack(pady=10)

        self.steps_frame = tk.Frame(results_frame, bg='#f3e8ff', relief=tk.RAISED, bd=2)
        self.steps_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(
            self.steps_frame,
            text="⚙️ Processing Steps",
            font=("Arial", 9, "bold"),
            bg='#f3e8ff',
            fg='#7c3aed'
        ).pack(pady=5)

        self.steps_value_label = tk.Label(
            self.steps_frame,
            text="300",
            font=("Arial", 18, "bold"),
            bg='#f3e8ff',
            fg='#7c3aed'
        )
        self.steps_value_label.pack()

        tk.Label(
            self.steps_frame,
            text="Optimization iterations",
            font=("Arial", 7),
            bg='#f3e8ff',
            fg='#7c3aed'
        ).pack(pady=5)

        self.show_advanced = tk.BooleanVar(value=False)
        advanced_toggle = tk.Checkbutton(
            strength_frame,
            text="⚙️ Show Advanced Parameters",
            variable=self.show_advanced,
            command=self.toggle_advanced,
            font=("Arial", 9, "bold"),
            bg='white',
            fg='#6366f1',
            selectcolor='white',
            activebackground='white'
        )
        advanced_toggle.pack(pady=10)

        self.advanced_frame = tk.Frame(strength_frame, bg='#f8fafc')

        upload_frame = tk.Frame(main_frame, bg='#f8fafc')
        upload_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        content_frame = tk.LabelFrame(
            upload_frame,
            text="Your Artwork",
            font=("Arial", 11, "bold"),
            bg='white',
            relief=tk.RAISED,
            bd=2
        )
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.content_button = tk.Button(
            content_frame,
            text="📁 Upload Artwork",
            command=lambda: self.upload_image('content'),
            font=("Arial", 10, "bold"),
            bg='#6366f1',
            fg='white',
            relief=tk.RAISED,
            bd=0,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.content_button.pack(pady=20)

        self.content_label = tk.Label(
            content_frame,
            text="No image selected",
            font=("Arial", 9),
            bg='white',
            fg='#64748b'
        )
        self.content_label.pack()

        style_frame = tk.LabelFrame(
            upload_frame,
            text="Target Style (Optional)",
            font=("Arial", 11, "bold"),
            bg='white',
            relief=tk.RAISED,
            bd=2
        )
        style_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.style_button = tk.Button(
            style_frame,
            text="📁 Upload Style Reference",
            command=lambda: self.upload_image('style'),
            font=("Arial", 10, "bold"),
            bg='#a855f7',
            fg='white',
            relief=tk.RAISED,
            bd=0,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.style_button.pack(pady=20)

        self.style_label = tk.Label(
            style_frame,
            text="No image selected",
            font=("Arial", 9),
            bg='white',
            fg='#64748b'
        )
        self.style_label.pack()

        process_frame = tk.Frame(main_frame, bg='#f8fafc')
        process_frame.pack(fill=tk.X, pady=10)

        self.fine_tune_button = tk.Button(
            process_frame,
            text="⚙️ Fine Tune Parameters",
            command=self.open_fine_tune_prompt,
            font=("Arial", 11, "bold"),
            bg='#a78bfa',
            fg='white',
            relief=tk.RAISED,
            bd=0,
            padx=30,
            pady=12,
            cursor='hand2'
        )
        self.fine_tune_button.pack(fill=tk.X, pady=(0, 10))

        self.process_button = tk.Button(
            process_frame,
            text="🛡️Apply DCT Protection",
            command=self.process_protection,
            font=("Arial", 12, "bold"),
            bg='#6366f1',
            fg='white',
            relief=tk.RAISED,
            bd=0,
            padx=30,
            pady=15,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.process_button.pack(fill=tk.X)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            process_frame,
            variable=self.progress_var,
            maximum=100
        )

        self.progress_label = tk.Label(
            process_frame,
            text="",
            font=("Arial", 9),
            bg='#f8fafc',
            fg='#64748b'
        )

        self.results_text = tk.Text(
            main_frame,
            height=8,
            font=("Courier", 9),
            bg='white',
            relief=tk.SUNKEN,
            bd=2
        )

        self.ft_progress_var = tk.DoubleVar()
        self.ft_progress_bar = ttk.Progressbar(
            process_frame,
            variable=self.ft_progress_var,
            maximum=100
        )
        self.ft_progress_label = tk.Label(
            process_frame,
            text="",
            font=("Arial", 9),
            bg='#f8fafc',
            fg='#64748b'
        )

        self.on_slider_change(38)

    def stop_if_exceeded(self, runs_used, max_runs):
        return runs_used >= max_runs

    def open_fine_tune_prompt(self):
        popup = tk.Toplevel(self.root)
        popup.title("Fine Tune Parameters")
        width = 300
        height = 160
        popup.geometry(str(width) + 'x' + str(height))
        popup.configure(bg="white")
        popup.grab_set()

        popup.update_idletasks()
        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        x = (screen_w // 2) - (width // 2)
        y = (screen_h // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")

        tk.Label(
            popup,
            text="Input Run Time (hours):",
            font=("Arial", 11, "bold"),
            bg="white",
            fg="#1e293b"
        ).pack(pady=10)

        runtime_var = tk.StringVar()
        runtime_entry = tk.Entry(
            popup,
            textvariable=runtime_var,
            font=("Arial", 11),
            width=10,
            justify="center"
        )
        runtime_entry.pack(pady=5)

        btn_frame = tk.Frame(popup, bg="white")
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="Confirm",
            font=("Arial", 10, "bold"),
            bg="#4ade80",
            fg="white",
            padx=15,
            pady=5,
            command=lambda: self.confirm_fine_tune(runtime_var.get(), popup)
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Cancel",
            font=("Arial", 10, "bold"),
            bg="#f87171",
            fg="white",
            padx=15,
            pady=5,
            command=popup.destroy
        ).pack(side=tk.LEFT, padx=5)

    def confirm_fine_tune(self, runtime_value, popup):
        try:
            hours = float(runtime_value)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")
            return

        self.tune_hours = max(hours, 0)
        popup.destroy()

        if hours == 0:
            messagebox.showinfo("Fine Tune Disabled", "Fine-tuning set to 0 hours.\nNormal protection will run.")
        else:
            messagebox.showinfo("Fine Tuning Enabled",
                                f"Fine-tuning set to {hours} hours.\nClick Apply to begin optimization.")

    def estimate_time_per_run(self, base_params):
        time_per_step = 115.4 / 190
        return base_params['steps'] * time_per_step

    def update_fine_tune_progress(self, runs_used, max_runs):
        if max_runs <= 0:
            return
        progress = (runs_used / max_runs) * 100
        self.ft_progress_var.set(progress)
        self.ft_progress_label.config(
            text=f"Fine-tuning… {runs_used}/{max_runs} model runs ({progress:.1f}%)"
        )
        self.root.update_idletasks()

    def toggle_advanced(self):
        if self.show_advanced.get():
            self.advanced_frame.pack(fill=tk.X, padx=20, pady=10)
            self.setup_advanced_params()
        else:
            self.advanced_frame.pack_forget()

    def setup_advanced_params(self):
        for widget in self.advanced_frame.winfo_children():
            widget.destroy()

        params = self.calculate_params(self.slider.get())

        grid_frame = tk.Frame(self.advanced_frame, bg='#f8fafc')
        grid_frame.pack(fill=tk.X, pady=5)

        param_configs = [
            ("Steps:", params['steps']),
            ("Step Size (/255):", params['stepSize']),
            ("L∞ Budget (/255):", params['linfBudget']),
            ("LPIPS Budget:", f"{params['lpipsBudget']:.2f}"),
            ("Style Strength:", f"{params['styleStrength']:.1f}"),
            ("Freq Weight:", f"{params['frequencyWeight']:.2f}"),
            ("DCT Block:", f"{params['dctBlockSize']}×{params['dctBlockSize']}")
        ]

        for i, (label, value) in enumerate(param_configs):
            row = i // 4
            col = i % 4

            param_frame = tk.Frame(grid_frame, bg='#f8fafc')
            param_frame.grid(row=row, column=col, padx=10, pady=5, sticky='w')

            tk.Label(
                param_frame,
                text=label,
                font=("Arial", 8, "bold"),
                bg='#f8fafc',
                fg='#475569'
            ).pack(anchor='w')

            tk.Label(
                param_frame,
                text=str(value),
                font=("Arial", 9),
                bg='white',
                fg='#1e293b',
                relief=tk.SUNKEN,
                padx=5,
                pady=2
            ).pack(anchor='w')

    def get_protection_level(self, shift):
        if shift < 17.5:
            return ("MINIMAL", "#f1f5f9", "#64748b")
        elif shift < 21.0:
            return ("WEAK", "#dbeafe", "#2563eb")
        elif shift < 26.1:
            return ("MODERATE", "#dcfce7", "#16a34a")
        elif shift < 37.8:
            return ("MODERATE-STRONG", "#fef3c7", "#ca8a04")
        elif shift < 45.7:
            return ("STRONG", "#fed7aa", "#ea580c")
        elif shift < 58.6:
            return ("VERY STRONG", "#fecaca", "#dc2626")
        elif shift < 67.8:
            return ("EXTREME", "#fce7f3", "#9333ea")
        else:
            return ("MAXIMUM", "#7c2d12", "#ffffff")

    def get_expected_lpips(self, shift):
        if shift <= 17.5:
            return 0.05 + (shift - 5) / 12.5 * 0.030
        elif shift <= 21.0:
            return 0.080 + (shift - 17.5) / 3.5 * 0.023
        elif shift <= 26.1:
            return 0.103 + (shift - 21.0) / 5.1 * 0.047
        elif shift <= 37.8:
            return 0.150 + (shift - 26.1) / 11.7 * 0.110
        elif shift <= 45.7:
            return 0.260 + (shift - 37.8) / 7.9 * 0.079
        elif shift <= 58.6:
            return 0.339 + (shift - 45.7) / 12.9 * 0.120
        elif shift <= 67.8:
            return 0.459 + (shift - 58.6) / 9.2 * 0.083
        else:
            return 0.542 + (shift - 67.8) / 12.4 * 0.106

    def get_visual_impact(self, shift):
        if shift < 17.5:
            return "Nearly\nimperceptible"
        elif shift < 21.0:
            return "Imperceptible\nto most"
        elif shift < 26.1:
            return "Subtle\nmodifications"
        elif shift < 37.8:
            return "Noticeable on\nclose inspection"
        elif shift < 45.7:
            return "Clearly visible\nchanges"
        elif shift < 58.6:
            return "Obvious\nmodifications"
        elif shift < 67.8:
            return "Heavy artifacts\nquality loss"
        else:
            return "Severe degradation\nuse with caution"

    def calculate_params(self, target_shift):
        if target_shift <= 17.5:
            t = (target_shift - 5) / 12.5
            return {
                'steps': int(80 + t * 20),
                'stepSize': int(4 + t * 1),
                'linfBudget': int(8 + t * 4),
                'lpipsBudget': 0.06 + t * 0.02,
                'styleStrength': 4.0 + t * 2.0,
                'frequencyWeight': 0.60 + t * 0.05,
                'dctBlockSize': 16
            }
        elif target_shift <= 21.0:
            t = (target_shift - 17.5) / 3.5
            return {
                'steps': int(100 + t * 20),
                'stepSize': int(5 + t * 1),
                'linfBudget': int(12 + t * 3),
                'lpipsBudget': 0.08 + t * 0.02,
                'styleStrength': 6.0 + t * 1.5,
                'frequencyWeight': 0.65 + t * 0.05,
                'dctBlockSize': 16
            }
        elif target_shift <= 26.1:
            t = (target_shift - 21.0) / 5.1
            return {
                'steps': int(120 + t * 30),
                'stepSize': int(6 + t * 1),
                'linfBudget': int(15 + t * 3),
                'lpipsBudget': 0.10 + t * 0.03,
                'styleStrength': 7.5 + t * 1.5,
                'frequencyWeight': 0.70 + t * 0.05,
                'dctBlockSize': 16
            }
        elif target_shift <= 37.8:
            t = (target_shift - 26.1) / 11.7
            return {
                'steps': int(150 + t * 35),
                'stepSize': int(7 + t * 1),
                'linfBudget': int(18 + t * 4),
                'lpipsBudget': 0.13 + t * 0.04,
                'styleStrength': 9.0 + t * 2.0,
                'frequencyWeight': 0.75 + t * 0.05,
                'dctBlockSize': 16
            }
        elif target_shift <= 45.7:
            t = (target_shift - 37.8) / 7.9
            return {
                'steps': int(185 + t * 20),
                'stepSize': int(8 + t * 1),
                'linfBudget': int(22 + t * 3),
                'lpipsBudget': 0.17 + t * 0.03,
                'styleStrength': 11.0 + t * 2.0,
                'frequencyWeight': 0.80 + t * 0.05,
                'dctBlockSize': 16
            }
        elif target_shift <= 58.6:
            t = (target_shift - 45.7) / 12.9
            return {
                'steps': int(205 + t * 30),
                'stepSize': int(9 + t * 1),
                'linfBudget': int(25 + t * 5),
                'lpipsBudget': 0.20 + t * 0.04,
                'styleStrength': 13.0 + t * 2.0,
                'frequencyWeight': 0.85 + t * 0.05,
                'dctBlockSize': 16
            }
        elif target_shift <= 67.8:
            t = (target_shift - 58.6) / 9.2
            return {
                'steps': int(235 + t * 25),
                'stepSize': int(10 + t * 1),
                'linfBudget': int(30 + t * 5),
                'lpipsBudget': 0.24 + t * 0.04,
                'styleStrength': 15.0 + t * 2.0,
                'frequencyWeight': 0.90 + t * 0.05,
                'dctBlockSize': 16
            }
        else:
            t = (target_shift - 67.8) / 12.4
            return {
                'steps': int(260 + t * 40),
                'stepSize': int(11 + t * 2),
                'linfBudget': int(35 + t * 10),
                'lpipsBudget': 0.28 + t * 0.06,
                'styleStrength': 17.0 + t * 3.0,
                'frequencyWeight': 0.95 + t * 0.05,
                'dctBlockSize': 16
            }

    def on_slider_change(self, value):
        shift = int(float(value))
        self.slider_label.config(text=f"Target Style Shift: {shift}%")

        level, bg_color, fg_color = self.get_protection_level(shift)
        self.protection_level_label.config(text=level, bg=bg_color, fg=fg_color)

        lpips = self.get_expected_lpips(shift)
        self.lpips_value_label.config(text=f"{lpips:.3f}")

        impact = self.get_visual_impact(shift)
        self.visual_impact_label.config(text=impact)

        params = self.calculate_params(shift)
        self.steps_value_label.config(text=str(params['steps']))

        if self.show_advanced.get():
            self.setup_advanced_params()

    def upload_image(self, image_type):
        filename = filedialog.askopenfilename(
            title=f"Select {image_type} image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not filename:
            return

        try:
            img = validate_image_file(filename)

            if image_type == 'content':
                self.content_image = img
                self.content_label.config(
                    text=f"✓ {os.path.basename(filename)}\n{img.size[0]}×{img.size[1]}",
                    fg="#16a34a"
                )
                self.content_button.config(text="✓ Change Artwork")

                if self.style_image and self.style_image.size != self.content_image.size:
                    self.style_image = self.style_image.resize(self.content_image.size, Image.LANCZOS)

            else:
                if self.content_image and img.size != self.content_image.size:
                    img = img.resize(self.content_image.size, Image.LANCZOS)

                self.style_image = img
                self.style_label.config(
                    text=f"✓ {os.path.basename(filename)}\n{img.size[0]}×{img.size[1]}",
                    fg="#9333ea"
                )
                self.style_button.config(text="✓ Change Style")

            if self.content_image and self.style_image:
                self.process_button.config(state=tk.NORMAL, bg='#6366f1')

        except Exception as e:
            messagebox.showerror(
                "Invalid Image",
                f"Unable to load image:\n\n{e}"
            )

    def update_progress(self, current, total):
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.progress_label.config(
            text=f"Processing... {current}/{total} steps ({progress:.1f}%)"
        )
        self.root.update_idletasks()

    def run_fine_tuning_pipeline(self):
        self.is_fine_tuning = True
        self.is_processing = True
        self.process_button.config(state=tk.DISABLED, bg='#9ca3af')

        self.progress_bar.pack(fill=tk.X, pady=10)
        self.progress_label.pack()
        self.progress_var.set(0)
        self.progress_label.config(text="Preparing fine-tuning…")

        self.ft_progress_bar.pack(fill=tk.X, pady=10)
        self.ft_progress_label.pack()
        self.ft_progress_var.set(0)
        self.ft_progress_label.config(text="Fine-tuning starting...")

        target_shift = self.slider.get()
        base_params = self.calculate_params(target_shift)
        time_per_run = self.estimate_time_per_run(base_params)
        hours = self.tune_hours

        Thread(target=lambda: self.fine_tune(hours, time_per_run)).start()

    def process_protection(self):
        if not self.content_image:
            messagebox.showwarning("Warning", "Please upload your artwork first!")
            return

        if self.is_processing:
            return

        if self.tune_hours > 0:
            self.run_fine_tuning_pipeline()
            return

        self.is_processing = True
        self.process_button.config(state=tk.DISABLED, bg='#9ca3af')
        self.progress_bar.pack(fill=tk.X, pady=10)
        self.progress_label.pack()

        def run_protection():
            try:
                target_shift = self.slider.get()
                params = self.calculate_params(target_shift)

                content_img = self.content_image.copy()
                max_dim = 512
                if max(content_img.size) > max_dim:
                    ratio = max_dim / max(content_img.size)
                    new_size = (int(content_img.width * ratio), int(content_img.height * ratio))
                    content_img = content_img.resize(new_size, Image.LANCZOS)

                if self.style_image:
                    style_img = self.style_image.resize(content_img.size, Image.LANCZOS)
                else:
                    style_img = content_img.copy()

                start_time = time.time()

                protected_img, lpips_orig, lpips_target = pgd_glaze_dct_adaptive(
                    content_pil=content_img,
                    target_style_pil=style_img,
                    steps=params['steps'],
                    step_size=params['stepSize'] / 255,
                    linf_budget_base=params['linfBudget'] / 255,
                    lpips_budget=params['lpipsBudget'],
                    style_shift_strength=params['styleStrength'],
                    frequency_weight=params['frequencyWeight'],
                    dct_block_size=params['dctBlockSize'],
                    target_style_shift_percent=target_shift,
                    progress_callback=self.update_progress,
                    compression_method=self.compression_var.get()
                )

                processing_time = time.time() - start_time

                orig_to_target = lpips_distance(content_img, style_img)
                style_shift = (1 - lpips_target / orig_to_target) * 100

                orig_arr = np.array(self.content_image).astype(float)
                prot_arr = np.array(protected_img.resize(self.content_image.size, Image.LANCZOS)).astype(float)
                pixel_diff = np.abs(orig_arr - prot_arr)

                try:
                    h, w = pixel_diff.shape[:2]
                    mid_h, mid_w = h // 2, w // 2
                    quadrants = {
                        'Top-Left': pixel_diff[:mid_h, :mid_w].mean(),
                        'Top-Right': pixel_diff[:mid_h, mid_w:].mean(),
                        'Bottom-Left': pixel_diff[mid_h:, :mid_w].mean(),
                        'Bottom-Right': pixel_diff[mid_h:, mid_w:].mean()
                    }
                except:
                    quadrants = {
                        'Top-Left': 0,
                        'Top-Right': 0,
                        'Bottom-Left': 0,
                        'Bottom-Right': 0
                    }

                output_path = os.path.join(output_dir, "protected_artwork.png")
                protected_img.resize(self.content_image.size, Image.LANCZOS).save(output_path)

                self.protected_image = protected_img

                self.root.after(0, lambda: self.show_results(
                    style_shift, lpips_orig, lpips_target,
                    pixel_diff.mean(), quadrants, processing_time, output_path
                ))

            except Exception as e:
                self.root.after(0, lambda e=e: messagebox.showerror("Error", f"Protection failed: {str(e)}"))
            finally:
                self.root.after(0, self.reset_ui)

        Thread(target=run_protection).start()

    def fine_tune(self, hours, time_per_run):
        global_best_result = None
        global_best_ratio = -float("inf")
        global_best_img = None

        try:
            target_shift = self.slider.get()
            base_params = self.calculate_params(target_shift)

            content_img = self.content_image.copy()
            max_dim = 512
            if max(content_img.size) > max_dim:
                ratio = max_dim / max(content_img.size)
                new_size = (int(content_img.width * ratio), int(content_img.height * ratio))
                content_img = content_img.resize(new_size, Image.LANCZOS)

            style_img = (
                self.style_image.resize(content_img.size, Image.LANCZOS)
                if self.style_image else content_img.copy()
            )

            max_runs = int((hours * 3600) / time_per_run)
            if max_runs <= 0:
                self.root.after(0, lambda: messagebox.showerror("Error", "Not enough time for any model run."))
                return

            runs_used = 0

            baseline_result = self.run_glaze_iteration(
                content_img=content_img,
                style_img=style_img,
                params=base_params,
                target_shift=target_shift
            )
            global_best_result = baseline_result
            global_best_ratio = baseline_result["ratio"]
            global_best_img = baseline_result["protected_img"]

            runs_used = 1
            self.update_fine_tune_progress(runs_used, max_runs)

            if runs_used >= max_runs:
                output_path = os.path.join(output_dir, "best_protected_artwork.png")
                global_best_img.resize(self.content_image.size, Image.LANCZOS).save(output_path)
                self.protected_image = global_best_img
                self.is_fine_tuning = False

                style_shift = global_best_result["style_shift"]
                lpips_orig = global_best_result["lpips_orig"]
                lpips_target = global_best_result["lpips_target"]
                pixel_diff_mean = global_best_result["pixel_diff_mean"]
                quadrants = global_best_result["quadrants"]
                proc_time = global_best_result["processing_time"]

                self.root.after(
                    50,
                    lambda ss=style_shift, lo=lpips_orig, lt=lpips_target,
                           pd=pixel_diff_mean, q=quadrants, pt=proc_time, op=output_path:
                    self.show_results(ss, lo, lt, pd, q, pt, op)
                )
                return

            tuning_plan = [
                ("linfBudget", 0.02),
                ("lpipsBudget", 0.05),
                ("styleStrength", 0.05),
                ("frequencyWeight", 0.5),
            ]

            num_params = len(tuning_plan)
            runs_per_param = max(1, max_runs // num_params)

            def safe_eval(param_name, val):
                nonlocal runs_used, global_best_result, global_best_ratio, global_best_img
                if runs_used >= max_runs:
                    return None

                res = self.evaluate_param_setting(
                    content_img, style_img, base_params,
                    param_name, val, target_shift
                )

                runs_used += 1
                self.update_fine_tune_progress(runs_used, max_runs)

                if global_best_result is None or res["ratio"] > global_best_ratio:
                    global_best_result = res
                    global_best_ratio = res["ratio"]
                    global_best_img = res["protected_img"]

                return res

            for param_name, delta_init in tuning_plan:
                if runs_used >= max_runs:
                    break

                center = base_params[param_name]
                delta = delta_init
                cache = {}

                def get_result(v):
                    if v in cache:
                        return cache[v]
                    r = safe_eval(param_name, v)
                    if r is not None:
                        cache[v] = r
                    return r

                center_result = get_result(center)
                if center_result is None:
                    break

                left_val = center - delta
                right_val = center + delta

                left_result = get_result(left_val)
                if left_result is None:
                    break

                right_result = get_result(right_val)
                if right_result is None:
                    break

                best = max([center_result, left_result, right_result], key=lambda x: x["ratio"])

                if best is left_result:
                    center = left_val
                elif best is right_result:
                    center = right_val

                base_params[param_name] = center

                remaining_runs = runs_per_param - 3
                extra_iters = max(0, remaining_runs // 2)

                for _ in range(extra_iters):
                    if runs_used >= max_runs:
                        break

                    left_val = center - delta
                    right_val = center + delta

                    left_result = get_result(left_val)
                    if left_result is None:
                        break

                    right_result = get_result(right_val)
                    if right_result is None:
                        break

                    center_result = get_result(center)
                    if center_result is None:
                        break

                    best2 = max([center_result, left_result, right_result], key=lambda x: x["ratio"])

                    if best2 is center_result:
                        delta *= 0.5
                    elif best2 is left_result:
                        center = left_val
                        delta *= 0.5
                    else:
                        center = right_val
                        delta *= 0.5

                    base_params[param_name] = center

                if runs_used >= max_runs:
                    break

            if global_best_result is not None:
                output_path = os.path.join(output_dir, "best_protected_artwork.png")
                global_best_img.resize(self.content_image.size, Image.LANCZOS).save(output_path)
                self.protected_image = global_best_img
                self.is_fine_tuning = False

                style_shift = global_best_result["style_shift"]
                lpips_orig = global_best_result["lpips_orig"]
                lpips_target = global_best_result["lpips_target"]
                pixel_diff_mean = global_best_result["pixel_diff_mean"]
                quadrants = global_best_result["quadrants"]
                proc_time = global_best_result["processing_time"]

                self.root.after(
                    50,
                    lambda ss=style_shift, lo=lpips_orig, lt=lpips_target,
                           pd=pixel_diff_mean, q=quadrants, pt=proc_time, op=output_path:
                    self.show_results(ss, lo, lt, pd, q, pt, op)
                )

        except Exception as e:
            err = str(e)
            self.root.after(0, lambda err=err: messagebox.showerror("Error", f"Protection failed: {err}"))

        finally:
            self.tune_hours = 0
            self.root.after(0, self.reset_ui)

    def run_glaze_iteration(self, content_img, style_img, params, target_shift):
        start_time = time.time()

        protected_img, lpips_orig, lpips_target = pgd_glaze_dct_adaptive(
            content_pil=content_img,
            target_style_pil=style_img,
            steps=params['steps'],
            step_size=params['stepSize'] / 255,
            linf_budget_base=params['linfBudget'] / 255,
            lpips_budget=params['lpipsBudget'],
            style_shift_strength=params['styleStrength'],
            frequency_weight=params['frequencyWeight'],
            dct_block_size=params['dctBlockSize'],
            target_style_shift_percent=target_shift,
            progress_callback=self.update_progress
        )

        processing_time = time.time() - start_time

        orig_to_target = lpips_distance(content_img, style_img)
        style_shift = (1 - lpips_target / orig_to_target) * 100 if orig_to_target != 0 else 0

        orig_arr = np.array(self.content_image).astype(float)
        prot_arr = np.array(
            protected_img.resize(self.content_image.size, Image.LANCZOS)
        ).astype(float)
        pixel_diff = np.abs(orig_arr - prot_arr)

        h, w = pixel_diff.shape[:2]
        mid_h, mid_w = h // 2, w // 2
        quadrants = {
            'Top-Left': pixel_diff[:mid_h, :mid_w].mean(),
            'Top-Right': pixel_diff[:mid_h, mid_w:].mean(),
            'Bottom-Left': pixel_diff[mid_h:, :mid_w].mean(),
            'Bottom-Right': pixel_diff[mid_h:, mid_w:].mean()
        }

        ratio = style_shift / lpips_target if lpips_target != 0 else float("inf")

        return {
            "protected_img": protected_img,
            "lpips_orig": lpips_orig,
            "lpips_target": lpips_target,
            "style_shift": style_shift,
            "pixel_diff_mean": pixel_diff.mean(),
            "quadrants": quadrants,
            "processing_time": processing_time,
            "ratio": ratio
        }

    def evaluate_param_setting(self, content_img, style_img, base_params, param_name, param_value, target_shift):
        params = base_params.copy()
        params[param_name] = param_value

        results = self.run_glaze_iteration(
            content_img=content_img,
            style_img=style_img,
            params=params,
            target_shift=target_shift
        )

        return results

    def get_output_directory(self):
        return output_dir

    def show_results(self, style_shift, lpips_orig, lpips_target, mean_pixel_diff, quadrants, proc_time, output_path):
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=10)
        self.results_text.delete(1.0, tk.END)

        results = f"""

PROTECTION METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Style Shift:           {style_shift:.1f}%
  LPIPS (protected):     {lpips_orig:.6f}
  LPIPS (to target):     {lpips_target:.6f}
  Mean Pixel Change:     {mean_pixel_diff:.2f} / 255
  Processing Time:       {proc_time:.1f}s

QUADRANT DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Top-Left:              {quadrants['Top-Left']:.2f} pixels
  Top-Right:             {quadrants['Top-Right']:.2f} pixels
  Bottom-Left:           {quadrants['Bottom-Left']:.2f} pixels
  Bottom-Right:          {quadrants['Bottom-Right']:.2f} pixels

OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Saved to: {output_path}

Your artwork is now protected from AI style mimicry!
"""

        self.results_text.insert(1.0, results)
        self.results_text.tag_config("center", justify='center')

        download_btn = tk.Button(
            self.root,
            text="💾 Open Output Folder",
            command=lambda: os.startfile(output_dir) if os.name == 'nt' else os.system(f'open \"{output_dir}\"'),
            font=("Arial", 11, "bold"),
            bg='#16a34a',
            fg='white',
            relief=tk.RAISED,
            bd=0,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        download_btn.pack(pady=10)

        if not self.is_fine_tuning:
            messagebox.showinfo(
                "Success!",
                f"Protection complete!\n\n"
                f"Style Shift: {style_shift:.1f}%\n"
                f"LPIPS: {lpips_orig:.4f}\n\n"
                f"Saved to: {output_path}"
            )

    def reset_ui(self):
        self.is_processing = False
        self.process_button.config(state=tk.NORMAL, bg='#6366f1')
        self.progress_bar.pack_forget()
        self.progress_label.pack_forget()
        self.progress_var.set(0)
        self.ft_progress_bar.pack_forget()
        self.ft_progress_label.pack_forget()
        self.ft_progress_var.set(0)


if __name__ == "__main__":
    print("AI Mimicry Prevention Mechanism")
    print(f"Device: {LPIPS_DEVICE}")
    print(f"Output directory: {output_dir}")

    root = tk.Tk()
    app = GlazeProtectionUI(root)
    root.mainloop()
