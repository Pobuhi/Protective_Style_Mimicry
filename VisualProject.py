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

home_dir = os.path.expanduser("~")
desktop_dir = os.path.join(home_dir, "Desktop")
if not os.path.isdir(desktop_dir):
    desktop_dir = home_dir
# Main app folder on Desktop
base_dir = os.path.join(desktop_dir, "ProtectionStudio")
os.makedirs(base_dir, exist_ok=True)
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# LPIPS setup
LPIPS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='vgg').to(LPIPS_DEVICE).eval()
_to_tensor_01 = transforms.ToTensor()


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
        steps=185,
        step_size=6/ 255,
        linf_budget_base=20/255,
        lpips_budget=0.16,
        style_shift_strength=10.0,
        frequency_weight=0.20,
        dct_block_size=16,
        target_style_shift_percent: float = 40.0,
        progress_callback=None
):
    energy_map = compute_dct_frequency_map(content_pil, block_size=dct_block_size)
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

        self.content_image = None
        self.style_image = None
        self.protected_image = None
        self.is_processing = False

        self.setup_ui()

    def setup_ui(self):
        # Title
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

        # Main container
        main_frame = tk.Frame(self.root, bg='#f8fafc')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Protection Strength Panel
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

        # Slider markers
        markers_frame = tk.Frame(slider_container, bg='white')
        markers_frame.pack(fill=tk.X)

        markers = [
            ("5%\nMinimal", 0),
            ("15%\nWeak", 0.125),
            ("30%\nModerate", 0.312),
            ("50%\nStrong", 0.562),
            ("70%\nVery Strong", 0.812),
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

        # Expected Results
        results_frame = tk.Frame(strength_frame, bg='white')
        results_frame.pack(fill=tk.X, padx=20, pady=10)

        # LPIPS Box
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

        # Visual Impact Box
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

        # Processing Steps Box
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

        # Advanced Parameters (collapsible)
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

        # Image Upload Section
        upload_frame = tk.Frame(main_frame, bg='#f8fafc')
        upload_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Content Image
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

        # Style Image
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

        # Process Button
        process_frame = tk.Frame(main_frame, bg='#f8fafc')
        process_frame.pack(fill=tk.X, pady=10)

        self.process_button = tk.Button(
            process_frame,
            text="🛡️ Apply DCT Protection",
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

        # Progress Bar
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

        # Results Display
        self.results_text = tk.Text(
            main_frame,
            height=8,
            font=("Courier", 9),
            bg='white',
            relief=tk.SUNKEN,
            bd=2
        )

        # Update initial values
        self.on_slider_change(40)

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

    def calculate_params(self, target_shift):
        if target_shift <= 15:
            return {
                'steps': 100,
                'stepSize': 5,
                'linfBudget': 12,
                'lpipsBudget': 0.08,
                'styleStrength': 6.0,
                'frequencyWeight': 0.30,
                'dctBlockSize': 16
            }
        elif target_shift <= 30:
            t = (target_shift - 15) / 15
            return {
                'steps': int(100 + t * 200),
                'stepSize': int(5 + t * 5),
                'linfBudget': int(12 + t * 28),
                'lpipsBudget': 0.08 + t * 0.17,
                'styleStrength': 6.0 + t * 14.0,
                'frequencyWeight': 0.40,
                'dctBlockSize': 16
            }
        elif target_shift <= 50:
            t = (target_shift - 30) / 20
            return {
                'steps': int(100 + t * 200),
                'stepSize': int(10 + t * 5),
                'linfBudget': int(40 + t * 30),
                'lpipsBudget': 0.25 + t * 0.15,
                'styleStrength': 20.0 + t * 20.0,
                'frequencyWeight': 0.35,
                'dctBlockSize': 16
            }
        elif target_shift <= 70:
            t = (target_shift - 50) / 20
            return {
                'steps': int(500 + t * 100),
                'stepSize': int(15 + t * 3),
                'linfBudget': int(70 + t * 30),
                'lpipsBudget': 0.40 + t * 0.20,
                'styleStrength': 40.0 + t * 40.0,
                'frequencyWeight': 0.40,
                'dctBlockSize': 16
            }
        else:
            t = (target_shift - 70) / 15
            return {
                'steps': int(600 + t * 200),
                'stepSize': int(18 + t * 7),
                'linfBudget': int(100 + t * 50),
                'lpipsBudget': 0.60 + t * 0.40,
                'styleStrength': 80.0 + t * 40.0,
                'frequencyWeight': 0.40,
                'dctBlockSize': 16
            }

    def get_protection_level(self, shift):
        if shift < 15:
            return ("MINIMAL", "#f1f5f9", "#64748b")
        elif shift < 30:
            return ("MODERATE", "#dbeafe", "#2563eb")
        elif shift < 50:
            return ("STRONG", "#dcfce7", "#16a34a")
        elif shift < 70:
            return ("VERY STRONG", "#fed7aa", "#ea580c")
        else:
            return ("MAXIMUM", "#fecaca", "#dc2626")

    def get_expected_lpips(self, shift):
        if shift <= 15:
            return 0.05 + (shift / 15) * 0.05
        elif shift <= 30:
            return 0.10 + ((shift - 15) / 15) * 0.15
        elif shift <= 50:
            return 0.25 + ((shift - 30) / 20) * 0.15
        elif shift <= 70:
            return 0.40 + ((shift - 50) / 20) * 0.20
        else:
            return 0.60 + ((shift - 70) / 15) * 0.20

    def get_visual_impact(self, shift):
        if shift < 15:
            return "Nearly\nimperceptible"
        elif shift < 30:
            return "Noticeable on\nclose inspection"
        elif shift < 50:
            return "Clearly visible\nmodifications"
        elif shift < 70:
            return "Obvious changes\nquality loss"
        else:
            return "Severe degradation\nuse with caution"

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

        if filename:
            try:
                img = Image.open(filename).convert("RGB")

                if image_type == 'content':
                    self.content_image = img
                    self.content_label.config(
                        text=f"✓ {os.path.basename(filename)}\n{img.size[0]}×{img.size[1]}",
                        fg='#16a34a'
                    )
                    self.content_button.config(text="✓ Change Artwork")
                else:
                    self.style_image = img
                    self.style_label.config(
                        text=f"✓ {os.path.basename(filename)}\n{img.size[0]}×{img.size[1]}",
                        fg='#9333ea'
                    )
                    self.style_button.config(text="✓ Change Style")

                if self.content_image:
                    self.process_button.config(state=tk.NORMAL, bg='#6366f1')

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def update_progress(self, current, total):
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.progress_label.config(
            text=f"Processing... {current}/{total} steps ({progress:.1f}%)"
        )
        self.root.update_idletasks()

    def process_protection(self):
        if not self.content_image:
            messagebox.showwarning("Warning", "Please upload your artwork first!")
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.process_button.config(state=tk.DISABLED, bg='#9ca3af')
        self.progress_bar.pack(fill=tk.X, pady=10)
        self.progress_label.pack()

        def run_protection():
            try:
                target_shift = self.slider.get()
                params = self.calculate_params(target_shift)

                # Resize for efficiency
                content_img = self.content_image.copy()
                max_dim = 512
                if max(content_img.size) > max_dim:
                    ratio = max_dim / max(content_img.size)
                    new_size = (int(content_img.width * ratio), int(content_img.height * ratio))
                    content_img = content_img.resize(new_size, Image.LANCZOS)

                # Use style image or duplicate content
                if self.style_image:
                    style_img = self.style_image.resize(content_img.size, Image.LANCZOS)
                else:
                    style_img = content_img.copy()

                # Run protection
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

                # Calculate metrics
                orig_to_target = lpips_distance(content_img, style_img)
                style_shift = (1 - lpips_target / orig_to_target) * 100

                # Calculate pixel differences
                orig_arr = np.array(self.content_image).astype(float)
                prot_arr = np.array(protected_img.resize(self.content_image.size, Image.LANCZOS)).astype(float)
                pixel_diff = np.abs(orig_arr - prot_arr)

                # Quadrant analysis
                h, w = pixel_diff.shape[:2]
                mid_h, mid_w = h // 2, w // 2
                quadrants = {
                    'Top-Left': pixel_diff[:mid_h, :mid_w].mean(),
                    'Top-Right': pixel_diff[:mid_h, mid_w:].mean(),
                    'Bottom-Left': pixel_diff[mid_h:, :mid_w].mean(),
                    'Bottom-Right': pixel_diff[mid_h:, mid_w:].mean()
                }

                # Save protected image
                output_path = os.path.join(output_dir, "protected_artwork.png")
                protected_img.resize(self.content_image.size, Image.LANCZOS).save(output_path)

                self.protected_image = protected_img

                # Display results
                self.root.after(0, lambda: self.show_results(
                    style_shift, lpips_orig, lpips_target,
                    pixel_diff.mean(), quadrants, processing_time, output_path
                ))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Protection failed: {str(e)}"))
            finally:
                self.root.after(0, self.reset_ui)

        thread = Thread(target=run_protection)
        thread.start()

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

        # Download button
        download_btn = tk.Button(
            self.root,
            text="💾 Open Output Folder",
            command=lambda: os.startfile(output_dir) if os.name == 'nt' else os.system(f'open "{output_dir}"'),
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


if __name__ == "__main__":
    print("AI Mimicry Prevention Mechanism")
    print(f"Device: {LPIPS_DEVICE}")
    print(f"Output directory: {output_dir}")

    root = tk.Tk()
    app = GlazeProtectionUI(root)
    root.mainloop()