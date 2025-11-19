import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import lpips
import cv2

base_dir = os.path.dirname(os.path.abspath(__file__))

content_dir = os.path.join(base_dir, "content")
target_dir = os.path.join(base_dir, "target style")
output_dir = os.path.join(base_dir, "output")

os.makedirs(content_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

content_path = os.path.join(content_dir, "CatContent.jpg")
target_style_path = os.path.join(target_dir, "StarryNights.jpg")
content_name = os.path.splitext(os.path.basename(content_path))[0]

# LPIPS setup using CPU
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


def compute_saliency_map(image_pil: Image.Image, method='combined') -> np.ndarray:
    img_array = np.array(image_pil)

    if method == 'spectral':
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliency_map) = saliency.computeSaliency(img_array)
        saliency_map = (saliency_map * 255).astype("uint8")
        return compute_saliency_map(image_pil, method='combined')

    elif method == 'fine_grained':
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliency_map) = saliency.computeSaliency(img_array)
        saliency_map = (saliency_map * 255).astype("uint8")
        return compute_saliency_map(image_pil, method='combined')

    elif method == 'edge':
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Multi-scale edge detection
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 70, 200)

        # Combine edges
        edges = cv2.addWeighted(edges1, 0.3, edges2, 0.4, 0)
        edges = cv2.addWeighted(edges, 1.0, edges3, 0.3, 0)

        # Dilate to connect nearby edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        saliency_map = edges

    elif method == 'variance':
        # Variance-based (focuses on textured regions)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(float)

        # Compute local variance at multiple scales
        kernel_sizes = [11, 21, 31]
        variance_maps = []

        for k in kernel_sizes:
            mean = cv2.blur(gray, (k, k))
            sqr_mean = cv2.blur(gray ** 2, (k, k))
            variance = np.abs(sqr_mean - mean ** 2)
            variance_maps.append(variance)

        # Combine multi-scale variances
        variance = np.mean(variance_maps, axis=0)
        saliency_map = (variance / (variance.max() + 1e-8) * 255).astype("uint8")

    elif method == 'combined':
        # Combined (edges + variance)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Multi-scale edges
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 70, 200)
        edges = cv2.addWeighted(edges1, 0.3, edges2, 0.4, 0)
        edges = cv2.addWeighted(edges, 1.0, edges3, 0.3, 0)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Variance
        gray_float = gray.astype(float)
        kernel_sizes = [11, 21, 31]
        variance_maps = []
        for k in kernel_sizes:
            mean = cv2.blur(gray_float, (k, k))
            sqr_mean = cv2.blur(gray_float ** 2, (k, k))
            variance = np.abs(sqr_mean - mean ** 2)
            variance_maps.append(variance)
        variance = np.mean(variance_maps, axis=0)
        variance_norm = (variance / (variance.max() + 1e-8) * 255).astype("uint8")

        # Combine (60% edges, 40% variance)
        saliency_map = (0.6 * edges + 0.4 * variance_norm).astype("uint8")


    # Normalize to [0, 1]
    saliency_map = saliency_map.astype(float) / 255.0

    # Smooth the map to avoid sharp boundaries
    saliency_map = cv2.GaussianBlur(saliency_map, (31, 31), 0)

    # Enhance contrast
    saliency_map = np.power(saliency_map, 0.7)  # Gamma correction

    return saliency_map


def visualize_saliency_quadrants(image_pil: Image.Image, saliency_map: np.ndarray):
    import matplotlib.pyplot as plt

    h, w = saliency_map.shape
    mid_h, mid_w = h // 2, w // 2

    # Compute quadrant importance (mean busyness per quadrant)
    quadrants = {
        'Top-Left': saliency_map[:mid_h, :mid_w].mean(),
        'Top-Right': saliency_map[:mid_h, mid_w:].mean(),
        'Bottom-Left': saliency_map[mid_h:, :mid_w].mean(),
        'Bottom-Right': saliency_map[mid_h:, mid_w:].mean()
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Saliency map
    im = axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title("Saliency Map (Busy = Bright)")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Quadrant overlay
    axes[2].imshow(image_pil)
    axes[2].axhline(mid_h, color='cyan', linewidth=2, alpha=0.7)
    axes[2].axvline(mid_w, color='cyan', linewidth=2, alpha=0.7)

    # Annotate quadrants
    for (name, importance), (y, x) in zip(
        quadrants.items(),
        [
            (mid_h // 2, mid_w // 2),
            (mid_h // 2, mid_w + mid_w // 2),
            (mid_h + mid_h // 2, mid_w // 2),
            (mid_h + mid_h // 2, mid_w + mid_w // 2),
        ],
    ):
        axes[2].text(
            x,
            y,
            f"{name}\n{importance:.3f}",
            color='yellow',
            fontsize=10,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
        )

    axes[2].set_title("Quadrant Busy-ness")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "saliency_analysis.jpg"), dpi=150, bbox_inches='tight')
    return quadrants


def pgd_glaze_saliency_adaptive(
        content_pil: Image.Image,
        target_style_pil: Image.Image,
        steps=200,
        step_size=10/255,
        linf_budget_base=50/255,
        lpips_budget=0.50,
        style_shift_strength=20.0,
        saliency_method='combined',
        saliency_min=0.1,   # ~0.3 * epsilon
        saliency_max=0.5,   # busy pixels up to full epsilon
        use_adam=True,
):

    saliency_map = compute_saliency_map(content_pil, method=saliency_method)
    # Visualize and analyze quadrants
    quadrants = visualize_saliency_quadrants(content_pil, saliency_map)


    h, w = saliency_map.shape
    mid_h, mid_w = h // 2, w // 2
    q_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    q_vals = np.array([quadrants[name] for name in q_names], dtype=np.float32)

    # Normalize quadrant scores to [0, 1]
    q_min = q_vals.min()
    q_max = q_vals.max()
    if q_max > q_min:
        q_norm = (q_vals - q_min) / (q_max - q_min)
    else:
        # If all quadrants are equally busy, treat them as mid-level
        q_norm = np.ones_like(q_vals) * 0.5

    # Map quadrants to multipliers [q_mult_min, q_mult_max]
    # Higher busy-ness → closer to q_mult_max
    q_mult_min = 0.7 # calm quadrant: 0.7x of per-pixel budget
    q_mult_max = 1.0   # busiest quadrant: 1.0x (no extra beyond per-pixel)
    q_mults = q_mult_min + (q_mult_max - q_mult_min) * q_norm

    # Build a per-pixel quadrant weight map
    quadrant_weight_map = np.ones_like(saliency_map, dtype=np.float32)

    # Top-Left
    quadrant_weight_map[:mid_h, :mid_w] *= q_mults[0]
    # Top-Right
    quadrant_weight_map[:mid_h, mid_w:] *= q_mults[1]
    # Bottom-Left
    quadrant_weight_map[mid_h:, :mid_w] *= q_mults[2]
    # Bottom-Right
    quadrant_weight_map[mid_h:, mid_w:] *= q_mults[3]

    print("\nQuadrant multipliers (applied on top of per-pixel busyness):")
    for name, m in zip(q_names, q_mults):
        print(f"  {name}: {m:.3f}x")

    # Convert saliency + quadrant weight to tensors
    saliency_tensor = torch.from_numpy(saliency_map).float().unsqueeze(0).unsqueeze(0)
    quadrant_weight_tensor = torch.from_numpy(quadrant_weight_map).float().unsqueeze(0).unsqueeze(0)

    # Per-pixel saliency scaling: less busy → saliency_min, more busy → saliency_max
    saliency_tensor = saliency_min + (saliency_max - saliency_min) * saliency_tensor

    # Apply quadrant weights on top
    combined_saliency = saliency_tensor * quadrant_weight_tensor

    # Keep combined saliency within [saliency_min, saliency_max]
    combined_saliency = torch.clamp(combined_saliency, saliency_min, saliency_max)

    x0 = pil_to_lpips_tensor(content_pil).detach()
    x_target = pil_to_lpips_tensor(target_style_pil).detach()

    # Get spatial dimensions from x0
    _, _, h, w = x0.shape

    # Resize combined saliency to match x0 dimensions and move to device
    saliency_tensor_resized = torch.nn.functional.interpolate(
        combined_saliency,
        size=(h, w),
        mode='bilinear',
        align_corners=False
    ).to(LPIPS_DEVICE)

    with torch.no_grad():
        baseline_target_dist = lpips_model(x0, x_target).mean()

    # Initialize perturbation
    init_scale = linf_budget_base / 3.0
    delta = torch.empty_like(x0).uniform_(-init_scale, init_scale)
    delta.requires_grad_()

    if use_adam:
        optim = torch.optim.Adam([delta], lr=step_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps, eta_min=step_size * 0.1)
    else:
        optim = torch.optim.SGD([delta], lr=step_size, momentum=0.9)
        scheduler = None

    for i in range(steps):
        optim.zero_grad()
        x_adv = (x0 + delta).clamp(-1, 1)

        dist_from_orig = lpips_model(x_adv, x0).mean()
        dist_to_target = lpips_model(x_adv, x_target).mean()

        style_shift_amount = baseline_target_dist - dist_to_target
        imperceptibility_penalty = torch.relu(dist_from_orig - lpips_budget)

        # Extra regularizer: discourage perturbations in smoother / less busy regions
        flat_weight = 1.0 / (saliency_tensor_resized + 1e-3)  # low busyness → big weight
        delta_reg = (flat_weight * delta.abs()).mean()

        # Softer loss: less push toward style, stronger imperceptibility + spatial regularizer
        loss = (
            -style_shift_strength * style_shift_amount
            + 10.0 * imperceptibility_penalty
            + 0.5 * delta_reg
        )

        loss.backward()
        optim.step()

        if scheduler is not None:
            scheduler.step()

        # SALIENCY-ADAPTIVE PROJECTION (per-pixel + per-quadrant budget)
        with torch.no_grad():
            adaptive_budget = linf_budget_base * saliency_tensor_resized

            # Clamp delta per-pixel based on local combined saliency
            delta.data = torch.clamp(delta.data, -adaptive_budget, adaptive_budget)
            delta.data = torch.clamp(x0 + delta.data, -1, 1) - x0

        if i % 20 == 0 or i == steps - 1:
            shift_pct = (style_shift_amount / baseline_target_dist * 100).item()

            delta_magnitude = delta.abs().mean().item() * 255
            max_delta = delta.abs().max().item() * 255

            print(
                f"[SALIENCY-GLAZE] step {i:3d} | "
                f"imperceptible={dist_from_orig:.4f} | "
                f"to_target={dist_to_target:.4f} | "
                f"shift={shift_pct:+.1f}% | "
                f"δ_mean={delta_magnitude:.1f} | "
                f"δ_max={max_delta:.1f} | "
                f"penalty={imperceptibility_penalty:.4f} | "
                f"δ_reg={delta_reg:.4f} | "
                f"loss={loss:.4f}"
            )

            if i % 40 == 0 and LPIPS_DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    # Final projection
    with torch.no_grad():
        adaptive_budget = linf_budget_base * saliency_tensor_resized
        delta.data = torch.clamp(delta.data, -adaptive_budget, adaptive_budget)
        delta.data = torch.clamp(x0 + delta.data, -1, 1) - x0
        x_final = (x0 + delta.data).clamp(-1, 1)

    x_final_pil = lpips_tensor_to_pil(x_final)

    # Final metrics
    final_dist_orig = float(lpips_model(x_final, x0).squeeze().cpu().item())
    final_dist_target = float(lpips_model(x_final, x_target).squeeze().cpu().item())

    # Visualize perturbation distribution
    visualize_perturbation_map(delta, saliency_map)

    if LPIPS_DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return x_final_pil, final_dist_orig, final_dist_target, saliency_map


def visualize_perturbation_map(delta: torch.Tensor, saliency_map: np.ndarray):
    import matplotlib.pyplot as plt

    # Get perturbation magnitude - detach from computation graph
    delta_magnitude = delta.detach().abs().mean(dim=1).squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Saliency map (busy-ness)
    im1 = axes[0].imshow(saliency_map, cmap='hot')
    axes[0].set_title("Saliency (Busy-ness)")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Perturbation magnitude
    im2 = axes[1].imshow(delta_magnitude, cmap='viridis')
    axes[1].set_title("Perturbation Magnitude")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(saliency_map, cmap='gray', alpha=0.5)
    axes[2].imshow(delta_magnitude, cmap='hot', alpha=0.5)
    axes[2].set_title("Overlay (Busy-ness × Cloak)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perturbation_map.jpg"), dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    original_pil = Image.open(content_path).convert("RGB")
    target_style_pil = Image.open(target_style_path).convert("RGB")

    max_dimension = 512
    original_size = original_pil.size

    if max(original_pil.size) > max_dimension:
        ratio = max_dimension / max(original_pil.size)
        new_size = (int(original_pil.width * ratio), int(original_pil.height * ratio))
        original_pil = original_pil.resize(new_size, Image.LANCZOS)

    target_style_pil = target_style_pil.resize(original_pil.size, Image.LANCZOS)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n Initial Measurements")
    orig_to_target_dist = lpips_distance(original_pil, target_style_pil)
    print(f"LPIPS(original, original): {lpips_distance(original_pil, original_pil):.6f}")
    print(f"LPIPS(original, target_style): {orig_to_target_dist:.6f}")

    steps = 200
    step_size = 10/255
    linf_budget_base = 50/255
    lpips_budget = 0.50
    style_shift_strength = 20
    saliency_method = 'combined'
    saliency_min = 0.1
    saliency_max = 0.5

    print(f"\n Saliency-Adaptive Configuration")
    print(f"Steps: {steps}")
    print(f"Step size: {step_size:.6f}")
    print(f"Base L∞ budget: {linf_budget_base:.6f} ({linf_budget_base * 255:.1f}/255)")
    print(f"LPIPS budget: {lpips_budget}")
    print(f"Style shift strength: {style_shift_strength}")
    print(f"Saliency method: {saliency_method}")
    print(f"Per-pixel saliency multipliers: {saliency_min:.2f}x to {saliency_max:.2f}x of base L∞")

    x_adv_pil, final_orig_dist, final_target_dist, saliency_map = pgd_glaze_saliency_adaptive(
        content_pil=original_pil,
        target_style_pil=target_style_pil,
        steps=steps,
        step_size=step_size,
        linf_budget_base=linf_budget_base,
        lpips_budget=lpips_budget,
        style_shift_strength=style_shift_strength,
        saliency_method=saliency_method,
        saliency_min=saliency_min,
        saliency_max=saliency_max,
        use_adam=True,
    )

    adv_out = os.path.join(output_dir, f"{content_name}_glaze_saliency_quadrants.jpg")

    if max(original_size) > max_dimension:
        x_adv_pil_full = x_adv_pil.resize(original_size, Image.LANCZOS)
    else:
        x_adv_pil_full = x_adv_pil

    x_adv_pil_full.save(adv_out)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory after: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    # Results
    style_shift_percentage = (1 - final_target_dist / orig_to_target_dist) * 100


    print("PROTECTION RESULTS")

    print(f"\nLPIPS(protected, original): {final_orig_dist:.6f}")
    print(f"LPIPS(protected, target): {final_target_dist:.6f}")
    print(f"Style shift: {style_shift_percentage:.1f}% toward target")


    # Pixel-level analysis at original resolution
    original_full = Image.open(content_path).convert("RGB")
    orig_arr = np.array(original_full).astype(float)
    adv_arr = np.array(x_adv_pil_full).astype(float)
    pixel_diff = np.abs(orig_arr - adv_arr)

    print(f"\n Pixel-Level Changes")
    print(f"Max pixel diff: {pixel_diff.max():.2f} / 255")
    print(f"Mean pixel diff: {pixel_diff.mean():.2f} / 255")

    # Quadrant pixel change distribution
    h, w = pixel_diff.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    quadrant_diffs = {
        'Top-Left': pixel_diff[:mid_h, :mid_w].mean(),
        'Top-Right': pixel_diff[:mid_h, mid_w:].mean(),
        'Bottom-Left': pixel_diff[mid_h:, :mid_w].mean(),
        'Bottom-Right': pixel_diff[mid_h:, mid_w:].mean()
    }

    print(f"\n Quadrant Protection Distribution (mean pixel change)")
    for name, diff in quadrant_diffs.items():
        print(f"{name}: {diff:.2f} pixels changed")
