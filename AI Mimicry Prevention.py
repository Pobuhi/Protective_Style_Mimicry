import os
import numpy as np
from PIL import Image
import imagehash
import torch
from torchvision import transforms
import lpips

base_dir = os.path.dirname(os.path.abspath(__file__))

content_dir = os.path.join(base_dir, "content")
target_dir = os.path.join(base_dir, "target style")
output_dir = os.path.join(base_dir, "output")

os.makedirs(content_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

content_path = os.path.join(content_dir, "CatContent.png")
target_style_path = os.path.join(target_dir, "Sunflowers.png")
content_name = os.path.splitext(os.path.basename(content_path))[0]

# LPIPS setup using CPU
LPIPS_DEVICE = torch.device("cpu")
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


def pgd_glaze_style_transfer(
        content_pil: Image.Image,
        target_style_pil: Image.Image,
        steps=200, # Loop iterations
        step_size=8 / 255, # Optimizer
        linf_budget=40 / 255, # Projection constraint
        lpips_budget=0.35, # Affects the penalty calculation
        style_shift_strength=15.0, # Used in the loss function
        use_adam=True,
):

    x0 = pil_to_lpips_tensor(content_pil).detach()
    x_target = pil_to_lpips_tensor(target_style_pil).detach()

    # Calculate baseline distance to target
    with torch.no_grad():
        baseline_target_dist = lpips_model(x0, x_target).mean()

    # Initialize perturbation
    init_scale = linf_budget / 3.0
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

        # Distance from original (must stay below lpips_budget)
        dist_from_orig = lpips_model(x_adv, x0).mean()

        # Distance from target style
        dist_to_target = lpips_model(x_adv, x_target).mean()

        # How much we've shifted toward target (positive = moved closer towards target)
        style_shift_amount = baseline_target_dist - dist_to_target

        # Hinge loss: only penalize if we exceed imperceptibility budget
        # Imperceptibility means how invisible/unnoticeable the changes are to the human eye. The larger the value
        # the more the changes are clearly visible. Therefore, a smaller imperceptibility means that the changes are invisible/barely noticeable
        imperceptibility_penalty = torch.relu(dist_from_orig - lpips_budget)

        # Loss: MAXIMIZE style shift aggressively
        # With relaxed imperceptibility, we can push much harder
        loss = -style_shift_strength * style_shift_amount + 5.0 * imperceptibility_penalty       # the int(5.0) represents the penalty weight, so it represents how much to penalize exceeding the budget

        loss.backward()
        optim.step()

        if scheduler is not None:
            scheduler.step()

        # Project back into Linf ball
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -linf_budget, linf_budget)
            delta.data = torch.clamp(x0 + delta.data, -1, 1) - x0

        if i % 20 == 0 or i == steps - 1:
            shift_pct = (style_shift_amount / baseline_target_dist * 100).item()
            print(
                f"[GLAZE] step {i:3d} | "
                f"imperceptible={dist_from_orig:.4f} | "
                f"to_target={dist_to_target:.4f} | "
                f"shift={shift_pct:+.1f}% | "
                f"penalty={imperceptibility_penalty:.4f} | "
                f"loss={loss:.4f}"
            )

    # Final projection
    with torch.no_grad():
        delta.data = torch.clamp(delta.data, -linf_budget, linf_budget)
        delta.data = torch.clamp(x0 + delta.data, -1, 1) - x0
        x_final = (x0 + delta.data).clamp(-1, 1)

    x_final_pil = lpips_tensor_to_pil(x_final)

    # Calculate final metrics
    final_dist_orig = float(lpips_model(x_final, x0).squeeze().cpu().item())
    final_dist_target = float(lpips_model(x_final, x_target).squeeze().cpu().item())

    return x_final_pil, final_dist_orig, final_dist_target


if __name__ == "__main__":
    # Load original image
    original_pil = Image.open(content_path).convert("RGB")

    target_style_pil = Image.open(target_style_path).convert("RGB")

    # Resize target to match content dimensions
    target_style_pil = target_style_pil.resize(original_pil.size, Image.LANCZOS)

    print("\n Initial Measurements")
    orig_to_target_dist = lpips_distance(original_pil, target_style_pil)
    print(f"LPIPS(original, original): {lpips_distance(original_pil, original_pil):.6f}")
    print(f"LPIPS(original, target_style): {orig_to_target_dist:.6f}")
    print(f"Baseline: How different the styles are")

    # Glaze configuration
    steps = 350
    step_size = 12 / 255  # MUCH larger steps
    linf_budget = 48 / 255  # 4x standard budget
    lpips_budget = 0.45  # Allow significant changes
    style_shift_strength = 30.0  # Very aggressive push toward target

    print(f"\n Glaze Configuration")
    print(f"Steps: {steps}")
    print(f"Step size: {step_size:.6f} (AGGRESSIVE)")
    print(f"L∞ budget: {linf_budget:.6f} ({linf_budget * 255:.1f}/255)")
    print(f"LPIPS budget (imperceptibility): {lpips_budget}")
    print(f"Style shift strength: {style_shift_strength}")

    print(f"\n Running Glaze Protection")
    x_adv_pil, final_orig_dist, final_target_dist = pgd_glaze_style_transfer(
        content_pil=original_pil,
        target_style_pil=target_style_pil,
        steps=steps,
        step_size=step_size,
        linf_budget=linf_budget,
        lpips_budget=lpips_budget,
        style_shift_strength=style_shift_strength,
        use_adam=True,
    )

    # Save cloaked image
    adv_out = os.path.join(output_dir, f"{content_name}_glaze_protected.jpg")
    x_adv_pil.save(adv_out, quality = 100,subsampling=0)
    print(f"\n Saved protected image to: {adv_out}")

    # Calculate effectiveness
    style_shift_percentage = (1 - final_target_dist / orig_to_target_dist) * 100


    print("PROTECTION RESULTS")
    print("\n Perceptual Metrics")
    print(f"LPIPS(protected, original): {final_orig_dist:.6f}")
    print(f"\n LPIPS(protected, target_style): {final_target_dist:.6f}")
    print(f"Style shift: {style_shift_percentage:.1f}% toward target")

    # Perceptual hashes
    print("\n Perceptual Hash Distances")
    hash_funcs = {
        "aHash": imagehash.average_hash,
        "pHash": imagehash.phash,
        "dHash": imagehash.dhash,
        "wHash": imagehash.whash,
    }
    for name, func in hash_funcs.items():
        h1 = func(original_pil)
        h2 = func(x_adv_pil)
        dist = h1 - h2
        norm = dist / len(h1.hash.ravel())
        print(f"  {name}: {dist} (normalized={norm:.3f})")

    # Pixel differences
    orig_arr = np.array(original_pil).astype(float)
    adv_arr = np.array(x_adv_pil).astype(float)
    print("\n Pixel-Level Changes")
    print(f"Max abs pixel diff : {np.abs(orig_arr - adv_arr).max():.2f} / 255")
    print(f"Mean abs pixel diff: {np.abs(orig_arr - adv_arr).mean():.2f} / 255")