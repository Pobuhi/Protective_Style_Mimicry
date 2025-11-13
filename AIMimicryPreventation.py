import os
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.core.display_functions import clear_output
from IPython.display import display
from tensorflow.keras.preprocessing import image as process_image
from tensorflow.keras.applications import vgg19
from tensorflow.keras import models
import imagehash
import torch
from torchvision import transforms
import lpips

base_dir = os.path.dirname(os.path.abspath(__file__))

content_dir = os.path.join(base_dir, "content")
style_dir = os.path.join(base_dir, "style")
output_dir = os.path.join(base_dir, "output")

os.makedirs(content_dir, exist_ok=True)
os.makedirs(style_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

content_path = os.path.join(content_dir, "CatContent.jpg")
style_path = os.path.join(style_dir, "StarryNightStyle.jpg")

content_name = os.path.splitext(os.path.basename(content_path))[0]
style_name = os.path.splitext(os.path.basename(style_path))[0]


def load_file(image_path):
    image = Image.open(image_path).convert("RGB")
    max_dim = 512
    factor = max_dim / max(image.size)
    image = image.resize(
        (round(image.size[0] * factor), round(image.size[1] * factor)),
        Image.Resampling.LANCZOS,
    )
    im_array = process_image.img_to_array(image)
    im_array = np.expand_dims(im_array, axis=0)
    return im_array


def img_preprocess(img_path):
    image = load_file(img_path)
    img = tf.keras.applications.vgg19.preprocess_input(image)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    x = x.astype('float32')
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # BGR->RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x


content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
number_content = len(content_layers)
number_style = len(style_layers)


def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_output = [vgg.get_layer(layer).output for layer in content_layers]
    style_output = [vgg.get_layer(layer).output for layer in style_layers]
    model_output = style_output + content_output
    return models.Model(vgg.input, model_output)


def get_content_loss(noise, target):
    return tf.reduce_mean(tf.square(noise - target))


def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    vector = tf.reshape(tensor, [-1, channels])
    n = tf.shape(vector)[0]
    gram = tf.matmul(vector, vector, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(noise, target_gram):
    gram_noise = gram_matrix(noise)
    return tf.reduce_mean(tf.square(target_gram - gram_noise))


def get_features(model, content_path, style_path):
    content_img = img_preprocess(content_path)
    style_image = img_preprocess(style_path)
    content_out = model(content_img)
    style_out = model(style_image)
    content_feat = [layer[0] for layer in content_out[number_style:]]
    style_feat = [layer[0] for layer in style_out[:number_style]]
    return content_feat, style_feat


def compute_loss(model, loss_weights, image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    output = model(image)
    noise_style_features = output[:number_style]
    noise_content_features = output[number_style:]

    style_loss = 0.0
    content_loss = 0.0

    w_style = 1.0 / float(number_style)
    for target_gram, noise_feat in zip(gram_style_features, noise_style_features):
        style_loss += w_style * get_style_loss(noise_feat[0], target_gram)

    w_content = 1.0 / float(number_content)
    for noise_feat, target_feat in zip(noise_content_features, content_features):
        content_loss += w_content * get_content_loss(noise_feat[0], target_feat)

    style_loss *= style_weight
    content_loss *= content_weight
    total_loss = style_loss + content_loss
    return total_loss, style_loss, content_loss


def compute_grads(dictionary):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**dictionary)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, dictionary['image']), all_loss


def run_style_transfer(content_path, style_path, epochs=50, content_weight=1e3, style_weight=1e-2, print_every=5):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    content_feature, style_feature = get_features(model, content_path, style_path)
    style_gram_matrix = [gram_matrix(f) for f in style_feature]

    noise = img_preprocess(content_path)
    noise = tf.Variable(noise, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=1e-1)

    best_loss, best_img = float('inf'), None
    loss_weights = (style_weight, content_weight)
    dictionary = {
        'model': model,
        'loss_weights': loss_weights,
        'image': noise,
        'gram_style_features': style_gram_matrix,
        'content_features': content_feature
    }

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []

    for i in range(epochs):
        grad, (total_loss, style_loss, content_loss) = compute_grads(dictionary)
        optimizer.apply_gradients([(grad, noise)])
        clipped = tf.clip_by_value(noise, min_vals, max_vals)
        noise.assign(clipped)

        if float(total_loss.numpy()) < best_loss:
            best_loss = float(total_loss.numpy())
            best_img = deprocess_img(noise.numpy())

        if (i % print_every) == 0 or i == epochs - 1:
            plot_img = deprocess_img(noise.numpy())
            imgs.append(plot_img)
            clear_output(wait=True)
            display(Image.fromarray(plot_img))
            print(f"Epoch: {i}")
            print(
                f"Total: {float(total_loss.numpy()):.4e}, Style Loss: {float(style_loss.numpy()):.4e}, Content: Loss {float(content_loss.numpy()):.4e}")

    return best_img, best_loss, imgs


# LPIPS utilities + PGD solver for: min_{delta} LPIPS(x+delta, target) s.t. ||delta||_inf <= p
USE_CUDA = torch.cuda.is_available()
LPIPS_DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
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


def pgd_minimize_lpips_hinge(
        content_pil: Image.Image,
        target_pil: Image.Image,
        p_hinge=0.05,  # LPIPS budget for perturbation magnitude LPIPS(x+δ, x)
        steps=40,
        step_size=2 / 255,
        alpha=1.0,  # weight on hinge penalty
        linf_budget=8 / 255  # optional pixel-space L_inf cap in [-1,1]
):
    if content_pil.size != target_pil.size:
        target_pil = target_pil.resize(content_pil.size, Image.Resampling.LANCZOS)

    x0 = pil_to_lpips_tensor(content_pil).detach()
    tgt = pil_to_lpips_tensor(target_pil).detach()

    delta = torch.zeros_like(x0, requires_grad=True)
    optim = torch.optim.SGD([delta], lr=step_size)

    for _ in range(steps):
        optim.zero_grad()
        x_adv = (x0 + delta).clamp(-1, 1)
        loss_target = lpips_model(x_adv, tgt).mean()
        pert_mag = lpips_model(x_adv, x0).mean()
        hinge = torch.relu(pert_mag - p_hinge)

        loss = loss_target + alpha * hinge
        loss.backward()
        optim.step()

        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -linf_budget, linf_budget)
            delta.data = torch.clamp(x0 + delta.data, -1, 1) - x0

    with torch.no_grad():
        delta.data = torch.clamp(delta.data, -linf_budget, linf_budget)
        delta.data = torch.clamp(x0 + delta.data, -1, 1) - x0
        x_final = (x0 + delta.data).clamp(-1, 1)

    x_final_pil = lpips_tensor_to_pil(x_final)
    final_to_target = float(lpips_model(x_final, tgt).squeeze().cpu().item())
    final_pert_mag = float(lpips_model(x_final, x0).squeeze().cpu().item())
    return x_final_pil, final_to_target, final_pert_mag


best, best_loss, _ = run_style_transfer(content_path, style_path, epochs=50)
output_filename = f"{content_name}_x_{style_name}.png"
output_path = os.path.join(output_dir, output_filename)
Image.fromarray(best).save(output_path)
print(f"Saved stylized image to: {output_path}")

original_pil = Image.open(content_path).convert("RGB")
stylized_pil = Image.fromarray(best).convert("RGB")

if original_pil.size != stylized_pil.size:
    original_pil_resized = original_pil.resize(stylized_pil.size, Image.Resampling.LANCZOS)
else:
    original_pil_resized = original_pil

hash_funcs = {
    "aHash": imagehash.average_hash,
    "pHash": imagehash.phash,
    "dHash": imagehash.dhash,
    "wHash": imagehash.whash,
}

results_lines = []
print("Perceptual Hash Distances (Hamming)")
for name, func in hash_funcs.items():
    h1 = func(original_pil_resized)
    h2 = func(stylized_pil)
    dist = h1 - h2
    norm = dist / len(h1.hash.ravel())
    line = f"{name}: {h1} vs {h2} | distance={dist} (normalized={norm:.3f})"
    print(line)
    results_lines.append(line)

lpips_cs = lpips_distance(original_pil_resized, stylized_pil)
lp_line = f"LPIPS(content vs stylized): {lpips_cs:.6f}"
print(lp_line)
results_lines.append(lp_line)


x_adv_pil, lpips_to_target, lpips_pert_mag = pgd_minimize_lpips_hinge(
    content_pil=original_pil_resized,
    target_pil=stylized_pil,
    p_hinge=0.05,
    steps=40,
    step_size=2/255,
    alpha=1.0,
    linf_budget=8/255
)

adv_out = os.path.join(output_dir, f"{content_name}_adv_to_{style_name}_hinge.png")
x_adv_pil.save(adv_out)
results_lines.append(f"LPIPS(x_adv, target): {lpips_to_target:.6f}")
results_lines.append(f"LPIPS(x_adv, x_content) [perturb mag]: {lpips_pert_mag:.6f} (p_hinge=0.05, alpha=1.0)")
results_lines.append(f"L_inf cap: 8/255, steps=40, step_size=2/255")

report_path = os.path.join(output_dir, f"{content_name}_vs_{style_name}_perceptual_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))
print(f"Saved perceptual report (hashes + LPIPS + hinge): {report_path}")
