import tensorflow as tf
from IPython.core.display_functions import clear_output
from tensorflow import keras

from tensorflow.keras.preprocessing import image as process_image
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models, losses, backend as K

from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
import functools
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Define folders
content_dir = os.path.join(base_dir, "content")
style_dir = os.path.join(base_dir, "style")
output_dir = os.path.join(base_dir, "output")

os.makedirs(content_dir, exist_ok=True)
os.makedirs(style_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Define file paths inside those folders
content_path = os.path.join(content_dir, "CatContent.jpg")
style_path = os.path.join(style_dir, "StarryNightStyle.jpg")



def load_file(image_path):
    image = Image.open(image_path)
    max_dim = 512
    factor = max_dim / max(image.size)
    image = image.resize(
        (round(image.size[0] * factor), round(image.size[1] * factor)),
        Image.Resampling.LANCZOS,
    )
    im_array = process_image.img_to_array(image)
    im_array = np.expand_dims(im_array, axis=0)
    return im_array


def show_im(img, title=None):
    img = np.squeeze(img, axis=0)
    plt.imshow(np.uint8(img))
    if title is None:
        pass
    else:
        plt.title(title)
    plt.imshow(np.uint8(img))


def img_preprocess(img_path):
    image=load_file(img_path)
    img=tf.keras.applications.vgg19.preprocess_input(image)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3  # Input dimension must be [1, height, width, channel] or [height, width, channel]

    # perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # converting BGR to RGB channel

    x = np.clip(x, 0, 255).astype('uint8')
    return x


content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
number_content=len(content_layers)
number_style =len(style_layers)


def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_output = [vgg.get_layer(layer).output for layer in content_layers]
    style_output = [vgg.get_layer(layer).output for layer in style_layers]
    model_output = style_output + content_output
    return models.Model(vgg.input, model_output)


def get_content_loss(noise,target):
    loss = tf.reduce_mean(tf.square(noise-target))
    return loss

def gram_matrix(tensor):
    channels=int(tensor.shape[-1])
    vector=tf.reshape(tensor,[-1,channels])
    n=tf.shape(vector)[0]
    gram_matrix=tf.matmul(vector,vector,transpose_a=True)
    return gram_matrix/tf.cast(n,tf.float32)


def get_style_loss(noise,target):
    gram_noise=gram_matrix(noise)
    #gram_target=gram_matrix(target)
    loss=tf.reduce_mean(tf.square(target-gram_noise))
    return loss


def get_features(model, content_path, style_path):
    content_img = img_preprocess(content_path)
    style_image = img_preprocess(style_path)

    content_output = model(content_img)
    style_output = model(style_image)

    content_feature = [layer[0] for layer in content_output[number_style:]]
    style_feature = [layer[0] for layer in style_output[:number_style]]
    return content_feature, style_feature


def compute_loss(model, loss_weights, image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights  # style weight and content weight are user given parameters
    # that define what percentage of content and/or style will be preserved in the generated image

    output = model(image)
    content_loss = 0
    style_loss = 0

    noise_style_features = output[:number_style]
    noise_content_feature = output[number_style:]

    #Style Loss
    weight_per_layer = 1.0 / float(number_style)
    for a, b in zip(gram_style_features, noise_style_features):
        style_loss += weight_per_layer * get_style_loss(b[0], a)
    #Content Loss
    weight_per_layer = 1.0 / float(number_content)
    for a, b in zip(noise_content_feature, content_features):
        content_loss += weight_per_layer * get_content_loss(a[0], b)

    style_loss *= style_weight
    content_loss *= content_weight

    total_loss = content_loss + style_loss

    return total_loss, style_loss, content_loss


def compute_grads(dictionary):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**dictionary)

    total_loss = all_loss[0]
    return tape.gradient(total_loss, dictionary['image']), all_loss


def run_style_transfer(content_path, style_path, epochs=50, content_weight=1e3, style_weight=1e-2):
    model = get_model()

    for layer in model.layers:
        layer.trainable = False

    content_feature, style_feature = get_features(model, content_path, style_path)
    style_gram_matrix = [gram_matrix(feature) for feature in style_feature]

    noise = img_preprocess(content_path)
    noise = tf.Variable(noise, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=1e-1)

    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    dictionary = {'model': model,
                  'loss_weights': loss_weights,
                  'image': noise,
                  'gram_style_features': style_gram_matrix,
                  'content_features': content_feature}

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(epochs):
        grad, all_loss = compute_grads(dictionary)
        total_loss, style_loss, content_loss = all_loss
        optimizer.apply_gradients([(grad, noise)])
        clipped = tf.clip_by_value(noise, min_vals, max_vals)
        noise.assign(clipped)

        if total_loss < best_loss:
            best_loss = total_loss
            best_img = deprocess_img(noise.numpy())

        # for visualization

            if i % 5 == 0:
                plot_img = deprocess_img(noise.numpy())
                imgs.append(plot_img)
                clear_output(wait=True)
                display(Image.fromarray(plot_img))
                print(f"Epoch: {i}")
                print(
                    f"Total loss: {total_loss:.4e}, "
                    f"style loss: {style_loss:.4e}, "
                    f"content loss: {content_loss:.4e}"
                )

        clear_output(wait=True)

    return best_img, best_loss, imgs


best, best_loss,image = run_style_transfer(content_path, style_path, epochs=50)
output_path = os.path.join(output_dir, "CatStarryNights.png")
Image.fromarray(best).save(output_path)

