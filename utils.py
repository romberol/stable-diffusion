import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator as GEN


def invert_images(image):
    return 255 - image


def variance_scheduler(T, s=0.008, max_beta=0.999):
    """
    Calculates the alpha, alpha_cumprod, and beta values for the variance scheduler.
    
    Parameters:
    - T: int, number of steps in the scheduler
    - s: float, shift value for the cosine function
    - max_beta: float, maximum value for beta
    
    Returns:
    - alpha: numpy array, alpha values for the scheduler
    - alpha_cumprod: numpy array, cumulative product of alpha values
    - beta: numpy array, beta values for the scheduler
    """
    t = np.arange(T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha = np.clip(f[1:] / f[:-1], 1 - max_beta, 1)
    alpha = np.append(1, alpha).astype(np.float32)
    beta = 1 - alpha
    alpha_cumprod = np.cumprod(alpha)
    return alpha, alpha_cumprod, beta


def form_dataset(data_dir, target_size, batch_size, validation_split, T, seed=0):
    """
    Forms a TensorFlow dataset from image data in a directory with data augmentation
    and noisy batch preparation for diffusion process.
    
    Parameters:
    - data_dir: str, path to the directory containing the image data
    - target_size: tuple, target size of the images in the dataset
        [height, weight, channels]
    - batch_size: int, batch size for the dataset
    - validation_split: float, fraction of the data to use for validation
    - T: int, number of steps in the variance scheduler
    - seed: int, random seed value
    
    Returns:
    - train_dataset: TensorFlow dataset containing the training data
    - val_dataset: TensorFlow dataset containing the validation data
    """

    train_augment = GEN(
        rescale=1. / 255,
        horizontal_flip=True,
        rotation_range=10,
        fill_mode="constant",
        cval=255,
        preprocessing_function=invert_images,
        validation_split=validation_split,
    )

    val_augment = GEN(
        rescale=1. / 255,
        preprocessing_function=invert_images,
        validation_split=validation_split,
    )

    train_data = train_augment.flow_from_directory(
        data_dir,
        target_size=target_size[:2],
        batch_size=batch_size,
        subset='training',
        class_mode=None,
        seed=seed,
    )

    val_data = val_augment.flow_from_directory(
        data_dir,
        target_size=target_size[:2],
        batch_size=batch_size,
        subset='validation',
        class_mode=None,
        seed=seed,
    )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_data,
        output_signature=tf.TensorSpec(shape=(None, *target_size), dtype=tf.float32)
    )
    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_data,
        output_signature=tf.TensorSpec(shape=(None, *target_size), dtype=tf.float32)
    )

    alpha, alpha_cumprod, beta = variance_scheduler(T)

    def prepare_batch(X):
        X = tf.cast(X, tf.float32)
        X_shape = tf.shape(X)
        t = tf.random.uniform([X_shape[0]], minval=1, maxval=T + 1, dtype=np.int32)
        alpha_cm = tf.gather(alpha_cumprod, t)
        alpha_cm = tf.reshape(alpha_cm, [X_shape[0]] + [1] * 3)
        noise = tf.random.normal(X_shape)
        return {
                   "X_noisy": alpha_cm ** 0.5 * X + (1 - alpha_cm) ** 0.5 * noise,
                   "time": t,
               }, noise

    train_dataset = train_dataset.map(prepare_batch).prefetch(1)
    val_dataset = val_dataset.map(prepare_batch).prefetch(1)

    return train_dataset, val_dataset


def generate(model, batch_size=32):
    """
    Generates samples using the diffusion model.
    
    Parameters:
    - model: TensorFlow diffusion model
    - batch_size: int, number of generated samples
    
    Returns:
    - Generated samples
    """
    X = tf.random.normal([batch_size, *input_shape])
    for t in range(T - 1, 0, -1):
        print(f"\rt = {t}", end=" ")
        noise = (tf.random.normal if t > 1 else tf.zeros)(tf.shape(X))
        X_noise = model({"X_noisy": X, "time": tf.constant([t] * batch_size)})
        X = (
                1 / alpha[t] ** 0.5
                * (X - beta[t] / (1 - alpha_cumprod[t]) ** 0.5 * X_noise)
                + (1 - alpha[t]) ** 0.5 * noise
        )
    return X
