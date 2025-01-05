import os
import numpy as np
from PIL import Image  # Для роботи із зображеннями


def otsu_thresholding(image):
    hist, bins = np.histogram(image.ravel(), 256, [0, 256])
    total_pixels = image.size
    current_max, threshold = 0, 0
    sum_total, sum_foreground = np.sum(np.arange(256) * hist), 0
    weight_bg, weight_fg = 0, 0

    for i in range(256):
        weight_bg += hist[i]
        weight_fg = total_pixels - weight_bg
        if weight_bg == 0 or weight_fg == 0:
            continue
        sum_foreground += i * hist[i]
        mean_bg = sum_foreground / weight_bg
        mean_fg = (sum_total - sum_foreground) / weight_fg
        variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance_between > current_max:
            current_max = variance_between
            threshold = i

    binarized_image = np.where(image > threshold, 255, 0).astype(np.uint8)
    return threshold, binarized_image


def niblack_thresholding(image, window_size=15, k=-0.2):
    padded_image = np.pad(image, window_size // 2, mode='edge')  # Розширення
    binarized = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x_min = i
            x_max = x_min + window_size
            y_min = j
            y_max = y_min + window_size
            window = padded_image[x_min:x_max, y_min:y_max]
            mean = np.mean(window)
            std_dev = np.std(window)
            threshold = mean + k * std_dev  # Корекція
            binarized[i, j] = 255 if image[i, j] > threshold else 0

    return binarized


def sauvola_thresholding(image, window_size=15, k=0.5, r=128):
    padded_image = np.pad(image, window_size // 2, mode='edge')
    binarized = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x_min = i
            x_max = x_min + window_size
            y_min = j
            y_max = y_min + window_size
            window = padded_image[x_min:x_max, y_min:y_max]
            mean = np.mean(window)
            std_dev = np.std(window)
            threshold = mean * (1 + k * ((std_dev / r) - 1))
            binarized[i, j] = 255 if image[i, j] > threshold else 0

    return binarized


def christians_thresholding(image, window_size=15, k=0.5, r=128):
    padded_image = np.pad(image, window_size // 2, mode='edge')
    binarized = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x_min = i
            x_max = x_min + window_size
            y_min = j
            y_max = y_min + window_size
            window = padded_image[x_min:x_max, y_min:y_max]
            mean = np.mean(window)
            std_dev = np.std(window)
            threshold = mean * (1 + k * ((std_dev / r) ** 0.5))
            binarized[i, j] = 255 if image[i, j] > threshold else 0

    return binarized


def load_grayscale_image(image_path):
    image = Image.open(image_path).convert('L')  # Відкриття зображення у відтінках сірого
    return np.array(image)


def save_image(image, path):
    result = Image.fromarray(image)
    result.save(path)


# Шлях до робочого столу
desktop_path = os.path.expanduser("~/Desktop/")

# Імена файлів зображень
image_files = ["1p.jpg", "2p.jpg", "1t.jpg", "2t.jpg"]

# Папка для результатів
script_dir = os.path.dirname(os.path.abspath(__file__))  # Поточна директорія скрипта
results_folder = os.path.join(script_dir, "results")  # Шлях до папки results в тій же директорії
os.makedirs(results_folder, exist_ok=True)  # Створення папки, якщо її немає

# Прохід по кожному зображенню
for image_file in image_files:
    image_path = os.path.join(desktop_path, image_file)
    if not os.path.exists(image_path):
        print(f"Зображення {image_file} не знайдено на робочому столі. Пропущено.")
        continue

    grayscale_image = load_grayscale_image(image_path)

    # Застосування методів
    print(f"Обробка зображення: {image_file}")
    otsu_result = otsu_thresholding(grayscale_image)
    niblack_result = niblack_thresholding(grayscale_image)
    sauvola_result = sauvola_thresholding(grayscale_image)
    christians_result = christians_thresholding(grayscale_image)

    # Збереження результатів
    base_name = os.path.splitext(image_file)[0]
    save_image(otsu_result[1], os.path.join(results_folder, f"{base_name}_otsu.jpg"))
    save_image(niblack_result, os.path.join(results_folder, f"{base_name}_niblack.jpg"))
    save_image(sauvola_result, os.path.join(results_folder, f"{base_name}_sauvola.jpg"))
    save_image(christians_result, os.path.join(results_folder, f"{base_name}_christians.jpg"))

    print(f"Обробка для {image_file} завершена. Результати збережено в {results_folder}.")

print("Обробку завершено.")
