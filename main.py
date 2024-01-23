import fitz
from PIL import Image
import pytesseract
import json
import io
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_info(image):
    image_array = np.asarray(bytearray(image), dtype=np.uint8)

    image_decode = cv2.rotate(cv2.imdecode(image_array, cv2.IMREAD_COLOR), cv2.ROTATE_90_CLOCKWISE)

    # Convert to grayscale
    image_gray = cv2.cvtColor(image_decode, cv2.COLOR_BGR2GRAY)

    # Opening
    image_opened = opening(image_gray)

    # Threshold
    image_thresholded = cv2.convertScaleAbs(thresholding(image_opened), alpha=1.95, beta=0)

    plt.imshow(image_thresholded, cmap='gray')
    plt.title('Processed Image')
    plt.show()

    text = pytesseract.image_to_string(image_thresholded, lang = 'eng')

    return text

    # Extract each parameter
    cross_street = extract_cs(text)

    renew_size = extract_rs(text)

    renew_date = extract_rd(text)

    house_number = extract_hn(text)

    data = {'Nearest Cross Street': [cross_street],
                'Renew - Size': [renew_size],
                'Renew - Date': [renew_date],
                'House No.': [house_number]}
    return data


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    med = np.median(image)
    low = int(max(0, .8 * med))
    high = int(min(255, 1.2 * med))
    return cv2.Canny(image, low, high)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def extract_cs(lines):
    match = re.search(r'Nearest Cross Street(.+)', lines)
    return match.group(1).strip() if match else ""


def extract_rs(lines):
    sizes = ['¾', '½', '1']
    for size in sizes:
        if size in lines:
            return size
    return ""


def extract_rd(lines):
    match = re.search(r'Renew (\d{2}/\d{2}/\d{4})', lines)
    return match.group(1) if match else ""


def extract_hn(lines):
    match = re.search(r'House No.(\d+)', lines)
    return match.group(1) if match else ""


def main():
    image = "/Users/allenwang/Downloads/two_images.pdf"
    document = fitz.open(image)

    info_list = []
    for page_num in range(document.page_count):
        page = document[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]

            extracted_info = extract_info(image_bytes)
            info_list.append(extracted_info)

            output_file_name = f"output_{page_num}_{img_index}.txt"
            with open(output_file_name, "w") as text_file:
                text_file.write(extracted_info)

    with open("output.json", "w") as json_file:
        json.dump(info_list, json_file, indent=2)


if __name__ == '__main__':
    main()