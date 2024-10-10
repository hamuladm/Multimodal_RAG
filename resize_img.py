import cv2 as cv

IMG_DIR = "scraped_images"
TEXT_DIR = "scraped_text"


def resize_imgs(path: str, scaling: float):
    if not path.endswith(".gif"):
        scaling_x = scaling_y = scaling
        source_img = cv.imread(path)
        resized_img = cv.resize(source_img, (0, 0), fx=scaling_x, fy=scaling_y)
        return resized_img
    else:
        return "GIF", path