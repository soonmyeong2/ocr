from pytesseract import image_to_string
import argparse
import cv2
from pathlib import Path
import os.path
import yaml

doc = yaml.load(open('freset.yaml', 'r'))
bil_op = doc["bilateralFilterData"]
th_op = doc["thresholdData"]


def main():
    parser = argparse.ArgumentParser(description='OCR program')
    parser.add_argument('img_file', type=str, help="Input Image File Name")
    parser.add_argument('--op', type=int,
                        help="Input option number to 0 ~12", default=0)
    parser.add_argument('--engine', type=str,
                        help="Select engine to Google-Vision and Tessreact")
    args = parser.parse_args()

    direct = Path(os.path.expanduser('~'))
    file_path = direct / 'Desktop' / args.img_file
    img = cv2.imread(str(file_path))
    select_freset(args.op, img)


def tesseract_orc_to_file(img):
    cv2.imshow('r',img)
    cv2.waitKey()
    text = image_to_string(img, lang='eng')
    print(text) # debug


def select_freset(op, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray, int(th_op['threshVal'][op]),
                           int(th_op['threshMax'][0]),
                           cv2.THRESH_BINARY_INV if op % 2 else cv2.THRESH_BINARY)
    if op>10 and op<5 and op!=0:
        gray = cv2.bilateralFilter(gray, int(bil_op[0]), int(bil_op[1]), int(bil_op[2]))
    tesseract_orc_to_file(img)


if __name__ == "__main__":
    main()