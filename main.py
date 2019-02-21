from pytesseract import image_to_string
from pytesseract import image_to_boxes
from pathlib import Path
import argparse
import os.path
import yaml
import cv2


from preprocessing import Preprocessing
from detection import TextDetection
from deskew import Deskew


doc = yaml.load(open('freset.yaml', 'r'))
bil_op = doc["bilateralFilterData"]
th_op = doc["thresholdData"]


def main():
    parser = argparse.ArgumentParser(description='OCR program')
    parser.add_argument('img_file', type=str, help="Input Image File Name")
    parser.add_argument('-o', '--option', type=int,
                        help="Input option number to 0 ~12", default=0)
    #parser.add_argument('-e', '--engine', type=str,
    #                    help="Select engine to Google-Vision and Tessreact")
    parser.add_argument('--natural', action='store_true', help='option for natural image')
    parser.add_argument('--noise', action='store_true', help='option for natural image')
    parser.add_argument('--text_boxes', action='store_true', help='display tesseract char detection')
    
    args = parser.parse_args()

    direct = Path(os.path.expanduser('~'))
    file_path = direct / args.img_file
    img = cv2.imread(str(file_path))
    
    print(" #pirnt image file path : " + str(file_path)) 
    
    if args.option:
        select_freset(args.option, img)
    else:
        default_freset(args.natural, args.noise, args.text_boxes, img)


def tesseract_ocr(img, lined=False):
    text = image_to_string(img, lang='kor+eng')
    if text:
        if lined:
            print(text),
        else:
            print(text) # debug


def tesseract_boxes(img):
    h, w = img.shape[:2]

    ## ------- result stdout and detection text boxing
    letters = image_to_boxes(img)
    letters = letters.split('\n')
    letters = [letter.split() for letter in letters]

    for letter in letters:
        cv2.rectangle(img, (int(letter[1]), h - int(letter[2])), (int(letter[3]), h - int(letter[4])), (0,0,255), 1)
     
    return img


def select_freset(op, img):
    _, img = cv2.threshold(img, int(th_op['threshVal'][op]),
                           int(th_op['threshMax'][0]),
                           cv2.THRESH_BINARY_INV if op % 2 else cv2.THRESH_BINARY)
    if op>10 and op<5:
        gray = cv2.bilateralFilter(gray, int(bil_op[0]), int(bil_op[1]), int(bil_op[2]))
    tesseract_ocr(img)


def default_freset(op, noise, boxes, img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #debug
    cv2.imshow('ori', img)
    cv2.waitKey()
    
    pcs_obj = Preprocessing(img)
    if op: 
        pcs_img = pcs_obj.natural_img_processing(noise)
    else:
        pcs_img = pcs_obj.digital_img_processing()

    #debug
    cv2.imshow('pre', pcs_img)
    cv2.waitKey()        

    skew_obj = Deskew(pcs_img)
    skew_img = skew_obj.run()

    #debug
    cv2.imshow('skew', skew_img)
    cv2.waitKey()        


    tdt_obj = TextDetection(skew_img, img)
    text_img = tdt_obj.detection()
  
    #debug
    flag = -1
    for t_img, y, x in text_img:
        #cv2.imshow('', t_img)
        #cv2.waitKey()
        if flag <= y+1 and flag >= y-1:
            tesseract_ocr(t_img, True)
        else:
            tesseract_ocr(t_img)
        
        if boxes: 
            b_img = tesseract_boxes(t_img)
            h, w = t_img.shape[:2]
            gray[y*2:y*2+h, x*2:x*2+w] = b_img
        flag = y

    #debug
    if boxes:
        cv2.imshow('origin', tesseract_boxes(img))
        cv2.imshow('result', gray)
        cv2.waitKey()
        
if __name__ == "__main__":
    main()
