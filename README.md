## 파일 실행 방법

```
CMD > 실행파일 img [--op optionNumber] [--engine engineOption] 
```
* option은 1~12까지 있으며 최적의 결과가 나오는 것을 선택, default는 0 
* 이미지 preprocessing은 위의 메소드를 이용해 운용되고 있는 모듈값과 같음.
* engine은 추후 google vision을 이용 할 경우를 대비,,

 ## Dependencies Libraries
 
 ```
 opencv v3.4.5
 pytesseract v0.2.6
 ```

## 해야할 일

text에서 한글 / 영어 분리
폰트 training