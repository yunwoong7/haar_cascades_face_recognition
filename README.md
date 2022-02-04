<h2 align="center">
Haar Cascades Face Recognition
</h2>

<div align="center">
  <img src="https://img.shields.io/badge/python-v3.8-blue.svg"/>
  <img src="https://img.shields.io/badge/opencv-v4.5.2.54-blue.svg"/>
</div>

2001년 Viola와 Jones가 "[Rapid Object Detection using a Boosted Cascade of Simple Features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)" 논문에서 특징(feature) 기반의 Object 검출 알고리즘(Haar cascades)을 소개하였습니다. 

근래에 많은 알고리즘(HOG + Linear SVM, SSD, Faster R-CNN, YOLO 등)이 Haar cascades보다 더 정확하지만 여전히 오늘날에도 Object 검출 연구와는 관련성이 있고 매우 유용합니다. 그리고 확실한 것은 Haar cascades 속도가 너무 빨라 그 속도를 능가하기 어렵다는 것입니다.

Haar cascades는 OpenCV와 여전히 사용되고 있고 리소스가 제한된 환경에서 많이 사용되고 있습니다.

알고리즘은 다음 4단계로 구성됩니다.

- Haar Feature Selection
- Creating Integral Images
- Adaboost Training
- Cascading Classifiers

OpenCV를 이용하여 Haar cascades를 사용하는 방법에 대해 소개하겠습니다.

------

OpenCV는 [사전 훈련된 Haar cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)를 제공합니다. 필요한 Haar cascades를 다운로드 후 진행하시기 바랍니다.

| **파일명**                                                   | **검출 대상**             |
| ------------------------------------------------------------ | ------------------------- |
| haarcascade_frontalface_default.xml<br/>haarcascade_frontalface_alt.xml<br/>haarcascade_frontalface_alt2.xml<br/>haarcascade_frontalface_alt_tree.xml | 정면 얼굴 검출            |
| haarcascade_profileface.xml                                  | 측면 얼굴 검출            |
| haarcascade_smile.xml                                        | 웃음 검출                 |
| haarcascade_eye.xml <br/>haarcascade_eye_tree_eyeglasses.xml <br/>haarcascade_lefteye_2splits.xml <br/>haarcascade_righteye_2splits.xml | 눈 검출                   |
| haarcascade_frontalcatface.xml<br/> haarcascade_frontalcatface_extended.xml | 고양이 얼굴 검출          |
| haarcascade_fullbody.xml                                     | 사람의 전신 검출          |
| haarcascade_upperbody.xml                                    | 사람의 상반신 검출        |
| haarcascade_lowerbody.xml                                    | 사람의 하반신 검출        |
| haarcascade_russian_plate_number.xml<br/> haarcascade_licence_plate_rus_16stages.xml | 러시아 자동차 번호판 검출 |

------

### **1. Import Packages**

```python
import matplotlib.pyplot as plt
import imutils
import cv2
import os
```

### **2. Function**

Colab 또는 Jupyter Notebook에서 이미지를 확인하기 위한 Function입니다.

```python
def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
```

### **3. Load Haar cascades**

```python
cascades_path = 'lib/haar_cascades'
 
detectorPaths = {
    "face": "haarcascade_frontalface_default.xml",
    "eyes": "haarcascade_eye.xml",
}
detectors = {}
    
for (name, path) in detectorPaths.items():
    path = os.path.sep.join([cascades_path, path])
    detectors[name] = cv2.CascadeClassifier(path)
```

### **4. Load Image**

```python
image_path = 'asset/images/Emmanuel_Macron.jpg'
 
image = cv2.imread(image_path)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### **5. Face Detection**

```python
faceRects = detectors["face"].detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
for (fX, fY, fW, fH) in faceRects:
    # 얼굴 ROI 추출
    faceROI = gray[fY:fY+ fH, fX:fX + fW]
    # 눈 ROI 추출
    eyeRects = detectors["eyes"].detectMultiScale(faceROI, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # 눈 bounding box
    for (eX, eY, eW, eH) in eyeRects:
        # draw the eye bounding box
        ptA = (fX + eX, fY + eY)
        ptB = (fX + eX + eW, fY + eY + eH)
        cv2.rectangle(image, ptA, ptB, (0, 0, 255), 2)
        
    # 얼굴 bounding box
    cv2.rectangle(image, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
```

### **6. Result**

```python
plt_imshow("Output", image, figsize=(16,10))
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/n33KY/btrstKCvxQk/wJxhvTPHeHxTsKcUbCUZGk/img.png" width="30%">
</div>

얼굴과 눈의 위치를 잘 찾았네요.

------

하지만..

Haar Cascades는 정면의 이미지에는 잘 찾지만 잘못 인식하는 경우도 매우 많습니다. 아래는 바위에서 많은 얼굴을 찾아냈네요.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/cf02Sh/btrswUq9AnB/mHy4hnazYCUVgCFlVyV9Xk/img.png" width="50%">
</div>

속도가 빠르기때문에 실시간으로 실행해도 문제가 없지만 보시다시피 인식 자체가 매우 정확한 편은 아닙니다.

<div align="center">
  <img src="/asset/images/img.gif" width="70%">
</div>

[[참고\] dlib, Python을 이용하여 얼굴 검출하기](https://yunwoong.tistory.com/83?category=902343)<br/>
[[참고\] dlib, Python을 이용하여 얼굴 인식하는 방법](https://yunwoong.tistory.com/84?category=902343)<br/>
[[참고\] dlib, Python을 이용하여 강아지 얼굴 인식하는 방법](https://yunwoong.tistory.com/86?category=902343)<br/>
