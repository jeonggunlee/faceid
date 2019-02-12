## OpenFace를 활용한 얼굴 인식

이 문서에서는 OpenFace를 이용한 얼굴 인식을 소개한다.

OpenFace를 이용하기 위해 필요한 라이브러리와 설치법은 다음과 같다.
```
sudo apt-get update
sudo apt-get upgrade
```

**OpenCV**
```
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev libv4l-dev v4l-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libqt4-dev mesa-utils libgl1-mesa-dri libqt4-opengl-dev libatlas-base-dev gfortran libeigen3-dev python2.7-dev python3-dev python-numpy python3-numpy
wget https://github.com/Itseez/opencv/archive/2.4.11.zip
unzip 2.4.11.zip
cd opencv-2.4.11
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
```

**dlib**
```
mkdir -p ~/src
cd ~/src
wget https://github.com/davisking/dlib/releases/download/v18.16/dlib-18.16.tar.bz2
tar xf dlib-18.16.tar.bz2
cd dlib-18.16/python_examples
mkdir build
cd build
cmake ../../tools/python
cmake --build . --config Release
sudo cp dlib.so /usr/local/lib/python2.7/dist-packages
```

**torch**
```
git clone https://github.com/torch/distro.git --recursive
cd torch/
./install-deps
./install.sh
source ~/.bashrc
for NAME in dpnn nn optim optnet csvigo cutorch cunn fblualib torchx tds; do luarocks install $NAME; done
```

**OpenFace**
```
git clone https://github.com/cmusatyalab/openface.git --recursive
cd openface
./models/get-models.sh
sudo pip2 install -r requirements.txt
sudo python2 setup.py install 
sudo pip2 install -r demos/web/requirements.txt
sudo pip2 install -r training/requirements.txt
```

**사진 변환하기**  
학습시킬 얼굴 이미지는 얼굴 부분만 추출한 뒤 얼굴이 같은 위치에 오도록 변환시켜야 한다.
옵션은 두 가지가 있는데 눈 바깥쪽과 코를 정렬하는 outerEyesAndNose, 눈 안쪽과 입술을 정렬하는 innerEyesAndBottomLip 이 있다.
학습에 이용할 얼굴 이미지는 쉽게 구할 수 있는 연예인 이미지이며 twice 일부 멤버, 닮은꼴 연예인으로 유명한 박소담과 김고은의 이미지와 내 사진을 각각 20장씩 저장하였다.
![example004](https://user-images.githubusercontent.com/39741011/52662393-6f6b5780-2f47-11e9-8984-3021f7193508.png)
변환에는 openface에서 제공된 소스코드를 사용하였다.
```
cd openface
./util/align-dlib.py [학습할 이미지들이 저장된 디렉토리] align [outerEyesAndNose or innerEyesAndBottomLip] [조정된 이미지를 저장할 디렉토리]
# ./util/align-dlib.py /home/wyh/face_image/input_image align outerEyesAndNose /home/wyh/face_image/aligned_image/
```
좌측이 변환되기 전의 사진이며 우측은 좌측에서 얼굴만을 추출하여 눈과 코의 위치를 조정한 사진이다.
얼굴을 탐색하는 능력이 뛰어나다는 것을 알 수 있다.
![example003](https://user-images.githubusercontent.com/39741011/52658821-4ba41380-2f3f-11e9-9472-60110e5cea45.png)

변환된 이미지들을 기학습된 DNN모델을 사용하여 각 얼굴에 대해 128개의 측정값을 얻는다.
```
./batch-represent/main.lua -outDir [측정값이 저장될 디렉토리] -data [조정된 이미지들이 저장된 디렉토리]
# ./batch-represent/main.lua -outDir /home/wyh/face_image/embeddings/ -data /home/wyh/face_image/aligned_image
```

위의 결과값을 사용하여 분류기를 훈련시킨다.
그 결과로 embeddings 디렉토리에 classifiers.pkl 이 생성된다.
```
./demos/classifier.py train [측정값이 저장된 ]
# ./demos/classifier.py train /home/wyh/face_image/embeddings/
```

학습된 결과로 얼굴 인식을 진행한다.  
**cam_identify.py**
```python
#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import dlib
from skimage import io
import sys
import openface
import os
import pickle

import numpy as np
# 소수점 두 번째 자리까지 출력
np.set_printoptions(precision=2)

# 얼굴 탐색을 위한 클래스
face_detector = dlib.get_frontal_face_detector()
# 매 프레임 얼굴인식을 수행할 경우 속도가 매우 느려지므로 보다 원활한 실시간 처리를 위해 프레임을 절반으로 줄인다.
frame_cycle = 2
frame_check = 0
# 기학습된 DNN모델의 patg
modelPath = sys.argv[1]

# 캠으로부터 영상을 얻는다.
cam = cv2.VideoCapture(0)

if len(sys.argv) != 4:
	quit(0)

# 얼굴 정렬을 위한 클래스
align = openface.AlignDlib(sys.argv[2])
net = openface.TorchNeuralNet(model=modelPath)

while True:
	ret_val, img = cam.read()

	# 프레임을 절반으로 줄이는 과정
	if frame_check%frame_cycle == 0:
		frame_check = 1

		# 처리속도 증가를 위해 캠으로부터 얻은 영상의 크기를 4분의 1로 줄인다.
		img_resize = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

		# 크기를 줄인 영상에서 얼굴을 탐색한다.
		detected_faces = align.getAllFaceBoundingBoxes(img_resize)
		
		# 찾은 얼굴들의 눈과 코의 위치를 조정한다.
		aligned_faces = []
		for detected_face in detected_faces:
			aligned_faces.append(align.align(96, img_resize, detected_face, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

		# forward 함수
		reps = []
		for aligned_face in aligned_faces:
			reps.append(net.forward(aligned_face))

		# 학습 결과를 읽어온다.
		with open(sys.argv[3]) as f:
			if sys.version_info[0] < 3:
				(le, clf) = pickle.load(f)
			else:
				(le, clf) = pickle.load(f, encoding='latin1')
				
		# 학습 결과를 바탕으로 가장 높은 일치율을 갖는 사람의 비율과 확률을 얻는다.
		persons = []
		confidences = []
		for rep in reps:
			rep = rep.reshape(1, -1)

			predictions = clf.predict_proba(rep).ravel()
			maxI = np.argmax(predictions)
			persons.append(le.inverse_transform(maxI))
			confidences.append(predictions[maxI])

		# 위에서 얻는 결과로 각 얼굴에 사각형을 만들고 이름과 확률을 출력한다.
		for idx, person in enumerate(persons):
			top = detected_faces[idx].top() * 4
			bottom = detected_faces[idx].bottom() * 4
			left = detected_faces[idx].left() * 4
			right = detected_faces[idx].right() * 4

			cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
			cv2.rectangle(img, (left, bottom), (right, bottom+25), (0, 255, 0), -1)
			cv2.putText(img, "{} @{:.2f}".format(person, confidences[idx]), (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
		# 화면에 
		cv2.imshow("Cam Viewer", img)

	else:
		frame_check = frame_check+1


	key = cv2.waitKey(1)

	# esc 입력 시 종료, 'c' 입력 시 캡쳐
	if key%256 == 27:
		break
	if key%256 == 99:
		cv2.imwrite('opencv.png', img)
		print("Captured");

```

```
./cam_identify.py [기학습된 DNN 모델: openface/models/openface/nn4.small2.v1.t7] [얼굴 조정을 위한 68개의 랜드마크 데이터: openface/models/dlib/shape_predictor_68_face_landmarks.dat] [학습된 결과: classifier.pkl]
# ./cam_identify.py /home/wyh/OpenFace/openface/models/openface/nn4.small2.v1.t7 
 /home/wyh/OpenFace/openface/models/dlib/shape_predictor_68_face_landmarks.dat /home/wyh/face_image/embeddings/classifier.pkl
```
**실행결과**
![result1](https://user-images.githubusercontent.com/39741011/52668492-0db2e980-2f57-11e9-8b09-d936e4e50ee5.png)
정면은 물론 옆 얼굴도 인식이 잘 된다.

![result2](https://user-images.githubusercontent.com/39741011/52668659-7bf7ac00-2f57-11e9-9004-1b8b83ebad5f.png)
트와이스 멤버들 또한 잘 인식되는 것을 볼 수 있다.

![result3](https://user-images.githubusercontent.com/39741011/52668714-a0538880-2f57-11e9-819c-f7629b2a3130.png)
닮은꼴 연예인으로 유명한 김고은과 박소담도 잘 구분된다.

**개선할 점**  
정확도에 대해서는 더 많은 사진을 학습에 이용하면 높아질 것이므로 더 많은 사진자료 수집이 필요하다.
실시간으로 얼굴인식이 가능하지만 웹캠의 프레임이 낮아지는 현상이 발생한다. 좀 더 빠르게 처리할 수 있는 방법이 필요하다.

1. [OpenFace](https://cmusatyalab.github.io/openface/)
2. https://github.com/cmusatyalab/openface
3. [엑소사랑하자 - OpenFace로 우리 오빠들 얼굴 인식하기](https://www.popit.kr/openface-exo-member-face-recognition/)
4. [딥러닝 기반 고성능 얼굴인식 기술 동향](https://ettrends.etri.re.kr/ettrends/172/0905172005/33-4_43-53.pdf)
