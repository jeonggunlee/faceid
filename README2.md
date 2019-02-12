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
```
1. [OpenFace](https://cmusatyalab.github.io/openface/)
2. https://github.com/cmusatyalab/openface
3. [엑소사랑하자 - OpenFace로 우리 오빠들 얼굴 인식하기](https://www.popit.kr/openface-exo-member-face-recognition/)
4. [딥러닝 기반 고성능 얼굴인식 기술 동향](https://ettrends.etri.re.kr/ettrends/172/0905172005/33-4_43-53.pdf)
