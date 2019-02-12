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
변환에는 openface에서 제공된 소스코드를 사용하였다.
```
cd openface
./util/align-dlib.py /home/wyh/face_image/input_image align outerEyesAndNose /home/wyh/face_image/aligned_image/
```


1. [OpenFace](https://cmusatyalab.github.io/openface/)
2. https://github.com/cmusatyalab/openface
3. [엑소사랑하자 - OpenFace로 우리 오빠들 얼굴 인식하기](https://www.popit.kr/openface-exo-member-face-recognition/)
4. [딥러닝 기반 고성능 얼굴인식 기술 동향](https://ettrends.etri.re.kr/ettrends/172/0905172005/33-4_43-53.pdf)
