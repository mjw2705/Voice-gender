# Live voice-gender recognition

## Description

음성 기반의 성별 분류


## 모델 구축
### 1. Dataset
- 영어 Data - 1,440개 files
    - [RAVDESS](https://zenodo.org/record/1188976#.YjgTLupByBI) 

- 한국어 Data - 1,287개 files
    - 국내방송영화추출 DB
    - 연기자섭외자체 구축 DB 

- Total 2,727 audio files

### 2. Feature extraction
- 음성 데이터를 0.2s 단위로 shift 시키며 모든 음성 데이터를 동일하게 2.5s씩 load하여 [MFCC](https://librosa.org/doc/latest/feature.html) 특징 추출
- sampling rate = 22050
- n_mfcc  = 20

### 3. Train
- 음성 데이터에서 뽑은 1차원 특징들을 2차원으로 변환하여 CNN 모델의 feature로 활용
- 94% accuracy

## Demo
실시간 음성 성별 분류

<img src="./demo.gif" width="50%" height="50%">