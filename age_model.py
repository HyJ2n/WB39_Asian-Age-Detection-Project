import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import optimizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from collections import Counter
import os
import face_recognition as fr
from tqdm import tqdm
import random
from preprocess_images import preprocess_images 


# 배경노이즈 제거 위한 얼굴위치인식 모델
#  cfg(config) 파일 :  모델 학습 및 추론에 필요한 다양한 설정 정보
CFG = {
    'IMG_SIZE':128, #이미지 사이즈128
    'EPOCHS':100, #에포크
    'BATCH_SIZE':16, #배치사이즈
    'SEED':1, #시드
}

# 장치 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 시드 설정
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)

# 데이터셋과 모델 설정
class ageDataset(Dataset):
    def __init__(self, image, label, train=True, transform=None):
        self.transform = transform
        self.img_list = image
        self.label_list = label

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        label = self.label_list[idx]
        img = Image.fromarray(np.uint8(self.img_list[idx])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# custom dataset 사용 transform 
train_transform = torchvision.transforms.Compose([
                    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),          # 각 이미지 같은 크기로 resize
                    transforms.ToTensor(),                                          # 이미지를 텐서로 변환
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # 평균과 표준편차를 0.5로 정규화
                    ])

test_transform = torchvision.transforms.Compose([
                    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),          # 각 이미지 같은 크기로 resize
                    transforms.ToTensor(),                                          # 이미지를 텐서로 변환
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # 평균과 표준편차를 0.5로 정규화
                    ])
Horizontal_transform=torchvision.transforms.Compose([
                    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),          # 각 이미지 같은 크기로 resize
                    transforms.RandomHorizontalFlip(1.0),                           # Horizontal = 좌우반전
                    transforms.ToTensor(),                                          # 이미지를 텐서로 변환
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # 평균과 표준편차를 0.5로 정규화
                    ])



# 예시로 train_path를 정의하고 사용 (실제 데이터셋 경로에 맞게 설정)
train_path = sorted(glob.glob(r"C:\Users\admin\Desktop\test\All-Age-Faces Dataset\original images\*.jpg"))
    
    
# 이미지 전처리 실행
train_path2, face_list = preprocess_images(train_path)


# 파일명을 이용한 나이 추출 및 레이블 설정
# 00003A02.jpg 에서 A와 . 사이는 나이 정보
file_name = []
for id in train_path2:
    file_name_with_extension = os.path.basename(id)  # 00003A02.jpg
    name = file_name_with_extension.split('A')[1]    # 02.jpg
    age = name.split('.')[0]                         # 02
    file_name.append(age)                            # 파일명의 A와 . 사이의 나이정보를 label로 설정
    

# str 형태를 int로 변경
label_list = list(map(int, file_name))

# 나이 레이블링
train_y = []
for age in label_list:
    if age < 15:
        train_y.append(0)   # 0세 이상 15세 미만 (어린이)
    elif age < 30:
        train_y.append(1)   # 15세 이상 30세 미만 (청소년 및 청년)
    elif age < 50: 
        train_y.append(2)   # 30세 이상 50세 미만 (중년)
    elif age < 70:
        train_y.append(3)   # 50세 이상 70세 미만 (노년)
    else:
        train_y.append(4)   # 최고령

# 데이터셋 설정 및 분할
# 시드 고정 후 shuffle : 독립변수와 종속변수가 어긋나는 것을 막음
# 독립 변수 (=원인변수 , 예측변수 , 설명변수 , 가설변수) : 종속변수의 변화를 가져오거나 영향을 미치는 변수 ,       <결과 예측을 하게되는 변수>
# 종속 변수 (=결과변수 , 피예측변수 , 피설명변수 , 준거변수) : 독립변수의 영향으로 나타나는 결과가 되는 결과 변수 , <예측되는 변수>
random.Random(19991006).shuffle(face_list) 
random.Random(19991006).shuffle(train_path2)
random.Random(19991006).shuffle(train_y)

train_img_list = face_list[:int(len(face_list)*0.8)]   # 0.8비율로 trainset과 validationset으로 스플릿
train_label_list = train_y[:int(len(train_y)*0.8)]
valid_img_list = face_list[int(len(face_list)*0.8):]
valid_label_list = train_y[int(len(train_y)*0.8):]

# 가중치 설정 및 데이터로더 생성
def make_weights(labels, nclasses):
    labels = np.array(labels)
    weight_arr = np.zeros_like(labels)

    _, counts = np.unique(labels, return_counts=True)
    for cls in range(nclasses):
        weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr)
        # 각 클래스의의 인덱스를 산출하여 해당 클래스 개수의 역수를 확률로 할당
        # 이를 통해 각 클래스의 전체 가중치를 동일하게 함

    return weight_arr

# 각 weights의 모든 가중치의 합은 1
weights = make_weights(train_label_list, 4)
weights = torch.DoubleTensor(weights)
weights1 = make_weights(valid_label_list, 4)
weights1 = torch.DoubleTensor(weights1)

# 데이터 불균형을 해결하기 위한 sampler를 정의
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
sampler1 = torch.utils.data.sampler.WeightedRandomSampler(weights1, len(weights1))

train_dataset = ageDataset(image=train_img_list, label=train_label_list, train=True, transform=train_transform)
valid_dataset = ageDataset(image=valid_img_list, label=valid_label_list, train=False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, num_workers=2, sampler=sampler)
valid_loader = DataLoader(valid_dataset, batch_size=16, num_workers=2, sampler=sampler1)

# 모델 정의 및 학습 함수
# resnet18모델을 프로젝트의 아웃풋에 맞게 수정
class ResNetAgeModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetAgeModel, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)

model1 = ResNetAgeModel(num_classes=4)
model1.to(device)

# loss function으로는 multioutput classification이기 때문에 
# crossentropy 최적화 함수는 adam을 사용
criterion = torch.nn.CrossEntropyLoss() # loss function으로 crossentropy 설정
optimizer = optim.Adam(model1.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = None
