import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from collections import Counter
import os
import face_recognition as fr
from tqdm import tqdm
import random
from age_model import criterion , model1 , train_loader , valid_loader , scheduler , device , optimizer


# 학습 부분 / 학습 함수 정의
def train(model, optimizer, train_loader, vali_loader, scheduler, device):
    model.to(device)      #모델에 디바이스 할당
    n = len(train_loader) #데이터 갯수 파악

    #Loss Function 정의
    #criterion = nn.CrossEntropyLoss().to(device)
    best_acc = 0
    best_epoch = 0
    val_loss = []
    tr_loss = []
    
    for epoch in range(1, 101): # 에포크 설정
        model.train()           # 모델 학습
        running_loss = 0.0
        
        for img, label in tqdm(iter(train_loader)):
            img, label = img.to(device), label.to(device) # 배치 데이터
            optimizer.zero_grad()                         # 배치마다 optimizer 초기화

            # Data -> Model -> Output
            logit = model(img)              # 예측값 산출
            loss = criterion(logit, label)  # 손실함수 계산

            # 역전파
            loss.backward()   # 손실함수 기준 역전파
            optimizer.step()  # 가중치 최적화
            running_loss += loss.item()
            train_loss = running_loss / len(train_loader)
        
        print('[%d] Train loss: %.10f' %(epoch, running_loss / len(train_loader)))
        tr_loss.append(running_loss / len(train_loader))
        
        if scheduler is not None:
            scheduler.step()

        # Validation set 평가
        # validation에서 argmax로 제일 높은 값의 인덱스를 뽑아 accuracy를 측정하였고 
        # validation의 정확도가 가장 높을때 model을 저장
        model.eval()     # evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
        vali_loss = 0.0
        correct = 0
        
        # 파라미터 업데이트 안하기 때문에 no_grad 사용
        with torch.no_grad():  
            for img, label in tqdm(iter(vali_loader)):
                img, label = img.to(device), label.to(device)

                logit = model(img)
                vali_loss += criterion(logit, label)
                pred = logit.argmax(dim=1, keepdim=True)                # 4개의 class중 가장 값이 높은 것을 예측 label로 추출
                correct += pred.eq(label.view_as(pred)).sum().item()    # 예측값과 실제값이 맞으면 1 아니면 0으로 합산
        
        vali_acc = 100 * correct / len(vali_loader.dataset)
        print('[유효성 검사] 손실: {:.4f}, 정확성: {}/{} ({:.1f}%)\n'.format(
            vali_loss / len(vali_loader), correct, len(vali_loader.dataset), vali_acc))
        
        val_loss.append(vali_loss / len(vali_loader))
        
         # early stopping기법 validation의 정확도가 최고치 갱신시 모델 저장
        if best_acc < vali_acc:
            best_acc = vali_acc
            best_epoch = epoch
            # best_model.pth을 저장
            torch.save(model.state_dict(), r'C:\Users\admin\Desktop\test\age_best.pth')
            print('모델이 저장되었습니다.')

    return best_acc, tr_loss, val_loss


if __name__ == "__main__":
    best_acc, tr_loss, val_loss = train(model1, optimizer, train_loader, valid_loader, scheduler, device)
