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
import cv2
import matplotlib.pyplot as plt  # matplotlib import 수정
from age_model import ageDataset, model1, device, CFG, test_transform

def detect_face(image_path):
    image = fr.load_image_file(image_path)
    encodings = fr.face_encodings(image)
    if len(encodings) > 0:
        top, right, bottom, left = fr.face_locations(image)[0]
        face_image = image[top:bottom, left:right]
        return face_image
    else:
        return None

def predict_age(model, image):
    test_dataset = ageDataset(image=[image], label=[0], train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=0)  # num_workers를 0으로 설정

    model.eval()
    with torch.no_grad():
        for img, _ in tqdm(iter(test_loader)):
            img = img.to(device)
            logit = model(img)
            pred = logit.argmax(dim=1, keepdim=True)
            return logit, pred

if __name__ == "__main__":
    person_path = r'C:\Users\admin\Desktop\test\18.jpg'
    img = cv2.imread(person_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    
    
    face_image = detect_face(person_path)
    if face_image is None:
        print("얼굴이 인식되지 않습니다.")
    else:
        check_point = torch.load(r'C:\Users\admin\Desktop\test\age_best.pth')
        model = model1
        model = model.to(device)
        model.load_state_dict(check_point)

        logit, pred = predict_age(model, face_image)
        pred = pred.tolist()
        logit = logit.tolist()
        print(logit)
        print(pred)

        if pred[0][0] == 0:
            print("{:.2f}확률로 미성년자입니다.".format(logit[0][pred[0][0]]))
        elif pred[0][0] == 1:
            print("{:.2f}확률로 청년층입니다.".format(logit[0][pred[0][0]]))
        elif pred[0][0] == 2:
            print("{:.2f}확률로 중년층입니다.".format(logit[0][pred[0][0]]))
        elif pred[0][0] == 3:
            print("{:.2f}확률로 노년층입니다.".format(logit[0][pred[0][0]]))
