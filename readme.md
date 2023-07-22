# Project Title

## Overview
This project is a web framework configured using Django, with the frontend created using Bootstrap. The core technology of the backend is an AI-based prediction function that predicts the safety of water based on user-inputted data.

## Code Snippet
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
from django.conf import settings

# Define the model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)

class WaterAnalysis():
    def __init__(self) -> None:
        pass

    def water_predict(self, f_1, f_2, f_3, f_4):
        # Create a new model object
        new_model = Model()

        # Load the saved model
        model_path = 'waterApp/ai/model.pt'
        new_model.load_state_dict(torch.load(model_path))

        # Predict with the loaded model
        with torch.no_grad():
            x_test = torch.FloatTensor([[f_1, f_2, f_3, f_4]])  # set x_test data
            y_test = new_model(x_test)
            print(y_test)

        # Convert the tensor to a NumPy array
        output_numpy = y_test.detach().numpy()

        # Set a threshold to convert the output to binary labels
        threshold = 0.5
        binary_label = (output_numpy > threshold).astype(int)

        print(binary_label)
        if binary_label == [[0]]:
            result = "This water is not recommended to drink."
        else:
            result = "This water is expected to be drinkable."

        return result

# Test
if __name__ == "__main__":
    main = WaterAnalysis()

    f_1 = 0.01
    f_2 = 0.005
    f_3 = 0.001
    f_4 = 0.015

    water_pred = main.water_predict(f_1, f_2, f_3, f_4)

    print(water_pred)









Overview

이 프로젝트는 Django를 사용하여 웹 프레임워크를 구성하고, 프론트는 Bootstrap을 이용하여 작업하였습니다. 백엔드의 핵심 기술로는 인공지능을 이용한 예측 기능을 적용하여 사용자가 입력한 데이터를 기반으로 물의 안전성을 예측하도록 구현하였습니다.


Code Snippet

import torch import torch.nn as nn import torch.optim as optim import torch.nn.functional as F

import os from django.conf import settings

모델 정의
class Model(nn.Module): def init(self): super().init() self.linear = nn.Linear(4, 1)

def forward(self, x):
    return self.linear(x)
class WaterAnalysis():

def __init__(self) -> None:
    pass

def water_predict(self, f_1, f_2, f_3, f_4):
    # 새로운 모델 객체 생성
    new_model = Model()

    # 저장된 모델 불러오기
    model_path = 'waterApp/ai/model.pt'
    new_model.load_state_dict(torch.load(model_path))

    # 불러온 모델로 예측하기
    with torch.no_grad():
        x_test = torch.FloatTensor([[f_1, f_2, f_3, f_4]]) # x_test 데이터 설정
        y_test = new_model(x_test)
        print(y_test)

    # Convert the tensor to a NumPy array
    output_numpy = y_test.detach().numpy()

    # Set a threshold to convert the output to binary labels
    threshold = 0.5
    binary_label = (output_numpy > threshold).astype(int)

    print(binary_label)
    if binary_label == [[0]]:
        result = "This water is not recommended to drink."
    else:
        result = "This water is expected to be drinkable."

    return result
Test
if name == "main":

main = WaterAnalysis()

f_1 = 0.01
f_2 = 0.005
f_3 = 0.001
f_4 = 0.015

water_pred = main.water_predict(f_1, f_2, f_3, f_4)

print(water_pred)



Dependencies
Python 
Django 
Bootstrap 
PyTorch 



How to Run 
프로젝트를 실행하려면, 먼저 가상환경을 구성하고, 필요한 종속성을 설치하세요. 
그 다음 WaterAnalysis 클래스를 사용하여 water_predict 함수를 호출합니다.



License 이 프로젝트는 MIT 라이센스에 따라 라이센스가 부여됩니다.
