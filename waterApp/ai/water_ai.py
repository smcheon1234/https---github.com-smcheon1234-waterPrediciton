import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
from django.conf import settings


# 모델 정의
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
        # 새로운 모델 객체 생성
        new_model = Model()

        # 저장된 모델 불러오기
        # model_path = os.path.join(settings.BASE_DIR, 'models', 'model.pt')
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
    


# Test
if __name__ == "__main__":

    main = WaterAnalysis()

    f_1 = 0.01
    f_2 = 0.005
    f_3 = 0.001
    f_4 = 0.015
    
    water_pred = main.water_predict(f_1, f_2, f_3, f_4)

    print(water_pred)

    