동양인 얼굴 인식해서 나이 구별하는 코드
python VSCode 사용 

동양인 얼굴 set이 필요.

age_model.py -> train_model.py -> age_model_test.py 순서대로 실행하면 됩니다.
                 
    
train_model.py 실행 시 : age_best.pth 생성 ( age_model.py 정보 기반으로 resnet18 모델 기반 훈련된 모델 생성 )
preprocess_images.py 실행 시 : preprocessed_data.pkl 생성 ( 데이터 전처리 작업 오래걸려서 캐싱 ) 
age_model_test.py 실행 시 : face_recognition (얼굴 검출 모델) 기반으로 얼굴 검출 후 유아층 , 청년층 , 중년층 , 노년층 구분 (4 추가 시 70세 이상 가능하나 데이터 셋이 81살까지라 딱히 구분 넣지 않음)