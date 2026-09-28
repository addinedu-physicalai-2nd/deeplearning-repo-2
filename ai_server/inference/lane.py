'''from ultralytics import YOLO

def load_lane_model():
    model = YOLO("models/lane.pt")
    model.to("cuda")
    return model

def run_lane(model, image):
    result = model.predict(image)
    return post_process_lane(result)


#작성 필요
def post_process_lane(result):

    return 

'''

def load_lane_model():
    return None  # 아직 로드할 모델 없음

def run_lane(model, image):
    return {"lane": []}  # 빈 결과지만 스키마는 맞춰서 반환