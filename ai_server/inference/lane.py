from ultralytics import YOLO

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