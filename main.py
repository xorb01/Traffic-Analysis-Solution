import cv2
import numpy as np
from ultralytics import YOLO
import os
import time 
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
"""
# 데이터셋 루트 경로
dataset_dir = "C:/Users/xorb6/Desktop/Vehicle_Detection_Image_Dataset"  

# 학습용(train) 이미지/라벨 경로
train_images_dir = os.path.join(dataset_dir, "train", "images")
train_labels_dir = os.path.join(dataset_dir, "train", "labels")

# 검증용(val) 이미지/라벨 경로
val_images_dir = os.path.join(dataset_dir, "valid", "images")
val_labels_dir = os.path.join(dataset_dir, "valid", "labels")

# Train 데이터 확인
train_image_files = [f for f in os.listdir(train_images_dir) if f.endswith(".jpg")]
train_label_files = [f for f in os.listdir(train_labels_dir) if f.endswith(".txt")]

print(f"Train 이미지 수: {len(train_image_files)}")
print(f"Train 라벨 수: {len(train_label_files)}")

# Validation 데이터 확인
val_image_files = [f for f in os.listdir(val_images_dir) if f.endswith(".jpg")]
val_label_files = [f for f in os.listdir(val_labels_dir) if f.endswith(".txt")]

print(f"Validation 이미지 수: {len(val_image_files)}")
print(f"Validation 라벨 수: {len(val_label_files)}")

base_dir = "C:/Users/xorb6/Desktop/Vehicle_Detection_Image_Dataset"
train_images_dir = os.path.join(base_dir, "train", "images")
train_labels_dir = os.path.join(base_dir, "train", "labels")

#  랜덤으로 이미지 1장 뽑기
image_files = [f for f in os.listdir(train_images_dir) if f.endswith(".jpg")]
sample_file = random.choice(image_files)

#  이미지 읽기
img_path = os.path.join(train_images_dir, sample_file)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Matplotlib용 변환
h, w, _ = img.shape

#  라벨 읽기
label_path = os.path.join(train_labels_dir, sample_file.replace(".jpg", ".txt"))
with open(label_path, "r") as f:
    lines = f.readlines()

#  시각화 (빨간 박스 그리기)
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.imshow(img)

for line in lines:
    # YOLO 좌표 역변환
    cls, x_c, y_c, bw, bh = map(float, line.strip().split())
    x1 = (x_c - bw/2) * w
    y1 = (y_c - bh/2) * h
    width = bw * w
    height = bh * h
    
    # 사각형 그리기
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-5, 'Vehicle', color='white', backgroundcolor='red', fontsize=10)

plt.title(f"Ground Truth Check: {sample_file}")
plt.axis("off")
plt.show()

#  데이터 경로 설정
dataset_dir = r"C:/Users/xorb6/Desktop/Vehicle_Detection_Image_Dataset"
train_images_dir = os.path.join(dataset_dir, "train", "images")

#  Train 이미지 파일 리스트
train_image_files = [f for f in os.listdir(train_images_dir) if f.endswith(".jpg")]
print(f"총 Train 이미지 수: {len(train_image_files)}")

#  COCO 사전학습 YOLOv8 모델 불러오기
model = YOLO("yolov8n.pt") 

#  랜덤 1장 이미지 선택
sample_files = random.sample(train_image_files, 1)

#  랜덤 이미지 탐지 및 시각화 (Matplotlib Patches 활용)
for img_file in sample_files:
    img_path = os.path.join(train_images_dir, img_file)
    
    #  이미지 읽기 (OpenCV 에서 Matplotlib)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #  YOLO 탐지 실행
    results = model.predict(source=img_path, imgsz=640, conf=0.5, verbose=False)
    
    #  시각화 준비
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    #  탐지된 박스 정보 하나씩 꺼내서 그리기
    for box in results[0].boxes:
        # 좌표 가져오기 (xyxy: 좌상단 x, y, 우하단 x, y)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # 너비와 높이 계산
        w = x2 - x1
        h = y2 - y1
        
        # 정보 가져오기 (신뢰도, 클래스명)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        
        # 사각형 그리기 (파란색)
        # Prediction(예측)은 파란색 사용하여 Ground Truth(빨강)와 구분.
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        
        # 라벨 텍스트 추가 (배경색 포함)
        label_text = f"{class_name} {conf:.2f}"
        ax.text(x1, y1 - 5, label_text, color='white', fontsize=10, backgroundcolor='blue')

    plt.title(f"YOLOv8 Prediction (Zero-shot): {img_file}", fontsize=16)
    plt.axis('off')
    plt.show() 

def train_model():
    model = YOLO('yolov8n.pt')

    model.train(
        data='data.yaml',   # data.yaml 파일이 있는 경로
        epochs=50,
        imgsz=640,
        batch=4,
        degrees=10.0,
        device='cpu',       
        name='traffic_analysis_v1' # 결과가 저장될 폴더
    )
    

    #  검증
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")


"""
VIDEO_PATH = "SampleVideo1.mp4" 
MODEL_PATH = "best.pt"

OUTPUT_FILENAME = "result0104.mp4"

# 영상 해상도 및 라인 
FRAME_WIDTH = 1920  
FRAME_HEIGHT = 720  

# 좌/우 차선을 나누는 X축 기준점
DIVIDER_X = 900  
# Y축 높이 
LINE_Y = 650  

# 좌측 차선 카운팅 라인 (시작점, 끝점)
LINE_LEFT_START = (0, LINE_Y)          
LINE_LEFT_END = (DIVIDER_X, LINE_Y)

# 우측 차선 카운팅 라인 (시작점, 끝점)
LINE_RIGHT_START = (DIVIDER_X, LINE_Y) 
LINE_RIGHT_END = (1920, LINE_Y) 


############################################################################################################
#  CCW
############################################################################################################

def ccw(p1, p2, p3): # 외적값
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def is_intersect(p1, p2, p3, p4):
    d1 = ccw(p1, p2, p3) * ccw(p1, p2, p4) # 차량 위치
    d2 = ccw(p3, p4, p1) * ccw(p3, p4, p2) # 라인
    if d1 <= 0 and d2 <= 0:
        if d1 == 0 and d2 == 0: return False
        return True
    return False

############################################################################################################
# 박스 및 대시보드
############################################################################################################

def draw_custom_box(frame, box, track_id, class_name, conf, color=(0, 255, 0)):
    x, y, w, h = box
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    
    # 박스 그리기
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # 라벨 그리기
    label = f"ID:{track_id} Conf:{conf:.2f}"
    font_scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (255, 255, 255), thickness)

def draw_dashboard(frame, left, right):
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 하단 검은색 배경 생성
    cv2.rectangle(overlay, (0, h - 150), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 왼쪽 차선 카운트 표시 (파란색)
    cv2.putText(frame, "LEFT LANE", (50, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, f"{left}", (50, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)

    # 오른쪽 차선 카운트 표시 (빨간색)
    cv2.putText(frame, "RIGHT LANE", (300, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, f"{right}", (300, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

    

############################################################################################################
# [메인 실행] 영상 처리 및 저장 루프
############################################################################################################

def run_final_v6_save():
    # 모델 로드
    target_model = MODEL_PATH if os.path.exists(MODEL_PATH) else 'yolov8n.pt'
    print(f"모델 로드: {target_model}")
    model = YOLO(target_model)
    
    # 저장 경로 
    try:
        save_dir = os.path.dirname(os.path.dirname(MODEL_PATH))
    except:
        save_dir = os.getcwd() 

    save_path = os.path.join(save_dir, OUTPUT_FILENAME)
    print(f"결과 영상 저장 위치: {save_path}")

    # 비디오 입력 
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 비디오 저장 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # 화면 창 설정
    cv2.namedWindow('Advanced Traffic System', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Advanced Traffic System', 1280, 720)

    # 변수 초기화
    counted_ids = set()
    left_count = 0
    right_count = 0
    previous_positions = {} 
    
    # FPS 측정용 변수 초기화
    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
        
        draw_frame = frame.copy()

        #############################################################################################################
        #  Bytetrack 트래킹 설정 
        #############################################################################################################
        results = model.track(
            draw_frame, 
            persist=True, 
            tracker="bytetrack.yaml",
            verbose=False,
            conf=0.2,         # 흐릿한 차도 잡기 위해 낮춤 (0.3 -> 0.15)           0.15 = 68대 예측  0.1  =72
            iou=0.7,           # 겹침 허용치를 낮춰 중복 박스 제거 (0.85 -> 0.5)     0.5 = 68대 예측. 0.7 = 72
            agnostic_nms=True  # 클래스가 달라도 겹치면 하나로 병합
        )
        
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            current_cars = len(track_ids)

            for box, track_id, conf, cls in zip(boxes, track_ids, confs, clss):
                x, y, w, h = box
                cur_center = (int(x), int(y))
                class_name = model.names[cls] 

                # 색상 설정 (왼쪽: 파랑, 오른쪽: 빨강)
                color = (255, 0, 0) if x < DIVIDER_X else (0, 0, 255)
                draw_custom_box(draw_frame, box, track_id, class_name, conf, color)

                # [카운팅 로직] CCW 사용
                if track_id in previous_positions:
                    prev_center = previous_positions[track_id]

                    # 왼쪽 차선 교차 검사
                    if is_intersect(LINE_LEFT_START, LINE_LEFT_END, prev_center, cur_center):
                        if track_id not in counted_ids:
                            counted_ids.add(track_id)
                            left_count += 1
                            cv2.line(draw_frame, LINE_LEFT_START, LINE_LEFT_END, (255, 255, 255), 5)

                    # 오른쪽 차선 교차 검사
                    elif is_intersect(LINE_RIGHT_START, LINE_RIGHT_END, prev_center, cur_center):
                        if track_id not in counted_ids:
                            counted_ids.add(track_id)
                            right_count += 1
                            cv2.line(draw_frame, LINE_RIGHT_START, LINE_RIGHT_END, (255, 255, 255), 5)
                
                # 현재 위치 저장
                previous_positions[track_id] = cur_center


        # 라인 및 대시보드 그리기
        cv2.line(draw_frame, LINE_LEFT_START, LINE_LEFT_END, (255, 0, 0), 2)
        cv2.line(draw_frame, LINE_RIGHT_START, LINE_RIGHT_END, (0, 0, 255), 2)
        draw_dashboard(draw_frame, left_count, right_count)

        ############################################################################################################
        # FPS 표시 로직
        ############################################################################################################

        curr_time = time.time()
        fps_val = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(draw_frame, f"FPS: {fps_val:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 결과 저장 및 출력
        out.write(draw_frame)
        cv2.imshow("Advanced Traffic System", draw_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # 리소스 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_final_v6_save()