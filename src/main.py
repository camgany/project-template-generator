from fastapi import FastAPI, Depends
from starlette.middleware.cors import CORSMiddleware
from src.llm_service import TemplateLLM
from src.prompts import ProjectParams
from src.parsers import ProjectIdeas
import io
import time
from fastapi.responses import JSONResponse
from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    HTTPException, 
    status,
    Depends
)
from fastapi.responses import Response
import numpy as np
from PIL import Image
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
from fastapi import UploadFile, Depends, File

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POSE_MODEL_PATH = "src/pose_landmarker_lite.task"

# Colocamos en una lista los datos de cada request de /poses
execution_logs = []
class PoseDetector:
    def __init__(self, model_path=POSE_MODEL_PATH):
        base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        self.model = vision.PoseLandmarker.create_from_options(options)

    def predict_image(self, image_array: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
        detection = self.model.detect(mp_image)
        results = detection.pose_landmarks

        # Clasificamos la pose
        pose_labels = self.classify_pose(results)

        return detection, pose_labels

    def classify_pose(self, pose_landmarks_list):
        classifications = []

        for pose_landmarks in pose_landmarks_list:
            right_wrist_y = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y
            left_wrist_y = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y

            if right_wrist_y > left_wrist_y:
                classifications.append("Derecha Levantada")
            else:
                classifications.append("Izquierda Levantada")

        return classifications

    def display_color_row(*imgs):
        for i, img in enumerate(imgs):
            print(type(img), img.dtype, img[0, 0])
            plt.subplot(1, len(imgs), i + 1)
            plt.imshow(img)
            plt.title(f"{i}")
            plt.xticks([])
            plt.yticks([])

pose_detector = PoseDetector()   

def get_pose_detector():
    return pose_detector

def predict_uploadfile(predictor, file):
    start_time = time.time()
    
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="Not an image"
        )
    img_obj = Image.open(img_stream)
    img_array = np.array(img_obj)
    
    # Realizamos la predicción
    results, pose_labels = predictor.predict_image(img_array)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return results, img_array, execution_time, pose_labels     
def get_llm_service():
    return TemplateLLM()


@app.post("/generate")
def generate_project(params: ProjectParams, service: TemplateLLM = Depends(get_llm_service)) -> ProjectIdeas:
    return service.generate(params)

@app.post("/poses")
def detect_poses(
    file: UploadFile = File(...), 
    predictor: PoseDetector = Depends(get_pose_detector)
) -> JSONResponse:
    results, img, execution_time, pose_labels = predict_uploadfile(predictor, file)

    pose_landmarks_list = results.pose_landmarks
    annotated_image = np.copy(img)

    pose_landmarks_proto = []

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())

    img_pil = Image.fromarray(annotated_image)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)

    headers = {
        "landmarks_found": str(pose_landmarks_proto.landmark),
        "pose_labels": pose_labels,
        "execution_time": str(execution_time),
        "image_size": str(img.size),
        "shape": str(img.shape),
        "dtype": str(img.dtype),
        "date": str(time.ctime()),
        "filename": str(file.filename),
        "content_type": str(file.content_type),  
    }
    
    execution_logs.append(headers)
        
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.get("/")
def root():
    return {"status": "OK"}
