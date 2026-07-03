import os
import cv2
import mediapipe as mp
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional

app = FastAPI(title="FaceLab API")

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def decode_image(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# MediaPipe Tasks setup
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Get absolute path to the backend directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Face Detection setup
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

face_model_path = os.path.join(CURRENT_DIR, "blaze_face_short_range.tflite")
face_detector = None
if os.path.exists(face_model_path):
    fd_options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=VisionRunningMode.IMAGE
    )
    face_detector = FaceDetector.create_from_options(fd_options)

# Age Guess setup
age_model_path = os.path.join(CURRENT_DIR, "age_net.caffemodel")
age_proto_path = os.path.join(CURRENT_DIR, "age_deploy.prototxt")
age_net = None
if os.path.exists(age_model_path) and os.path.exists(age_proto_path):
    age_net = cv2.dnn.readNetFromCaffe(age_proto_path, age_model_path)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Face Landmarker setup
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

face_landmarker_path = os.path.join(CURRENT_DIR, "face_landmarker.task")
landmarker = None
if os.path.exists(face_landmarker_path):
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_landmarker_path),
        running_mode=VisionRunningMode.IMAGE
    )
    landmarker = FaceLandmarker.create_from_options(options)

# Hair Color setup
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions

segmenter_path = os.path.join(CURRENT_DIR, "hair_segmenter.tflite")
segmenter = None
if os.path.exists(segmenter_path):
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=segmenter_path),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=False,
        output_confidence_masks=True
    )
    segmenter = ImageSegmenter.create_from_options(options)

@app.post("/api/face-detect")
async def face_detect(file: UploadFile = File(...)):
    if face_detector is None:
        return {"status": "error", "message": "Face model not loaded."}
        
    img = decode_image(await file.read())
    if img is None:
        return {"status": "error", "message": "Invalid image"}
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = face_detector.detect(mp_image)
    
    count = 0
    if results.detections:
        for detection in results.detections:
            count += 1
            bboxC = detection.bounding_box
            x, y, w, h = bboxC.origin_x, bboxC.origin_y, bboxC.width, bboxC.height
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"Face {count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    base64_img = encode_image(img)
    return {
        "status": "success",
        "message": f"Detected {count} faces.",
        "result_b64": f"data:image/jpeg;base64,{base64_img}"
    }

@app.post("/api/age-guess")
async def age_guess(file: UploadFile = File(...)):
    if age_net is None:
        return {"status": "error", "message": "Age model not loaded. Run setup scripts first."}
        
    img = decode_image(await file.read())
    if img is None:
        return {"status": "error", "message": "Invalid image"}
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = face_detector.detect(mp_image)
    
    age_preds = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.bounding_box
            x, y, w, h = bboxC.origin_x, bboxC.origin_y, bboxC.width, bboxC.height
            ih, iw, _ = img.shape
            
            # Calculate center and make a square bounding box with padding
            # This preserves aspect ratio and prevents the Caffe model from squishing the face,
            # which drastically improves age detection accuracy.
            cx = x + w // 2
            cy = y + h // 2
            size = int(max(w, h) * 1.5)
            half_size = size // 2
            
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(iw, cx + half_size)
            y2 = min(ih, cy + half_size)
            
            face_img = img[y1:y2, x1:x2].copy()
            if face_img.size == 0:
                continue
                
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            age_net.setInput(blob)
            preds = age_net.forward()
            age = AGE_LIST[preds[0].argmax()]
            age_preds.append(age)
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
    base64_img = encode_image(img)
    msg = f"Guessed age for {len(age_preds)} faces." if age_preds else "No faces detected."
    return {
        "status": "success",
        "message": msg,
        "result_b64": f"data:image/jpeg;base64,{base64_img}"
    }

@app.post("/api/hair-color")
async def hair_color(file: UploadFile = File(...), color: Optional[str] = Form(None)):
    if segmenter is None:
        return {"status": "error", "message": "Segmenter model not loaded. Run setup scripts first."}
        
    img = decode_image(await file.read())
    if img is None:
        return {"status": "error", "message": "Invalid image"}
        
    # MediaPipe Tasks expects an mp.Image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    segmentation_result = segmenter.segment(mp_image)
    if not segmentation_result.confidence_masks:
        return {"status": "error", "message": "Failed to segment image."}
        
    # Use the confidence mask (float32, 0.0 to 1.0) for extremely accurate soft boundaries
    # Index 1 is the hair class in hair_segmenter.tflite
    hair_mask_float = np.squeeze(segmentation_result.confidence_masks[1].numpy_view())
    
    if np.max(hair_mask_float) < 0.1:
        return {"status": "success", "message": "No hair detected.", "result_b64": f"data:image/jpeg;base64,{encode_image(img)}"}
    
    # Enhance the matte to catch subtle semi-transparent outer strands perfectly
    # Gamma correction on the mask to boost mid-tones (outer strands)
    hair_mask_float = np.power(hair_mask_float, 0.8)
    
    # Apply color using advanced Soft Light blending
    if color:
        color = color.lstrip('#')
        bgr_color = np.array([int(color[4:6], 16), int(color[2:4], 16), int(color[0:2], 16)], dtype=np.uint8)
    else:
        bgr_color = np.array([128, 0, 128], dtype=np.uint8)
    
    # Normalize images to 0-1 for blending
    base = img.astype(np.float32) / 255.0
    
    # Create solid color layer
    color_layer = np.full_like(img, bgr_color)
    blend = color_layer.astype(np.float32) / 255.0
    
    # Soft Light blend mode (Professional industry standard for natural color tinting)
    D = np.where(base <= 0.25, 
                 ((16 * base - 12) * base + 4) * base, 
                 np.sqrt(base))
                 
    colorized_float = np.where(blend <= 0.5,
                   base - (1 - 2 * blend) * base * (1 - base),
                   base + (2 * blend - 1) * (D - base))
    
    # Prepare 3D alpha matte
    alpha_matte = np.repeat(hair_mask_float[:, :, np.newaxis], 3, axis=2)
    
    # Combine original image with colorized image using the high-precision alpha matte
    blended = (base * (1.0 - alpha_matte) + colorized_float * alpha_matte)
    
    # Convert back to uint8
    blended_uint8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    
    base64_img = encode_image(blended_uint8)
    return {
        "status": "success",
        "message": "Hair color changed successfully.",
        "result_b64": f"data:image/jpeg;base64,{base64_img}"
    }

@app.post("/api/face-shape")
async def face_shape(file: UploadFile = File(...)):
    img = decode_image(await file.read())
    if img is None:
        return {"status": "error", "message": "Invalid image"}
        
    if landmarker is None:
        return {"status": "error", "message": "Face landmarker model not loaded. Run setup scripts first."}
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    results = landmarker.detect(mp_image)
    
    if not results.face_landmarks:
        return {"status": "error", "message": "No face detected."}
        
    landmarks = results.face_landmarks[0]
    ih, iw, _ = img.shape
    
    def get_pt(idx):
        pt = landmarks[idx]
        return np.array([pt.x * iw, pt.y * ih])
        
    # Extract key points
    top = get_pt(10)
    chin = get_pt(152)
    face_length = np.linalg.norm(top - chin)
    
    left_cheek = get_pt(234)
    right_cheek = get_pt(454)
    face_width = np.linalg.norm(left_cheek - right_cheek)
    
    left_jaw = get_pt(132)
    right_jaw = get_pt(361)
    jaw_width = np.linalg.norm(left_jaw - right_jaw)
    
    left_forehead = get_pt(54)
    right_forehead = get_pt(284)
    forehead_width = np.linalg.norm(left_forehead - right_forehead)
    
    # Simple heuristic classification
    shape = "Oval"
    if face_length > face_width * 1.3:
        if forehead_width > jaw_width * 1.1:
            shape = "Heart"
        elif face_width > forehead_width and face_width > jaw_width:
            shape = "Diamond"
        else:
            shape = "Oblong"
    else:
        if jaw_width > forehead_width * 0.95:
            shape = "Square"
        else:
            shape = "Round"
            
    # Draw visual feedback
    cv2.line(img, tuple(top.astype(int)), tuple(chin.astype(int)), (0, 255, 0), 2)
    cv2.line(img, tuple(left_cheek.astype(int)), tuple(right_cheek.astype(int)), (255, 0, 0), 2)
    cv2.line(img, tuple(left_jaw.astype(int)), tuple(right_jaw.astype(int)), (0, 0, 255), 2)
    cv2.line(img, tuple(left_forehead.astype(int)), tuple(right_forehead.astype(int)), (255, 255, 0), 2)
    
    base64_img = encode_image(img)
    return {
        "status": "success",
        "message": "Face shape analyzed.",
        "shape": shape,
        "result_b64": f"data:image/jpeg;base64,{base64_img}"
    }

# Serve the static files correctly from the root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
