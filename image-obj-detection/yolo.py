# USAGE
# python yolo.py --image images/baggage_claim.jpg
# import the necessary packages
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
from ultralytics import YOLO
from PIL import Image

app = FastAPI()

class YOLODetector:
    def __init__(self, model_name):
        self.model = YOLO(model_name)

    def process_image_array(self, source):
        # Run inference on the source
        results = self.model(source)  # list of Results objects
        return results

@app.post("/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    if image is None:
        return {"error": "Invalid image"}
    detector = YOLODetector("yolo11n.pt")   
    result_image = detector.process_image_array(image)

    # Visualize the results
    for i, r in enumerate(result_image):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Save im_rgb to a BytesIO buffer in JPEG format
    buf = io.BytesIO()
    im_rgb.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")