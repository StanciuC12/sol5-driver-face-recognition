import cv2
import torch
from typing import List, Union
import numpy as np
from PIL import Image
import time
from ultralytics import YOLO
from ultralytics.yolo.utils import ROOT, SETTINGS

MODEL = 'yolov8n-face.pt'
CFG = 'yolov8n.yaml'
SOURCE = r"C:\Users\Crispy\Downloads\14_Traffic_Traffic_14_179_TEST.jpg" #ROOT / 'assets/bus.jpg'

class YOLOFaceDetector:

    def __init__(self, model_location='yolov8n-face.pt'):

        self.model_location = model_location
        self.model = None
        if model_location:
            self.set_model(model_location)

    def set_model(self, model_location):

        self.model_location = model_location
        self.model = YOLO(self.model_location)

    def open_images(self, img_addresses: list):

        images = []
        for adr in img_addresses:
            try:
                images.append(Image.open(adr))
            except Exception as e:
                print('Image could not be read', e, 'adr=', adr)

        return images

    def detect_face_imgs_batch(self, imgs: List[Union[Image.Image, np.ndarray]], save_results=False, return_faces=False):

        output = self.model(source=imgs, save=save_results, save_txt=save_results)  # batch
        print('Run for ', len(imgs), 'images')

        processed_out = []
        faces = []
        for idx, img_out in enumerate(output):
            data = img_out.cpu().boxes.data.numpy()[:, 0:5]
            processed_out.append(data)
            if return_faces:
                faces.append([np.array(imgs[idx])[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])] for coord in data])

        if return_faces:
            return processed_out, faces

        else:
            return processed_out

    def detect_face_imgs_stream(self, imgs: List[Union[Image.Image, np.ndarray]], save_results=False, return_faces=False):

        output = self.model(source=imgs, save=save_results, save_txt=save_results, stream=True)  # batch

        for idx, out in enumerate(output):

            data = out.cpu().boxes.data.numpy()[:, 0:5]
            if return_faces:
                faces = [np.array(imgs[idx])[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])] for coord in data]
                yield data, faces
            else:
                yield data


if __name__ == "__main__":

    # tests
    SOURCE = r"C:\Users\Crispy\Downloads\14_Traffic_Traffic_14_179_TEST.jpg"
    m = YOLOFaceDetector('yolov8n-face.pt')
    imgs = m.open_images([SOURCE, SOURCE])

    # out, faces = m.detect_face_imgs_batch(imgs, return_faces=True)
    #
    # for idx, f in enumerate(faces):
    #     # Convert NumPy array to PIL image
    #     pil_image = Image.fromarray(f[0])
    #
    #     # Save the image
    #     pil_image.save(f"runs/output{idx}.png")

    # output = m.detect_face_imgs_stream(imgs, return_faces=True)
    # for out, faces in output:
    #     # Convert NumPy array to PIL image
    #     pil_image = Image.fromarray(faces[0])
    #
    #     # Save the image
    #     pil_image.save(f"runs/output{5}.png")

    video = r"F:\videoplayback.mp4"
    cap = cv2.VideoCapture(video)
    o = []
    a = time.time()
    i = 0
    while True:
        if i % 60 == 0:
            print(i)
        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the stream is not working

        # Convert the BGR frame (OpenCV format) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to a PIL image
        pil_image = Image.fromarray(rgb_frame)
        #output = m.detect_face_imgs_stream([frame], return_faces=False)
        output = m.detect_face_imgs_stream([frame], return_faces=False)

        # for out in output:
        #     try:
        #         print(out)
        #     except:
        #         pass
        i += 1

    print(time.time() - a, 'seconds')