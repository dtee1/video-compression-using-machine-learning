from flask import Flask
from flask_socketio import SocketIO, emit
import logging
import cv2
import torch
from torchvision import transforms
import numpy
import torch.nn.functional as F
import pytorch_lightning as pl
import time
from skimage.metrics import structural_similarity as ssim


class Autoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 2, 2, 1),
            torch.nn.LeakyReLU(True),
            torch.nn.Conv2d(32, 64, 2, 2, 1),
            # torch.nn.LeakyReLU(True),
            # torch.nn.Conv2d(256, 512, 4, 2, 1),
            # torch.nn.LeakyReLU(True),
        )
        self.decoder = torch.nn.Sequential(
            # torch.nn.Conv2d(512, 256, 4, 2, 1),
            # torch.nn.LeakyReLU(True),
            torch.nn.ConvTranspose2d(64, 32, 2, 2, 1),
            torch.nn.LeakyReLU(True),
            torch.nn.ConvTranspose2d(32, 3, 2, 2, 1),
            torch.nn.LeakyReLU(True),
        )

    def forward(self, image):
        encoded = self.encoder(image)
        decoded = self.decoder(encoded)
        return decoded


app = Flask(__name__)
socketio = SocketIO(app)
camera = cv2.VideoCapture(0)
img_width_before = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
img_width_after = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)


@socketio.on("message")
def message(data):
    transform1 = transforms.ToPILImage()
    frame_count = 0
    start_time = time.time()
    fps = 0
    while True:
        success, frame = camera.read()
        frame_count += 1
        dim = (512, 512)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        elapsed_time = time.time() - start_time

        if not success:
            break
        else:
            ret, buffer_before = cv2.imencode(".jpg", frame)
            frameN = buffer_before.tobytes()

        if elapsed_time > 1:
            # Calculate the frame rate and reset the frame counter and timer
            fps = int(frame_count / elapsed_time)
            frame_count = 0
            start_time = time.time()

        image = tensor_transform(frame)
        compressed_image = model(image)

        img = transform1(compressed_image)
        image = numpy.array(img)
        image_new = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        original_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        reconstructed_gray = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)
        mse = numpy.mean((original_gray - reconstructed_gray) ** 2)
        max_pixel = numpy.max(original_gray)
        psnr = 10 * numpy.log10((max_pixel**2) / mse)
        ssim_score = ssim(
            original_gray,
            reconstructed_gray,
            data_range=original_gray.max() - original_gray.min(),
        )
        data_loss = (mse / (numpy.mean(original_gray) ** 2)) * 100
        ret, buffer_after = cv2.imencode(".jpg", image_new)
        img = buffer_after.tobytes()
        emit(
            "response",
            {
                "from": frameN,
                "size_before": buffer_before.size,
                "FPS": fps,
                "compressed_image": img,
                "size_after": buffer_after.size,
                "psnr": psnr,
                "ssim_score": ssim_score,
                "data_loss": data_loss,
            },
        )


if __name__ == "__main__":
    model = Autoencoder()
    tensor_transform = transforms.ToTensor()
    socketio.run(app, host="0.0.0.0")
