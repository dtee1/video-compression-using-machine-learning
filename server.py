from flask import Flask
from flask_socketio import SocketIO, emit
import logging
import cv2
import torch
from torchvision import transforms
import numpy
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image


class Autoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 512, 2, 2, 1),
            # torch.nn.LeakyReLU(True),
            # torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            # torch.nn.LeakyReLU(True),
            # torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            # torch.nn.LeakyReLU(True),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(512, 3, 2, 2, 1),
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
    # specify loss function
    criterion = torch.nn.BCELoss()

    # specify loss function
    optimizer = torch.torch.optim.Adam(model.parameters(), lr=0.001)
    while True:
        success, frame = camera.read()
        dim = (512, 512)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        fps = camera.get(cv2.CAP_PROP_FPS)
        if not success:
            break
        else:
            ret, buffer_before = cv2.imencode(".jpg", frame)
            frameN = buffer_before.tobytes()
        optimizer.zero_grad()
        image = tensor_transform(frame)
        compressed_image = model(image)
        loss = criterion(compressed_image, image)
        loss.backward()
        img = transform1(compressed_image)
        image = numpy.array(img)
        image_new = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
            },
        )


if __name__ == "__main__":
    model = Autoencoder()
    tensor_transform = transforms.ToTensor()
    socketio.run(app, host="0.0.0.0")
