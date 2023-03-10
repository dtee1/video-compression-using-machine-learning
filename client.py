import socketio
from flask import Flask, render_template, Response, jsonify


sio = socketio.Client()

app = Flask(__name__)


@sio.on("response")
def response(frame):
    frames.append(frame["compressed_image"])
    image_size.append(frame["size_before"])
    fps_host.append(frame["FPS"])
    image_size_a.append(frame["size_after"])
    frames_b.append(frame["from"])


def generate_frame(type_):
    if type_ == "frame_after":
        while True:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frames[len(frames) - 1] + b"\r\n"
            )
    else:
        while True:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frames_b[len(frames_b) - 1]
                + b"\r\n"
            )


@app.route("/video_after")
def video_after():
    return Response(
        generate_frame("frame_after"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_before")
def video_before():
    return Response(
        generate_frame("frame"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/size_b")
def size_b():
    return jsonify(image_size[len(image_size) - 1] / 1000)


@app.route("/fps")
def fps():
    return jsonify(fps_host[len(fps_host) - 1])


@app.route("/size_a")
def size_a():
    return jsonify(image_size_a[len(image_size_a) - 1] / 1000)


@app.route("/com_fac")
def com_fac():
    result = image_size[len(image_size) - 1] / image_size_a[len(image_size_a) - 1]
    return jsonify(int(result))


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    frames = [b"None"]
    image_size = []
    fps_host = []
    image_size_a = []
    frames_b = [b"None"]

    sio.connect("http://10.0.0.100:5000")
    sio.emit("message", {"from": "client"})
    app.run(port="8080")
