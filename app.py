from flask import Flask, Response, request
from flask_cors import CORS

from serve.worker.deblur_worker import DeBlurWorker
from serve.worker.lama_worker import LaMaWorker

app = Flask(__name__)
CORS(app, supports_credentials=True)

lama_worker = LaMaWorker()
deblur_worker = DeBlurWorker()

@app.route('/inpaint', methods=['POST'])
def inpaint():
    if request.method == 'POST':
        files = request.files
        input_image = files['origin'].read()
        input_mask = files['mask'].read()
        inpaint_image = lama_worker.process(input_image, input_mask)
        return Response(inpaint_image[0], mimetype="image/jpeg")

@app.route('/deblur', methods=['POST'])
def deblur():
    if request.method == 'POST':
        file = request.files['origin']
        input_image = file.read()
        deblur_image = deblur_worker.process(input_image)
        return Response(deblur_image, mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(port=5003)