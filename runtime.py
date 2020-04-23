import numpy as np
from PIL import Image

import tvm
import tvm.relay as relay
from tvm.contrib import graph_runtime

class ModelAgent:

    ctx = tvm.cl(0)
    dtype = 'float32'

    def __init__(self):
        self.graph = open('shufflenet.json').read()
        self.lib = tvm.module.load("shufflenet.tar")
        self.params = bytearray(open("shufflenet.params", "rb").read())
        # Compute with GPU
        self.mod = graph_runtime.create(self.graph, self.lib, self.ctx)
        self.mod.load_params(self.params)

    def preprocess_image(self, image):
        image = image.resize((224, 224))
        image = np.array(image)/np.array([255, 255, 255])
        image -= np.array([0.485, 0.456, 0.406])
        image /= np.array([0.229, 0.224, 0.225])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    def execute(self, inputs):
        inputs = self.preprocess_image(inputs)
        self.mod.set_input("input", tvm.nd.array(inputs.astype(self.dtype)))
        self.mod.run()
        outputs = self.mod.get_output(0)
        return outputs

model_agent = ModelAgent()

image = Image.open("keyboard.jpg").resize((224, 224))

for i in range(1000):
  print("Run")
  outputs = model_agent.execute(image)
  top1 = np.argmax(outputs.asnumpy()[0])
  print(top1)


