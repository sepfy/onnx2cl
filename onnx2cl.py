import onnx
import numpy as np
import tvm
import tvm.relay as relay
from tvm.contrib import graph_runtime

target = 'opencl'
onnx_model = onnx.load("shufflenet.onnx")
shape_dict = {"input": (1, 3, 224, 224)}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
graph, lib, params = relay.build(mod, target, params=params)

path_lib = 'onnx2cl.tar'
lib.export_library(path_lib)

with open('./shufflenet.json', 'w') as f:
    f.write(graph)

with open('./shufflenet.params', 'wb') as f:
    f.write(relay.save_param_dict(params))
