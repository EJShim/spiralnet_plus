import sys, os

import onnxruntime
import numpy as np
import onnx

curPath = os.path.dirname( os.path.abspath(__file__) )


if __name__ == "__main__":
    
    onnx_model = onnx.load_model(os.path.join(curPath, "spiralnet.onnx"))
    onnx.checker.check_model(onnx_model)
    dummy_data = np.zeros([1,5023,3], dtype=np.float32)

    input_tensor = {"input" : dummy_data}
    session = onnxruntime.InferenceSession(os.path.join(curPath, "spiralnet.onnx"))

    pred = session.run(None, input_tensor)

    print(pred[0].shape)



