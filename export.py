import vtk
import os
import pickle
import utils
import torch
from reconstruction import AE
import numpy as np

from datasets import MeshData
import onnxruntime
#Set Torch device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"



class Encoder(torch.nn.Module):
    def __init__(self, model):
        super(Encoder, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.encoder(x)

class Decoder(torch.nn.Module):
    def __init__(self, model):
        super(Decoder, self).__init__()
        self.model = model

    def forward(self, z):
        return self.model.decoder(z)



if __name__ == "__main__":
    
    dilation = [1, 1, 1, 1]
    seq_length = [9, 9, 9, 9]

    transform_fp = os.path.join( "data", "CoMA", "transform.pkl" )
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        utils.preprocess_spiral(tmp['face'][idx], seq_length[idx], tmp['vertices'][idx], dilation[idx]).to(device)
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [
        utils.to_sparse(down_transform, device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform, device)
        for up_transform in tmp['up_transform']
    ]


    meshdata = MeshData("data/CoMA", "data/CoMA/template/template.obj", split="interpolation", test_exp="bareteeth", normalize=False)
    

    mean = meshdata.mean
    std = meshdata.std




    model = AE(3, [32, 32, 32,64], 16, spiral_indices_list, down_transform_list, up_transform_list, std, mean).to(device)
    checkpoint = torch.load("out/interpolation_exp/checkpoints/checkpoint_040.pt")
    model.load_state_dict( checkpoint["model_state_dict"] )
    model.eval()


    encoder = Encoder(model)
    decoder = Decoder(model)
    
    dummy_input = torch.zeros([1,5023,3])

    z = encoder(dummy_input)
    print(z.shape)

    z = torch.tensor([[0.7733, -3.6339,  4.1113, -1.9286,  0.1162, -0.2889, -0.1077,  2.8485, 5.5649,  0.4998, -2.8854, -3.3035,  2.3326,  5.0924, -1.2362,  1.8607]])
    pred = decoder(z)
    print(pred.shape)
    
    torch.onnx.export(
        encoder, 
        dummy_input, 
        os.path.join( os.path.dirname(__file__), "spiralnet.onnx" ),
        verbose = True,
        do_constant_folding = True,
        opset_version = 12,
        input_names = ["input"],
        output_names = ["output"],        
    )

    torch.onnx.export(
        encoder, 
        dummy_input, 
        os.path.join( os.path.dirname(__file__), "spiralnetEncoder.onnx" ),
        verbose = True,
        do_constant_folding = True,
        opset_version = 12,
        input_names = ["input"],
        output_names = ["output"],        
    )

    torch.onnx.export(
        decoder, 
        z, 
        os.path.join( os.path.dirname(__file__), "spiralnetDecoder.onnx" ),
        verbose = True,
        do_constant_folding = True,
        opset_version = 12,
        input_names = ["input"],
        output_names = ["output"],        
    )

    print("Done")



    #Evaluation
    decoderSession = onnxruntime.InferenceSession(os.path.join( os.path.dirname(__file__), "spiralnetDecoder.onnx" ),)

    onnxPred = decoderSession.run(None, {"input":z.detach().numpy()})[0]
    print((onnxPred - pred.detach().numpy()))