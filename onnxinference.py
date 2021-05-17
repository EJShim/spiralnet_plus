import sys, os

import onnxruntime
import numpy as np
import vtk
from datasets import MeshData
curPath = os.path.dirname( os.path.abspath(__file__) )


def getInputData(polydata):
    nPoints = polydata.GetNumberOfPoints()

    result = []

    for pid in range(nPoints):
        point = polydata.GetPoint(pid)
        result.append(point)


    tensor = np.array([result])

    return tensor


def getOutputPoly(polydata, pred):
    output = vtk.vtkPolyData()
    output.DeepCopy(polydata)

    for pid, pos in enumerate(pred[0]):
        output.GetPoints().SetPoint(pid, pos[0], pos[1], pos[2])
    

    output.GetPoints().Modified()

    return output


def updatePoly(polydata, pred):
    for pid, pos in enumerate(pred[0]):        
        polydata.GetPoints().SetPoint(pid, pos[0], pos[1], pos[2])
    

    polydata.GetPoints().Modified()


def MakeActor(polydata):
    
    #Visualize
    mapper = vtk.vtkOpenGLPolyDataMapper()
    mapper.SetInputData(polydata)
    # mapper.SetFragmentShaderCode(frag)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

#Initialize Renderer
ren = vtk.vtkRenderer()
ren.GradientBackgroundOn()
ren.SetBackground(135/255, 206/255, 235/255)
ren.SetBackground2(44/255, 125/255, 158/255)
renWin = vtk.vtkRenderWindow()
renWin.SetFullScreen(False)
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

#JPolydata
reader  = vtk.vtkOBJReader()
reader.SetFileName("data/CoMA/template/template.obj")
reader.Update()
polydata = reader.GetOutput()


class LatentInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, encoder, decoder, target, samples, parent=None):


        
        self.AddObserver("LeftButtonPressEvent", self.LeftButtonPressed)
        self.AddObserver("MouseMoveEvent", self.MouseMove)
        self.AddObserver("LeftButtonReleaseEvent", self.LeftButtonReleased)

        
        self.encoder = encoder
        self.decoder = decoder


        #Initialize Plane        
        planeSource = vtk.vtkPlaneSource()
        planeSource.SetCenter(0, 0, 0)
        planeSource.Update()
        planePoly = planeSource.GetOutput()
        planePoly.GetPointData().RemoveArray("Normals")
        self.planeActor = MakeActor(planePoly)
        self.planeActor.GetProperty().SetRepresentationToWireframe()
        self.planeActor.GetProperty().SetColor(1, 0, 0)


    

        ren.AddActor(self.planeActor)


        #ADd Target Actor
        # target = target*std +mean
        self.polydata = getOutputPoly(polydata, target)
        self.actor = MakeActor(self.polydata)
        ren.AddActor(self.actor)




        bounds = self.planeActor.GetBounds()
        self.latentPositions = np.array( [
            [bounds[0], bounds[2], 0],
            [bounds[0], bounds[3], 0],
            [bounds[1], bounds[2], 0],
            [bounds[1], bounds[3], 0]
        ])
        

        self.latentSize = 16
        self.outputLatents = []
        #Add Sample Actor
        for idx, sample in enumerate(samples):

            #ADd Target Actor
            z = self.encoder.run(None,{"input":sample})[0]
            self.latentSize = z.shape[1]
            self.outputLatents.append(z[0])

            print(z)
            
            outpoly = getOutputPoly(polydata, sample)
            actor = MakeActor(outpoly)
            actor.SetPosition(self.latentPositions[idx])
            ren.AddActor(actor)

        self.pickedPosition = -1
            

    def LeftButtonPressed(self, obj, ev):
        
        self.OnLeftButtonDown()

        pos = obj.GetInteractor().GetEventPosition()

        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        picker.AddPickList(self.planeActor)
        picker.Pick(pos[0], pos[1], 0, ren)

    

        position = picker.GetPickPosition()
        
        if picker.GetActor() == self.planeActor:
            self.pickedPosition = position

            

    def MouseMove(self, obj, ev):
        if self.pickedPosition == -1:
            self.OnMouseMove()
            return

        pos = obj.GetInteractor().GetEventPosition()

        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        picker.AddPickList(self.planeActor)
        picker.Pick(pos[0], pos[1], 0, ren)
        
        
        position = picker.GetPickPosition()
        targetPos = np.array([position[0], position[1], 0])


        if targetPos[0] < -0.5 : targetPos[0] = -0.5
        elif targetPos[0] > 0.5 : targetPos[0] = 0.5
        if targetPos[1] < -0.5 : targetPos[1] = -0.5
        elif targetPos[1] > 0.5 : targetPos[1] = 0.5

        distances = []
        for sample in self.latentPositions:            
            distances.append(np.linalg.norm(targetPos-sample))
                

        weights = np.array(distances)

        weights[weights > 1] = 1
        weights = 1 - weights
        
        calculatedLatent = np.zeros(self.latentSize, dtype=np.float32)

        for idx, weight in enumerate(weights):
            calculatedLatent += self.outputLatents[idx] * weight


        calculatedLatent = np.array([0.7733, -3.6339,  4.1113, -1.9286,  0.1162, -0.2889, -0.1077,  2.8485,
         5.5649,  0.4998, -2.8854, -3.3035,  2.3326,  5.0924, -1.2362,  1.8607])
        print(calculatedLatent)
        
         
        out = self.decoder.run(None, {"input":[calculatedLatent]})[0]
        print(out)


        
        updatePoly(self.polydata, out)
        renWin.Render()


    def LeftButtonReleased(self, obj, ev):

        self.pickedPosition = -1
        self.OnLeftButtonUp()
if __name__ == "__main__":
        
    dummy_data = np.zeros([1,5023,3], dtype=np.float32)

    
    encoderSession = onnxruntime.InferenceSession(os.path.join(curPath, "spiralnetEncoder.onnx"))
    decoderSession = onnxruntime.InferenceSession(os.path.join(curPath, "spiralnetDecoder.onnx"))
    
    
    input_tensor = {"input" : dummy_data}
    z = encoderSession.run(None, input_tensor)    

    z_tensor = {"input": z[0]}
    pred = decoderSession.run(None, z_tensor)    



    meshdata = MeshData("data/CoMA", "data/CoMA/template/template.obj", split="interpolation", test_exp="bareteeth", normalize=False)

    x = meshdata.train_dataset[10].x.unsqueeze(0).numpy()

    samples = [
        meshdata.train_dataset[10].x.unsqueeze(0).numpy(),
        meshdata.train_dataset[5768].x.unsqueeze(0).numpy(),
        meshdata.train_dataset[8654].x.unsqueeze(0).numpy(),
        meshdata.train_dataset[2054].x.unsqueeze(0).numpy()
    ]


    interactorStyle = LatentInteractorStyle(encoderSession, decoderSession, x, samples)



    iren.SetInteractorStyle(interactorStyle)
    
    renWin.Render()
    iren.Initialize()
    iren.Start()
