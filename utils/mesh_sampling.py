from os import close
from numpy.lib.polynomial import poly
import torch
import vtk
import numpy as np
import scipy.sparse as sp
import argparse
import math
import heapq
import pickle
import os
#Set Torch device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


class Mesh:
    def __init__(self, v=None, f=None):
        self.v = v
        self.f = f
        self.polydata = None
        
        if self.v is not None and self.f is not None:
            self.polydata = vtk.vtkPolyData()

            points = vtk.vtkPoints()
            for vertex in self.v:
                points.InsertNextPoint(vertex)
            self.polydata.SetPoints(points)

            cellArray = vtk.vtkCellArray()
            for face in self.f:
                cell = vtk.vtkTriangle()
                cell.GetPointIds().SetId(0, face[0])
                cell.GetPointIds().SetId(1, face[1])
                cell.GetPointIds().SetId(2, face[2])
                cellArray.InsertNextCell(cell)
            self.polydata.SetPolys(cellArray)            
            

    def SetPolyData(self, polydata):
        # print(polydata.GetNumberOfPoints())

        nPoints = polydata.GetNumberOfPoints()
        nFaces = polydata.GetNumberOfCells()

        vertices = []
        for vid in range(nPoints):
            v = polydata.GetPoint(vid)
            vertices.append(v)
        self.v = np.array(vertices)
        
        faces = []
        for fid in range(nFaces):
            cell = polydata.GetCell(fid)
            faces.append( [cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)] )
            # print(cell.GetPointId(0))

        self.f = np.array(faces)   

    


    def is_on(self, a, b, c):
        "Return true iff point c intersects the line segment from a to b."
        # (or the degenerate case that all 3 points are coincident)


        def collinear(a, b, c):
            "Return true iff a, b, and c all lie on the same line."
            return (b[0] - a[0]) * (c[1] - a[1]) == (c[0] - a[0]) * (b[1] - a[1])

        def within(p, q, r):
            "Return true iff q is between p and r (inclusive)."
            return p <= q <= r or r <= q <= p


        return (collinear(a, b, c)
                and (within(a[0], c[0], b[0]) if a[0] != b[0] else 
                    within(a[1], c[1], b[1])))


    def nearest(self, v_samples):

        nearest_faces = []
        "nearest_part tells you whether the closest point in triangle abc is in the interior (0), on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)"

        nearest_parts = [] 
        nearest_vertices = []


        cellLocator = vtk.vtkCellLocator()
        cellLocator.SetDataSet(self.polydata)
        cellLocator.BuildLocator()


        for vertex in v_samples:

            closestPoint = [0, 0, 0]
            cellId = vtk.reference(0)            

            
            cellLocator.FindClosestPoint(vertex, closestPoint, cellId, vtk.mutable(-1), vtk.mutable(-1.0))


            #Check cloesstpoint In the triangle (0), on the Edge (1), On the Vertex(2)
            part = 0
            triangle = self.polydata.GetCell(cellId)

            
            #TODO  : Add Part checker!!!
            a = np.array(self.polydata.GetPoint(triangle.GetPointId(0)))
            b = np.array(self.polydata.GetPoint(triangle.GetPointId(1)))
            c = np.array(self.polydata.GetPoint(triangle.GetPointId(2)))
            target = np.array(closestPoint)
            if np.array_equal(  a ,target ):
                part = 4
            elif np.array_equal( b  , target ):
                part = 5
            elif np.array_equal( c  , target ):
                part = 6
            elif self.is_on(a, b, target):
                part = 1
            elif self.is_on(b, c, target):
                part = 2
            elif self.is_on(c, a, target):
                part = 3

            #TODO : Debug This Part Later        
            weights = [0, 0, 0]
            inside = triangle.EvaluatePosition(closestPoint, [0, 0, 0], vtk.mutable(0), [0,0,0], vtk.mutable(0.0), weights)
            # if weights[0] == 1.0: part = 4
            # elif weights[1] == 1.0 : part = 5
            # elif weights[2] == 1.0 : part = 6

            if part == 1:
                print(inside, "\t", weights)
                

            nearest_faces.append(cellId)
            nearest_parts.append(part) #???
            nearest_vertices.append(closestPoint)

        
        # print(v_samples, v_samples.shape, self.polydata.GetNumberOfPoints())

        nearest_faces = np.array(nearest_faces)
        nearest_parts = np.array(nearest_parts)
        nearest_vertices = np.array(nearest_vertices)
        return nearest_faces, nearest_parts, nearest_vertices



def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))



def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij),
                        shape=(len(verts_left), num_original_verts))

    return (new_faces, mtx)


def setup_deformation_transfer(source, target, use_normals=False):
    rows = np.zeros(3 * target.v.shape[0])
    cols = np.zeros(3 * target.v.shape[0])
    coeffs_v = np.zeros(3 * target.v.shape[0])
    coeffs_n = np.zeros(3 * target.v.shape[0])

    # nearest_faces, nearest_parts, nearest_vertices = source.compute_aabb_tree().nearest(target.v, True)
    nearest_faces, nearest_parts, nearest_vertices = source.nearest(target.v) # Not Sure For this

    nearest_faces = nearest_faces.ravel().astype(np.int64)
    nearest_parts = nearest_parts.ravel().astype(np.int64) #Wjat is "Parts"?
    nearest_vertices = nearest_vertices.ravel()

    for i in range(target.v.shape[0]):
        # Closest triangle index
        f_id = nearest_faces[i]
        # Closest triangle vertex ids
        nearest_f = source.f[f_id]

        # Closest surface point
        nearest_v = nearest_vertices[3 * i:3 * i + 3]
        # Distance vector to the closest surface point
        dist_vec = target.v[i] - nearest_v

        rows[3 * i:3 * i + 3] = i * np.ones(3)
        cols[3 * i:3 * i + 3] = nearest_f

        n_id = nearest_parts[i]
        if n_id == 0:
            # Closest surface point in triangle
            A = np.vstack((source.v[nearest_f])).T
            coeffs_v[3 * i:3 * i + 3] = np.linalg.lstsq(A, nearest_v, rcond=-1)[0]
        elif n_id > 0 and n_id <= 3:
            # Closest surface point on edge
            A = np.vstack((source.v[nearest_f[n_id - 1]], source.v[nearest_f[n_id % 3]])).T
            tmp_coeffs = np.linalg.lstsq(A, target.v[i], rcond=-1)[0]
            coeffs_v[3 * i + n_id - 1] = tmp_coeffs[0]
            coeffs_v[3 * i + n_id % 3] = tmp_coeffs[1]
        else:
            # Closest surface point a vertex
            coeffs_v[3 * i + n_id - 4] = 1.0

    matrix = sp.csc_matrix((coeffs_v, (rows, cols)), shape=(target.v.shape[0], source.v.shape[0]))
    return matrix


def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v), len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:, i]
        JS = mesh_f[:, (i + 1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.ravel()), row(JS.ravel())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv



def get_vertices_per_edge(mesh_v, mesh_f):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()"""

    vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:, 0] < result[:, 1]]  # for uniqueness

    return result

def vertex_quadrics(mesh):
    """Computes a quadric for each vertex in the Mesh.
    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    v_quadrics = np.zeros((len(mesh.v), 4, 4,))

    # For each face...
    for f_idx in range(len(mesh.f)):

        # Compute normalized plane equation for that face
        vert_idxs = mesh.f[f_idx]
        verts = np.hstack((mesh.v[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(verts)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)

    return v_quadrics

def qslim_decimator_transformer(mesh, factor=None, n_verts_desired=None):
    """Return a simplified version of this mesh.
    A Qslim-style approach is used here.
    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
    """

    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')

    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh.v) * factor)

    Qv = vertex_quadrics(mesh)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    # from psbody.mesh.topology.connectivity import get_vertices_per_edge
    vert_adj = get_vertices_per_edge(mesh.v, mesh.f)
    # vert_adj = sp.lil_matrix((len(mesh.v), len(mesh.v)))
    # for f_idx in range(len(mesh.f)):
    #     vert_adj[mesh.f[f_idx], mesh.f[f_idx]] = 1

    vert_adj = sp.csc_matrix((vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])), shape=(len(mesh.v), len(mesh.v)))
    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum}
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c)))

    # decimate
    collapse_list = []
    nverts_total = len(mesh.v)
    faces = mesh.f.copy()
    while nverts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
            which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = (len(np.unique(faces.flatten())))

    new_faces, mtx = _get_sparse_transform(faces, len(mesh.v))
    return new_faces, mtx




def generate_transform_matrices(mesh, factors):
    """Generates len(factors) meshes, each of them is scaled by factors[i] and
       computes the transformations between them.

    Returns:
       M: a set of meshes downsampled from mesh by a factor specified in factors.
       A: Adjacency matrix for each of the meshes
       D: csc_matrix Downsampling transforms between each of the meshes
       U: Upsampling transforms between each of the meshes
       F: a list of faces
    """

    factors = map(lambda x: 1.0 / x, factors)
    M, A, D, U, F = [], [], [], [], []
    F.append(mesh.f)  # F[0]
    A.append(get_vert_connectivity(mesh.v, mesh.f).astype('float32'))  # A[0]
    M.append(mesh)  # M[0]

    for factor in factors:
        print(factor)
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        D.append(ds_D.astype('float32'))
        new_mesh_v = ds_D.dot(M[-1].v)
        new_mesh = Mesh(v=new_mesh_v, f=ds_f)
        F.append(new_mesh.f)
        M.append(new_mesh)
        A.append(get_vert_connectivity(new_mesh.v, new_mesh.f).astype('float32'))
        U.append(setup_deformation_transfer(M[-1], M[-2]).astype('float32'))

    return M, A, D, U, F



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description = 'Generate Transform Matrix')
    parser.add_argument('--template', type = str,  default="template/toothTemplate.vtp")
    parser.add_argument("--output", type=str, default = "toothTransform")
    # parser.add_argument('--output', type=str, default="//192.168.0.113/Imagoworks/AITeam/EJSHIM/dcmtosinglevtiforcau")
    args = parser.parse_args()


    output = os.path.join( os.path.dirname(args.template) , args.output+".pkl")
    print("Save Output To ", output)
    os.makedirs(os.path.dirname(output), exist_ok=True)


    #Read Template
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(args.template)
    reader.Update()

    

    templatePoly = reader.GetOutput()

    mesh = Mesh()
    mesh.SetPolyData(templatePoly)



    print("Generating Transform Matrices...")
    ds_factors = [4, 4, 4, 4, 4, 4]
    _, A, D, U, F = generate_transform_matrices(mesh, ds_factors)
    
    
    
    tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}
    with open(output, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')