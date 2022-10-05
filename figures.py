import numpy as np
import Math
from collections import namedtuple
V3 = namedtuple('Point3', ['x', 'y', 'z'])

WHITE = (1,1,1)
BLACK = (0,0,0)

OPAQUE = 0
REFLECTIVE = 1
TRANSPARENT = 2
class Material(object):
    def __init__(self, diffuse = WHITE, spec = 1, ior = 1, texture = None, matType = OPAQUE):
        self.diffuse = diffuse
        self.spec = spec
        self.ior = ior
        self.texture = texture
        self.matType = matType


class Intersect(object):
    def __init__(self, distance, point, normal, texCoords, sceneObj):
        self.distance = distance
        self.point = point
        self.normal = normal
        self.texCoords = texCoords
        self.sceneObj = sceneObj

class Sphere(object):
    def __init__(self, center, radius, material = Material()):
        self.center = center
        self.radius = radius
        self.material = material

    def ray_intersect(self, orig, dir):

        L = np.subtract(self.center, orig)
        l = np.linalg.norm(L)

        tca = np.dot(L, dir)

        d = (l**2 - tca**2)
        if d > self.radius ** 2:
            return None

        thc = (self.radius**2 - d) ** 0.5
        t0 = tca - thc
        t1 = tca + thc

        if t0 < 0:
            t0 = t1

        if t0 < 0:
            return None

        # P = O + t * D
        hit = np.add(orig, t0 * np.array(dir) )
        normal = np.subtract( hit, self.center )
        normal = normal / np.linalg.norm(normal) #la normalizo

        u = 1 - ((np.arctan2(normal[2], normal[0] ) / (2 * np.pi)) + 0.5)
        v = np.arccos(-normal[1]) / np.pi

        uvs = (u,v)

        return Intersect( distance = t0,
                          point = hit,
                          normal = normal,
                          texCoords = uvs,
                          sceneObj = self)

class Triangle(object):
    def __init__(self, v0, v1, v2, material = Material()):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.material = material

    def ray_intersect(self, orig, dir):
        # Moller-Trumbore ray/triangle intersection algorithm
        edge1 = np.subtract(self.v1, self.v0)
        edge2 = np.subtract(self.v2, self.v0)

        pvec = np.cross(dir, edge2)

        det = np.dot(edge1, pvec)
        if det > -0.000001 and det < 0.000001:
            return None
        inv_det = 1 / det

        tvec = np.subtract(orig, self.v0)

        u = np.dot(tvec, pvec) * inv_det
        if u < 0 or u > 1:
            return None

        qvec = np.cross(tvec, edge1)

        v = np.dot(dir, qvec) * inv_det
        if v < 0 or u + v > 1:
            return None

        t = np.dot(edge2, qvec) * inv_det

        if t > 0.000001:
            hit = np.add(orig, t * np.array(dir) )
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal) #la normalizo
            return Intersect( distance = t,
                              point = hit,
                              normal = normal,
                              texCoords = (u, v),
                              sceneObj = self)

        else:
            return None


class Plane(object):
    def __init__(self, position, normal, material = Material()):
        self.position = position
        self.normal = normal / np.linalg.norm(normal)
        self.material = material

    def ray_intersect(self, orig, dir):
        #t = (( planePos - origRayo) dot planeNormal) / (dirRayo dot planeNormal)
        denom = np.dot(dir, self.normal)

        if abs(denom) > 0.0001:
            num = np.dot(np.subtract(self.position, orig), self.normal)
            t = num / denom
            if t > 0:
                # P = O + t * D
                hit = np.add(orig, t * np.array(dir))

                return Intersect(distance = t,
                                 point = hit,
                                 normal = self.normal,
                                 texCoords = None,
                                 sceneObj = self)

        return None

class AABB(object):
    # Axis Aligned Bounding Box
    def __init__(self, position, size, material = Material()):
        self.position = position
        self.size = size
        self.material = material
        self.planes = []

        self.boundsMin = [0,0,0]
        self.boundsMax = [0,0,0]

        halfSizeX = size[0] / 2
        halfSizeY = size[1] / 2
        halfSizeZ = size[2] / 2

        #Sides
        self.planes.append(Plane( np.add(position, V3(halfSizeX,0,0)), V3(1,0,0), material))
        self.planes.append(Plane( np.add(position, V3(-halfSizeX,0,0)), V3(-1,0,0), material))

        # Up and down
        self.planes.append(Plane( np.add(position, V3(0,halfSizeY,0)), V3(0,1,0), material))
        self.planes.append(Plane( np.add(position, V3(0,-halfSizeY,0)), V3(0,-1,0), material))

        # Front and Back
        self.planes.append(Plane( np.add(position, V3(0,0,halfSizeZ)), V3(0,0,1), material))
        self.planes.append(Plane( np.add(position, V3(0,0,-halfSizeZ)), V3(0,0,-1), material))

        #Bounds
        epsilon = 0.001
        for i in range(3):
            self.boundsMin[i] = self.position[i] - (epsilon + self.size[i]/2)
            self.boundsMax[i] = self.position[i] + (epsilon + self.size[i]/2)


    def ray_intersect(self, orig, dir):
        intersect = None
        t = float('inf')

        uvs = None

        for plane in self.planes:
            planeInter = plane.ray_intersect(orig, dir)
            if planeInter is not None:
                # Si estoy dentro de los bounds
                if planeInter.point[0] >= self.boundsMin[0] and planeInter.point[0] <= self.boundsMax[0]:
                    if planeInter.point[1] >= self.boundsMin[1] and planeInter.point[1] <= self.boundsMax[1]:
                        if planeInter.point[2] >= self.boundsMin[2] and planeInter.point[2] <= self.boundsMax[2]:
                            #Si soy el plano mas cercano
                            if planeInter.distance < t:
                                t = planeInter.distance
                                intersect = planeInter

                                u, v = 0, 0

                                if abs(plane.normal[0]) > 0:
                                    # mapear uvs para eje X, uso coordenadas en Y y Z.
                                    u = (planeInter.point[1] - self.boundsMin[1]) / (self.boundsMax[1] - self.boundsMin[1])
                                    v = (planeInter.point[2] - self.boundsMin[2]) / (self.boundsMax[2] - self.boundsMin[2])

                                elif abs(plane.normal[1]) > 0:
                                    # mapear uvs para eje Y, uso coordenadas en X y Z.
                                    u = (planeInter.point[0] - self.boundsMin[0]) / (self.boundsMax[0] - self.boundsMin[0])
                                    v = (planeInter.point[2] - self.boundsMin[2]) / (self.boundsMax[2] - self.boundsMin[2])

                                elif abs(plane.normal[2]) > 0:
                                    # mapear uvs para eje Z, uso coordenadas en X y Y.
                                    u = (planeInter.point[0] - self.boundsMin[0]) / (self.boundsMax[0] - self.boundsMin[0])
                                    v = (planeInter.point[1] - self.boundsMin[1]) / (self.boundsMax[1] - self.boundsMin[1])

                                uvs = (u,v)


        if intersect is None:
            return None

        return Intersect(distance = intersect.distance,
                         point = intersect.point,
                         normal = intersect.normal,
                         texCoords = uvs,
                         sceneObj = self)



        