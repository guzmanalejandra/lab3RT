from gl import Raytracer, V3
from texture import *
from figures import *
from lights import *


width = 512
height = 512


# Materiales
brick = Material(diffuse = (0.8, 0.3, 0.3), spec = 16)
stone = Material(diffuse = (0.4, 0.4, 0.4), spec = 8)
grass = Material(diffuse = (0.3, 1.0, 0.3), spec = 64)
mirror = Material(diffuse = (0.9, 0.9, 0.9), spec = 64, matType = REFLECTIVE)
glass = Material(diffuse = (0.9, 0.9, 0.9), spec = 64, ior = 1.5, matType = TRANSPARENT)


mat1 = Material(diffuse = (0.9, 0.9, 0.2), spec = 64)
mat2 = Material(diffuse = (0.9, 0.2, 0.2), spec = 64)
mat3 = Material(diffuse = (0.2, 0.2, 0.9), spec = 20)


rtx = Raytracer(width, height)
rtx.envMap = Texture("parkingLot.bmp")


rtx.lights.append( AmbientLight(intensity = 0.8 ))
rtx.lights.append( DirectionalLight(direction = (0,0,-1), intensity = 0.5 ))
rtx.lights.append( PointLight(point = (1,-1,0) ))


rtx.scene.append( Triangle(V3(-3,0,-7), V3(3,0,-7), V3(0,3,-7), material = glass) )
rtx.scene.append( Triangle(V3(-3,0,-7), V3(3,0,-7), V3(0,-3,-7), material = mat3) )
rtx.scene.append( Triangle(V3(3,0,-7), V3(0,3,-7), V3(4,3,-10), material = mirror) )


rtx.glRender()

rtx.glFinish("output.bmp")