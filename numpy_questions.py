import numpy as numpy

vector1 = numpy.arange(8, 49)
print(vector1)

vector2 = numpy.arange(0, 9).reshape((3,3))
print(vector2)

vector3 = numpy.arange(100).reshape((10,10))
print(vector3)

vector4 = numpy.random.rand(10, 10)
print(vector4)

vec4_min = vector4.min()
vec4_max = vector4.max()
print(vec4_min)
print(vec4_max)

vector5 = numpy.arange(18).reshape((6,3))
vector6 = numpy.arange(6).reshape((3,2))
print(vector5)
print(vector6)  

vector7 = numpy.matmul(vector5, vector6)
print(vector7)