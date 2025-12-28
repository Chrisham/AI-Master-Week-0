import numpy as np

list1 = [1,2,3,4]
numpy_array = np.array(list1)
numpy_array

print(numpy_array)


numpy_array2 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

# dimension of numpy array
dim = numpy_array2.ndim
print('dimension = ',dim)
#shape of numpy array
shape = numpy_array2.shape
print('shape = ',shape)
#size of numpy array
size = numpy_array2.size
print('size = ',size)
#data type of numpy array
data_type = numpy_array2.dtype
print('data type = ',data_type)


#NOTE: The array you want to reshape to must have the same number of elements as the array that is being reshaped
array1 = np.arange(2,14,2) #creates an array within the range (start,end,jump)
print('original array = ',array1)
reshaped_array = array1.reshape(3,2) #(rows,columns)
print('reshaped array =',reshaped_array)

data = np.array([1, 2])
ones = np.ones(2, dtype=int) #creates an array of ones of size 2
#addition
add = data + ones
#subtraction
sub = data - ones
#multiplication
mul = data * ones
#division
div = data / ones
print('addition = ',add)
print('subtraction = ',sub)
print('multiplication = ',mul)
print('division = ',div)

#add up the elements in an N-dimensional array
sum = data.sum()
print('sum using sum function = ',sum)

#sum over rows or columns in 2D array
array1 = np.array([[1,2,3],[4,5,6]])
#row sum
row_sum = array1.sum(axis=0)
#column sum
col_sum = array1.sum(axis=1)
print('row sum = ',row_sum)
print('column sum = ',col_sum)

arr = np.arange(6).reshape((2, 3))
#transpose of an array
arr_T = arr.T
print('array = ',arr)
print('transpose of array = ',arr_T)

x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
flat_arr = x.flatten()
print('flattened array = ',flat_arr)


# To create array of ones
ones = np.ones((2,3))
# To create array of zeros
zero = np.zeros((2,3))
# To create array of random numbers
random_array = np.random.rand(2, 3)

print('ones array = ',ones)
print('zeros array = ',zero)
print('random numbers array = ',random_array)

array1 = [[1, 0], [0, 1]]
array2 = [[4, 1], [2, 2]]
np.dot(array1, array2)
print('dot product = ',np.dot(array1, array2))

array1 = np.array([[1, 0],
              [0, 1]])
array2 = np.array([[4, 1],
              [2, 2]])
np.matmul(array1, array2)
print('dot product = ',np.matmul(array1, array2))

array1 = np.random.rand(2, 3, 2)  # Shape (2, 3, 2)
array2 = np.random.rand(2, 2, 3)  # Shape (2, 2, 3)
matmul_result = np.matmul(array1, array2)  # Shape (2, 3, 3)
dot_result = np.dot(array1,array2)
print('np.matmul result = ',matmul_result)
print('np.dot result = ',dot_result)
print(matmul_result.shape)
print(dot_result.shape)

