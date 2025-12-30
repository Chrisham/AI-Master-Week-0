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

# 4. Matrix Multiplication
vector5 = numpy.arange(18).reshape((6,3))
vector6 = numpy.arange(6).reshape((3,2))
print(vector5)
print(vector6)  

vector7 = numpy.matmul(vector5, vector6)
print(vector7)



# 5. Subtract the mean of each row of a matrix
print("5. Subtract the mean of each row of a matrix")

vector8 = numpy.random.rand(3, 6)
print(f"vector8 = {vector8}")

vector8_mean = vector8.mean(axis=1)
print(f"vector8_mean = {vector8_mean}")

vector8_mean_col = numpy.reshape(vector8_mean, (3,1))
print(f"vector8_mean_col = {vector8_mean_col}")


vector8_sub_mean1 = vector8 - vector8_mean_col
print(f"vector8_sub_mean1 = {vector8_sub_mean1}")

vector8_sub_mean2 = vector8 - vector8_mean[:, numpy.newaxis]
print(f"vector8_sub_mean2 = {vector8_sub_mean2}")


# 6. Compute the mean, median, standard deviation of a numpy array
print("6. Compute the mean, median, standard deviation of a numpy array")

vector9 = numpy.random.rand(3, 6)
print(f"vector9 = {vector9}")

vector9_mean = vector9.mean()
print(f"vector9_mean = {vector9_mean}")

vector9_mean2 = numpy.mean(vector9)
print(f"vector9_mean2 = {vector9_mean2}")

vector9_median = numpy.median(vector9)     
print(f"vector9_median = {vector9_median}")

vector9_std = numpy.std(vector9)
print(f"vector9_std = {vector9_std}")



# 7. You are given a 3x3 matrix:
print("7. You are given a 3x3 matrix:")

matrix = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)

matrix_flattened = matrix.flatten()
print(matrix_flattened)

matrix_array = numpy.array(matrix_flattened)
print(matrix_array)

# 8. Given the following matrix:
print("8. Given the following matrix:")
# Compute:
# The sum of all elements in the matrix.
# The sum of each row (i.e., along axis 1).
# The sum of each column (i.e., along axis 0).
matrix_2 = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix_2)

sum_all = numpy.sum(matrix_2)
print(f"sum_all = {sum_all}")

sum_row = numpy.sum(matrix_2, axis=1)
print(f"sum_row = {sum_row}")

sum_col = numpy.sum(matrix_2, axis=0)
print(f"sum_col = {sum_col}")


# 9. Write a function to compute the Root Mean Squared Error (RMSE), which is simply the square root of the Mean Squared Error.
print("9. Root Mean Squared Error (RMSE) is the square root of the Mean Squared Error.")

y_pred = numpy.array([3, 5, 2.5, 7])
y_true = numpy.array([3, 5, 3, 7])

def root_mean_squared_error(y_pred, y_true):
    squared_error_sum = 0
    for i in range(len(y_pred)):
        squared_error = (y_pred[i] - y_true[i]) ** 2
        squared_error_sum += squared_error
    
    mse = squared_error_sum / len(y_pred)
    rmse = numpy.sqrt(mse)
    return rmse

rmse = root_mean_squared_error(y_pred, y_true)
print(f"RMSE = {rmse}")


# 10. Given a vector of raw scores(logits), implement softmax that converts them into probabilities.
print("10. Softmax implementation:")

vector = numpy.array([1.0, 2.0, 3.0])
print(f"Input vector: {vector}")

def softmax(vector,i):
    """
    Compute softmax values for each set of scores in z_i.
    
    Parameters:
    vector : array-like raw scores (logits) for each class
    
    Returns:
    numpy.ndarray
        Probabilities for each class
    """
    exp_z_i = numpy.exp(vector[i])
    sum_exp_z_j = 0
    for j in range(len(vector)):
        exp_z_j = numpy.exp(vector[j])

        sum_exp_z_j += exp_z_j
        
        
    softmax_i = exp_z_i / sum_exp_z_j

    return softmax_i
    
softmax_i = softmax(vector, 0)
print(f"softmax_0 = {softmax_i}")
    
softmax_i = softmax(vector, 1)
print(f"softmax_1 = {softmax_i}")

softmax_i = softmax(vector, 2)
print(f"softmax_2 = {softmax_i}")






