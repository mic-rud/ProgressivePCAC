import torch
import MinkowskiEngine as ME
import utils
# Define a sparse tensor with coordinates and features
coordinates = torch.tensor([
    [0, 0, 0, 0],
    [0, 1, 1, 1],
    [0, 2, 2, 2],
    [0, 3, 3, 3],
    [0, 4, 4, 4],
    [0, 8, 8, 8],
    [1, 1, 15, 10],
], dtype=torch.int32)


sparse_tensor = ME.SparseTensor(features=coordinates.float(), coordinates=coordinates)

# Define a sparse convolution layer
in_channels = 4
out_channels = 4
kernel_size = 3
stride = 2

convolution = ME.MinkowskiConvolution(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    dimension=3
)

# Perform the convolution
output = convolution(sparse_tensor)
print(output)
output = convolution(output)

# Print the result
print(output)


coordinates = utils.downsampled_coordinates(sparse_tensor.C, factor=4)
print(coordinates)