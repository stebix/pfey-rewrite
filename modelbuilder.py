import json
import torch

from pathlib import Path

fpath = Path('./modelconfig.json')

with open(fpath, mode='r') as handle:
    config = json.load(handle)

print(config)


from network import create_conv_block, ModularCellNet


result = create_conv_block(
    10, 20, 3, 'bnorm,conv,relu', 1, 'zeros',
    norm_kwargs={'affine' : False}
)
print(result)


config = {
    'in_channels' : 1,
    'target_class_count' : 10,
    'fmapspec' : [10, 32, 64, 128],
    'blockspec' : 'inorm,conv,relu',
    'final_pooling_mode' : 'adaptivemaxpool',
    'final_pooling_out_size' : (15, 15),
    'conv_kernel_size' : 3,
    'conv_padding' : 1,
    'downsample_mode' : 'stridedconv',
    'downsample_kernel_size' : 2,
    'downsample_stride' : 1,
    'activation_kwargs' : {'inplace' : True},
    'norm_kwargs' : {'affine' : False}
}

size = (3, 1, 128, 128)
testinput = torch.randn(size, dtype=torch.float32)

net = ModularCellNet(**config)
net.apply_final_softmax = True

print(net)

result = net(testinput)

print(f'Result shapoe: {result.size()}')

for i in range(3):
    print(f'Batch item {i} sum {result[i, :].sum()}')
    print(f'Batch item {i} {result[i, :]}')
