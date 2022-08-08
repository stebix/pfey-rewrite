"""
Explanation for the different parameters:

in_channels :
    Channel count for the input spectra. Is probably always 1
    for our problem space.

target_class_count : 
    Integer target class count, i.e. the number of different cell types
    we want to distinguish. Must be set manually according to the training
    data setup and the data read from disk.

fmapspec : 
    The specification of the feature map number for every encoder block.
    Implicitly determines the number of encoder blocks.
    Should be a list of integers.

blockspec : 
    The convolution block specification. Should be a string consisting
    of three string aliases separated by commata for norm, activation and convolution.
    Examples:

    'inorm,conv,relu' apply instance norm, convolution and leaky relu
    'conv,gelu,bnorm' apply convolution, Gaussian Error Linear Unit and batchnorm

final_pooling_mode : 
    Select the final pooling mode via a string alias. Can be
    'adaptiveavgpool' or 'adaptivemaxpool'

final_pooling_out_size : 
    Set the final spatial feaure map size (H x W) that is produced by the
    final pooling op. The selected size should be smaller than the spatial
    size of the previous encoder output.
    Should be a 2-tuple of ints.

conv_kernel_size : 
    Global setting for the convolution kernel size.
    Can be an int for a quadratic kernel size or a 2-tuple of ints
    for a rectangular convolution kernel.
    The kernel size 3 is a good and widely used starting point.

conv_padding : 
    Global setting for feature map padding in the convolution
    operation. The widely used standard setting of 1 maintains
    the spatial feature map size for the convolution operation.
    Use an int for isotropic padding and a 2-tuple of int
    for anisotropic padding.

downsample_mode : 
    Select the downsampling mode via a string alias.
    Possible choices {'stridedconv', 'maxpool' , 'avgpool'}
    Pooling operations are not learned, i.e. without free parameters
    while the srided convolution is learned.

downsample_stride : 
    Set the stride of the fownsampling operation. Adjust this setting in
    parallel to selection of the downsample mode to achieve the
    desired downsampling behaviour.

activation_kwargs : 
    Any keyword arguments provided via this dictionary are forwarded to all
    nonlinearities/activations inside the net.
    Can be used to set slope parameter @ LeakyReLU or inplace setting @ ReLU

norm_kwargs :
    Any keyword arguments provided via this dictionary are forwarded to all
    normalization layers inside the net.
    Can be used to set affine parameter at various norm operations.

conv_kwargs :
    Any keyword arguments provided via this dictionary are forwarded to all
    nonlinearities/activations inside the net.
    Can be used to set convolution padding mode or bias usage.


Other explanations:

    net.apply_final_softmax = {True, False}
        Can be used to set application of the final softmax function to raw
        logits on and off.
        During training with a fitting loss function, set this to off.
        During testing and/or validating set this to on.
        Not that e.g. the CrossEntropyLoss function applies the softmax for
        us. Thus we do not need it during training with loss function application.

    shapeinfo
        Function to track and return the internal tensor shape state during
        the propagation trhough the network.
        Provide the actual spatial input shape as a 2-tuple (H x W) of ints to get
        accurate results.
        The result is provided as a list of 3-tuples of the format:
        (layer_name, inputshape, outputshape)
"""

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
    'downsample_stride' : 2,
    'activation_kwargs' : {'inplace' : True},
    'norm_kwargs' : {'affine' : False}
}
