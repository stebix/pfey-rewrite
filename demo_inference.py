"""
Example script for inference with a trained model.
Just a skeleton for demonstration purposes: not intended to be runnable
out of the box
"""
import torch

# custom written code imports
from trainer import (train,
                     create_default_criterion,
                     create_default_trainloader,
                     create_default_validationloader,
                     validate)
from tensorboardutils import create_writer
from network import VGGCellNet

network_config = {
    'in_channels' : 1,
    'target_class_count' : 10,
    'conv_blockspec' : [10, 'maxpool', 32, 32, 'maxpool', 64, 64, 'maxpool', 128, 256],
    'pooling_outsize' : (8, 8),
    'conv_block_activation_type' : 'lrelu',
    'conv_block_activation_kwargs' : {'negative_slope' : 0.125},
    'classifier_widthspec' : [2048, 1024, 512],
    'classifier_activation_type' : 'lrelu',
    'classifier_activation_kwargs' : {'negative_slope' : 0.125},
    'use_dropout' : True,
    'p_dropout' : 0.5
}

# parameters
learning_rate = 1e-3
weigth_decay = 0.05
n_epoch = 100
device = torch.device('cuda:0')

train_loader = create_default_trainloader()
val_loader = create_default_validationloader()
model = VGGCellNet(**network_config)
criterion = create_default_criterion()
optimizer = torch.nn.AdamW(model.parameters(), lr=learning_rate,
                           weigth_decay=weigth_decay)
writer = create_writer(experiment_dir='.')

### Explanation:
# The model gets trained inside the below standing function. This means that the
# internal parameters of the model can be retrieved via `model.parameters()`
# are optimized after the `train(...)` function returns.
# The model object and instance themselves are callable. This means that
# the prediction result for any tensor `input_tensor` of the format (B x C x H x W)
# with B : batch dimension, C : channel dimension, H : height dimension and
# W : width dimension is given by calling the model object with the tensor
# as the sole argument : ´prediction = model(input_tensor)´
model.train() # set to training mode just to be sure ...
loss_history = train(
    model=model, optimizer=optimizer,
    n_epoch=n_epoch, trainloader=train_loader,
    validationloader=val_loader, validation_fn=validate,
    writer=writer, device=device
)
# Now we have to set the model to evaluation mode to allow the BatchNormalization
# layers, the dropout layers and the final application of the softmax nonlinearity
# to work as intended.
model.eval()
# Model is trained and optimized at this point. We can evaluate the prediction
# for any tensor:
size = (1, 1, 300, 300) # layout is (B x C x H x W)
input_tensor = torch.randn(size)
prediction = model(input_tensor)
# Above we get the prediction for the indicated tensor. The prediction can
# be utilized to evaluate accuracy or any other metric we like.