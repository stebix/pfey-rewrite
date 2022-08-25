import torch
import logging

from typing import Union, Optional
from torch.utils.tensorboard import SummaryWriter

from utils import recursive_to_device, deduce_device

NOTEBOOK_ENV = True
if NOTEBOOK_ENV:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

logger = logging.getLogger('.'.join(('main', 'trainer')))

def mocktrain(ep, iter_per_ep):
    """Test correct display of progress bars via tqdm"""
    from time import sleep
    for e in tqdm(range(ep), unit='bt'):
        for i in tqdm(range(iter_per_ep), unit='bt', leave=False):
            sleep(0.125)
    return None


def process_label(label: torch.Tensor) -> torch.Tensor:
    """Adjust label shape and dtype for the picky `CrossEntropyLoss`"""
    label = torch.squeeze(label).to(torch.long)
    return label


def process_data(data: torch.Tensor) -> torch.Tensor:
    """Recast data tensor to `torch.float32`"""
    return data.to(torch.float32)


def process_batchdata(batchdata: tuple) -> tuple:
    """
    Preprocess and split data coming from dataloader. 
    Inject data typecasting etc. here.
    """
    (data, label) = batchdata
    return (process_data(data), process_label(label))


def create_default_optimizer(model: torch.nn.Module,
                             learning_rate: float,
                             **optim_kwargs) -> torch.optim.Optimizer:
    """Create default optimizer: Standard ADAM"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 **optim_kwargs)
    return optimizer


def create_default_criterion() -> torch.nn.Module:
    """Create default cirterion for classification: cross entropy"""
    criterion = torch.nn.CrossEntropyLoss()
    return criterion


def create_default_trainloader(dataset: torch.utils.data.Dataset,
                               batch_size: int) -> torch.utils.data.DataLoader:
    """Quick-create a training dataloader with sane default settings."""
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_default_validationloader(dataset: torch.utils.data.Dataset,
                                    batch_size: int) -> torch.utils.data.DataLoader:
    """Quick-create a validation dataloader with sane default settings."""
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)



def validate(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
             writer: SummaryWriter, global_step: int, tag: str) -> None:
    """Perform a validation run and write the result to the writer instance."""
    model.eval()
    # deduce validation device from the model itself
    device = deduce_device(model)
    total_prediction_count = 0
    total_correct_predictions = 0
    loader = tqdm(loader, unit='bt', leave=False, desc='Validate')
    with torch.no_grad():
        for batchindex, batchdata in enumerate(loader):
            batchdata = recursive_to_device(batchdata, device)
            data, label = process_batchdata(batchdata)
            output = model(data)
            _, prediction = torch.max(output, dim=1)
            # compute correct prediction
            correct_predictions = (prediction == label).sum().item()
            batchitems = prediction.shape[0]
            total_prediction_count += batchitems
            total_correct_predictions += correct_predictions
    accuracy = total_correct_predictions / total_prediction_count
    writer.add_scalar(tag, scalar_value=accuracy,
                      global_step=global_step)
    # reset to training state
    model.train()
    return None


def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          n_epoch: int,
          trainloader: torch.utils.data.DataLoader,
          validationloader: torch.utils.data.DataLoader,
          validation_fn: callable,
          writer: SummaryWriter,
          device: Union[str, torch.device],
          validate_at_epoch_end: bool = True,
          validate_after_n_iters: Optional[int] = None) -> list:
    """
    Train model for n epochs and return the loss history.

    Parameters
    ----------

    model : torch.nn.Module
        The machine learning model.

    optimizer : torch.optim.optimizer
        Usable, instantiated and configured optimizer instance.
    
    criterion : torch.nn.Module
        Loss function callable.
    
    n_epoch : int
        Number of training epochs performed during the function
        run.
    
    trainloader : torch.utils.data.DataLoader
        The dataloader encapsulating the training data.
    
    validationloader : torch.utils.DataLoader
        The dataloader encapsulating the validation data.

    validation_fn : callable
        Validation function callable: Allows the injection of
        variable validation code. Must possess the signature
        `validation_fn(model, validationloader, writer)`
    
    writer : torch.utils.tensorboard.SummaryWriter
        Usable, instantiated and configured writer instance
        to log the experiment state during the runtime of the
        train function.
    
    device : str or torch.Device
        The training device. Model and tensors will be moved to
        this device for training purposes.

    validate_at_epoch_end : bool, optional
        Set validation run strictly at epoch end.
        Defaults to True.

    validate_after_n_iters : int, optional
        Set the validation run to be performed after
        indicated amount of iterations. For this
        setting to be effective, the validate_at_epoch_end
        flag must be set to False.
        Defaults to None.
    """
    # check consistency & set up validation behaviour
    if validate_at_epoch_end and validate_after_n_iters is not None:
        raise ValueError(
            'if validation_at_epoch_end is set, then validate_after_n_iters must be None'
        )
    # device setup
    device = torch.device(device)
    model.to(device)
    # cache variables
    loss_history = []
    # keep track of global step for correct logging via tensorboard
    global_step = 0

    model.train()
    for epoch in tqdm(range(n_epoch), unit='ep', desc='Total'):
        cumloss = 0
        # wrap training loader for progress feedback
        trainloader = tqdm(trainloader, unit='bt', leave=False, desc='IntraEpoch')
        for batchindex, batchdata in enumerate(trainloader):
            batchdata = recursive_to_device(batchdata, device)
            data, label = process_batchdata(batchdata)
            # TODO BOOTLEG NORMALIZATION ACTIVE
            data = data / 7500
            # zero the parameter gradients
            optimizer.zero_grad()
            # perform forward # backward pass and do optimization step
            prediction = model(data)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            # cast as single scalar
            loss = loss.item()
            cumloss += loss
            # logging stuff
            writer.add_scalar('loss/training-batch', scalar_value=loss, global_step=global_step)
            global_step += 1

            if validate_after_n_iters is not None:
                if global_step % validate_after_n_iters == 0:
                    validation_fn(model, validationloader, writer, global_step,
                                  tag='metrics/validation-accuracy')
                    validation_fn(model, trainloader, writer, global_step,
                                 tag='metrics/training-accuracy')
        if validate_at_epoch_end:
            validation_fn(model, validationloader, writer, global_step,
                          tag='metrics/validation-accuracy')
            validation_fn(model, trainloader, writer, global_step,
                          tag='metrics/training-accuracy')

        # record cumulative loss after every epoch
        loss_history.append(cumloss)
        writer.add_scalar('loss/training-epoch-cumulative', scalar_value=cumloss,
                          global_step=global_step)
    return loss_history
