import torch

from typing import Sequence

NOTEBOOK_ENV = True

if NOTEBOOK_ENV:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


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


def create_default_dataloader(dataset: torch.utils.data.Dataset,
                              batch_size: int) -> torch.utils.data.DataLoader:
    """Quick-create a dataloader with sane default settings."""
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          n_epoch: int) -> list:
    """
    Train model for n epochs and return the loss history.
    """
    loss_history = []
    model.train()
    dataloader = tqdm(dataloader, unit='bt', leave=False)

    for epoch in tqdm(range(n_epoch), unit='ep'):
        cumloss = 0
        for batchindex, batchdata in enumerate(dataloader):
            data, label = process_batchdata(batchdata)
            # zero the parameter gradients
            optimizer.zero_grad()
            # perform forward # backward pass and do optimization step
            prediction = model(data)
            loss = criterion(prediction, label)

            loss.backward()
            optimizer.step()

            cumloss += loss.item()
        # record cumulative loss after every epoch
        loss_history.append(cumloss)
    return loss_history
