import logging
import subprocess
import atexit

from multiprocessing import Process
from pathlib import Path
from typing import Union
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger('.'.join(('main', 'tensorboardutils')))

RUN_SUBDIR_PREFIX = 'run-'

def make_log_sub_directory(dirpath: Path, accept_empty: bool = True) -> Path:
    """
    Create the log directory `run-$N` depending on preexisting dirs
    and return the path.

    Parameters
    ----------

    dirpath : Path
        Path to the parent directory, i.e. the experiment 
        directory.
    
    accept_empty : bool, optional
        Flag to set scanning for empty directories: If set to True,
        the function considers directories without elements
        as eligible and does not create a new run directory `run-$(N+1)`
    """
    run = 1
    while True:
        tentative_path = dirpath / ''.join((RUN_SUBDIR_PREFIX, str(run)))
        if not tentative_path.exists():
            break
        if accept_empty:
            content = [item for item in tentative_path.iterdir()]
            if len(content) == 0:
                break
        run +=1
    tentative_path.mkdir()
    return tentative_path

def setup_log_directory(experiment_dir: Union[str, Path]) -> Path:
    """Set up the log dir inside the specified experiment directory."""
    experiment_dir = Path(experiment_dir)
    if experiment_dir.is_file():
        raise FileExistsError('Indicated experiment directory is a file')
    elif experiment_dir.is_dir():
        logger.info(f'Using existing experiment directory "{experiment_dir.resolve()}"')
    else:
        experiment_dir.mkdir(parents=True) 
        logger.info(f'Created new experiment directory: "{experiment_dir.resolve()}"')
    log_dir = make_log_sub_directory(experiment_dir)
    return log_dir
    
def create_writer(experiment_dir: Union[str, Path], **kwargs) -> SummaryWriter:
    """
    Simplified creation of a summary writer instance that writes to
    a run-specific subdirectory created inside the experiment directory. 
    """
    log_dir = setup_log_directory(experiment_dir)
    # summary writer settings
    if 'flush_secs' not in kwargs:
        kwargs.update({'flush_secs' : 15})
    return SummaryWriter(log_dir=log_dir, **kwargs)

def start_tensorboard_server(log_dir: Union[str, Path], port: int) -> None:
    """Simplified starting of tensorboard visualization server from python"""
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        raise NotADirectoryError('Indicated log directory does not exist')
    subprocess.run(['tensorboard', f'--logdir={log_dir}', f'--port={port}'])


def popen_server(log_dir: Union[str, Path], port: int):
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        raise NotADirectoryError('Indicated log directory does not exist')
    proc = subprocess.Popen(['tensorboard', f'--logdir={log_dir}', f'--port={port}'],
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return proc


class TensorboardServer(Process):
    """
    Start a tensorboard server as a separate subprocess.
    Auto-registers the shutdown method via `atexit` module
    inside the calling/parent interpreter during instantiation. 
    """
    def __init__(self, log_dir: Union[str, Path], port: int = 6006) -> None:
        self.logger = self._get_logger()
        name = 'TensorboardServer'
        kwargs = {'log_dir' : log_dir, 'port' : port}
        super().__init__(name=name, target=start_tensorboard_server, kwargs=kwargs)
        atexit.register(self._kill_server)


    def _get_logger(self) -> logging.Logger:
        """Retrieve the instance logger."""
        return logging.getLogger('.'.join(('main', self.__class__.__name__)))
    
    def _kill_server(self) -> None:
        """Attempt to kill the server process."""
        self.logger.info('Server shutdown initiated')
        try:
            self.close()
        except ValueError:
            self.logger.warning('Shutdown via close method failed. Using terminate method')
            self.terminate()
    
            

