"""Simplified starting of TensorBoard server"""
import argparse
from tensorboardutils import start_tensorboard_server

DEFAULT_PORT = 6006

def main():
    parser = argparse.ArgumentParser(
        description='TensorBoard server startup tool. Input experiment directory.'
    )
    parser.add_argument(
        'log_dir', type=str,
        help='The logging directory where the SummaryWriter instance dumps to.'
    )
    parser.add_argument(
        '-p', '--port', type=int, default=DEFAULT_PORT,
        help='Port number for the server.'
    )
    args = parser.parse_args()
    start_tensorboard_server(args.log_dir, args.port)



if __name__ == "__main__":
    main()