import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmer', type=int, default=12, help='k-mer value')
    parser.add_argument('--model', type=str, default='cnn', help='Model to use')
    parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1', help='Master address')
    parser.add_argument('--master-port', type=str, default='12345', help='Master port')
    parser.add_argument('--world-size', type=int, default=1, help='World size')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout for the distributed setup')
    parser.add_argument('--in-mem', type=int, default=0, help='Load data in memory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs')
    return parser.parse_args()
