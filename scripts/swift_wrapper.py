"""Wrapper to disable cuDNN before running swift training."""
import sys
import torch
torch.backends.cudnn.enabled = False

# Now run swift CLI
from swift.cli.main import cli_main
cli_main()
