"""
Entry point for: python -m nanogpt_volunteer

Prints usage and available sub-commands.
"""

import sys


USAGE = """\
nanoGPT Volunteer Computing
============================

Train GPT models using volunteer-contributed compute resources.

Sub-commands:
  python -m nanogpt_volunteer.prepare_data   Prepare training data (downloads tiny shakespeare by default)
  python -m nanogpt_volunteer.coordinator    Start the coordinator server
  python -m nanogpt_volunteer.volunteer      Start a volunteer client
  python -m nanogpt_volunteer.sample         Sample text from a trained checkpoint

Quick start:
  1. Prepare data:
       python -m nanogpt_volunteer.prepare_data

  2. Start coordinator (on a machine with the data):
       python -m nanogpt_volunteer.coordinator --data_dir data/shakespeare_char

  3. Connect volunteers (on any machine):
       python -m nanogpt_volunteer.volunteer --host <coordinator-ip> --port 9876

  4. Sample from trained model:
       python -m nanogpt_volunteer.sample --out_dir out-volunteer

Each sub-command supports --help for full option listing.
"""


def main():
    print(USAGE)


if __name__ == '__main__':
    main()
