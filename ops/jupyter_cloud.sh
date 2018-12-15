#!/usr/bin/env sh

ssh aws-gpu 'source activate dlb && cd workspace/dlbook && jupyter lab'
