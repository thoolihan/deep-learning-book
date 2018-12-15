#!/usr/bin/env bash

ssh aws-gpu mkdir .kaggle
scp ~/.kaggle/kaggle.json aws-gpu:.kaggle/
scp post-terraform-batch.sh aws-gpu:
ssh aws-gpu chmod 755 post-terraform-batch.sh
ssh aws-gpu ./post-terraform-batch.sh
ssh aws-gpu rm post-terraform-batch.sh
