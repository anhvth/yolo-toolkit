#!/bin/bash

# 1. export data from LS
# 2. train YOLO model
# all config will be loaded from ls_settings.json

python scripts/4_export_yolo.py
./train.sh