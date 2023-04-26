#!/bin/bash
python3.10 collector.py > oneai-$(date +"%m-%d-%Y").log 2>&1 &