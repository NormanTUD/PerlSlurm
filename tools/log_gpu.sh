#!/bin/bash

export LOGFILE=$1
export HOST=$(hostname)
export CUDA_FILE=/tmp/LOG_CUDA_VISIBLE_DEVICES

if [ -f $CUDA_FILE ]; then
    export MY_CUDA_VISIBLE_DEVICES=$(<$CUDA_FILE)
    touch $LOGFILE;
    if [[ -s $LOGFILE ]]; then
        # File is NOT empty
        nvidia-smi -i $MY_CUDA_VISIBLE_DEVICES --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader >> $LOGFILE
    else
        # File IS empty
        nvidia-smi -i $MY_CUDA_VISIBLE_DEVICES --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv > $LOGFILE
    fi
else
    echo "ERROR: $CUDA_FILE not found on $HOST"
fi

