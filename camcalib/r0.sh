#!/bin/sh

if [ -p fifo0 ] 
then 
    rm fifo0
fi

mkfifo fifo0
nc -l -v -p 5001 > fifo0
