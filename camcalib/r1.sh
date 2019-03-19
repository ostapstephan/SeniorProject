#!/bin/sh

if [ -p fifo1 ] 
then 
    rm fifo1
fi

mkfifo fifo1
nc -l -v -p 5002 > fifo1
