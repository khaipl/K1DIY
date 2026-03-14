#!/bin/bash

SERVER="192.168.127.101"
LOG_FILE="/var/log/chrony-presync.log"
MAX_RETRY=30
RETRY_DELAY=2

echo "$(date): Starting pre-synchronization" >> $LOG_FILE

if ! command -v ntpdate &> /dev/null; then
    echo "$(date): ntpdate not found, installing..." >> $LOG_FILE
    apt-get update && apt-get install -y ntpdate
fi

for i in $(seq 1 $MAX_RETRY); do
    if ! ping -c 1 -W 2 $SERVER &> /dev/null; then
        echo "$(date): Network not ready, skipping pre-sync" >> $LOG_FILE
    else
	break
    fi
    sleep $RETRY_DELAY
done

for i in $(seq 1 $MAX_RETRY); do
    echo "$(date): Pre-sync attempt $i/$MAX_RETRY" >> $LOG_FILE
    if ntpdate -b -u $SERVER; then
        echo "$(date): Pre-synchronization successful" >> $LOG_FILE
        hwclock --systohc
        exit 0
    fi
    sleep $RETRY_DELAY
done

echo "$(date): Pre-synchronization failed after $MAX_RETRY attempts" >> $LOG_FILE
exit 0
