#!/bin/bash
SPEAKER=04_MSY

declare -A SPEAKERID=([04_MSY]=0 [06_FWA]=1)
CONDID=bh
TRIAL=$SPEAKER-$CONDID
WORKDIR=.

python train.py --data-root data/$CONDID --speaker-id=${SPEAKERID[$SPEAKER]} --checkpoint-dir $WORKDIR/checkpoints/$TRIAL --preset presets/laughter-$CONDID.json --log-event-path log
