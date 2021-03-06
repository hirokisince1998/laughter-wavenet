#!/bin/bash
SPEAKER=06_FWA

declare -A SPEAKERID=([04_MSY]=0 [06_FWA]=1)
CONDID=bh
TRIAL=$SPEAKER-$CONDID
WORKDIR=.

python ${WNPATH:-.}/train.py --data-root data/$CONDID --speaker-id=${SPEAKERID[$SPEAKER]} --checkpoint-dir $WORKDIR/checkpoints/$TRIAL --preset presets/laughter-$CONDID.json --log-event-path log $checkpointoption
