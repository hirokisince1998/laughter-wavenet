#!/bin/sh
SPEAKER=04_MSY

CONDID=bh
TRIAL=$SPEAKER-$CONDID
WORKDIR=.

STEP=000590000
CHECKPOINTDIR=./pretrained
GENDIR=./gen
CONTEXTDIR=./test/fromlabel/$CONDID

for uttnpy in $CONTEXTDIR/${SPEAKER}*.npy; do
    utt=`basename $uttnpy .npy`
    python ${WNPATH:-.}/synthesis.py $CHECKPOINTDIR/$TRIAL/checkpoint_step${STEP}_ema.pth $GENDIR/$TRIAL/$utt --conditional=$uttnpy --preset presets/laughter-$CONDID.json
    mv $GENDIR/$TRIAL/$utt/checkpoint_step${STEP}_ema.wav $GENDIR/$TRIAL/${utt}.wav
    rmdir $GENDIR/$TRIAL/${utt}
done
