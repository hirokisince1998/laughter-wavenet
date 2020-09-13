#!/bin/sh
SPEAKER=04_MSY

CONDID=bh
TRIAL=$SPEAKER-$CONDID
WORKDIR=.

STEP=000590000
CHECKPOINTDIR=./pretrained
GENDIR=./gen
CONTEXTDIR=./data/$CONDID

for utt in 020 033 038 043 050; do
    for i in `seq 0 9`; do
	python ${WNPATH:-.}/synthesis.py $CHECKPOINTDIR/$TRIAL/checkpoint_step${STEP}_ema.pth $GENDIR/$TRIAL/${TRIAL}-${utt}-$i --conditional=$CONTEXTDIR/laughter-context-00${utt}.npy --preset presets/laughter-$CONDID.json
	mv $GENDIR/$TRIAL/${TRIAL}-${utt}-${i}/checkpoint_step${STEP}_ema.wav $GENDIR/$TRIAL/${TRIAL}-${utt}-${i}.wav
	rmdir $GENDIR/$TRIAL/${TRIAL}-${utt}-${i}
    done
done
