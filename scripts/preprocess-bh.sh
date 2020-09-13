#!/bin/sh
for CONDID in bh c0; do
    python preprocess.py laughter-$CONDID data data/$CONDID --preset presets/laughter-$CONDID.json
done
