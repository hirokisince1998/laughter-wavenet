# coding: utf-8
"""
Convert HTS-style label and mcep (c0) into numpy context file

usage: makecontext.py [options] <condid> <datadir> <out_dir>

options:
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""

import re
from glob import glob
from docopt import docopt
import os
from os.path import join, basename
from nnmnkwii.io import hts
import merlin as fe # <- modified by hiroki
import numpy as np
from hparams import hparams
from audio import _amp_to_db, _normalize

if __name__ == "__main__":
    args = docopt(__doc__)
    condid = args["<condid>"]
    datapath = args["<datadir>"]
    out_dir = args["<out_dir>"]
    preset = args["--preset"]
    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())

    assert condid in ["bh", "c0"]

    speaker = "04_MSY"
    labpath = join(datapath, "labels", "full-timealign", speaker)
    labfiles = sorted(glob(join(labpath, "{}_*.lab".format(speaker))))
    mgcpath = join(datapath, "mgc")
    binary_dict, continuous_dict = hts.load_question_set(join(datapath, "..", "questions", hparams.question_fn))

    for labfn in labfiles:
        # time-aligned context
        if hparams.frame_shift_ms is None:
            frame_shift_in_micro_sec = (hparams.hop_size * 10000000) // hparams.sample_rate
        else:
            frame_shift_in_micro_sec = hparams.frame_shift_ms * 10000
            labels = hts.HTSLabelFile(frame_shift_in_micro_sec)
            labels.load(labfn)
            linguistic_features = fe.linguistic_features(labels, binary_dict, continuous_dict, add_frame_features=True, frame_shift_in_micro_sec = frame_shift_in_micro_sec)

            if condid == "bh":
                context = linguistic_features
                
            if condid == "c0":
                mgcfn = join(mgcpath, speaker, re.sub('.lab$','.mgc', basename(labfn)))
                fp = open(mgcfn, 'rb')
                mgc = np.fromfile(fp, np.float32, -1) - np.log(32768)
                fp.close()
                N = len(mgc) // hparams.num_mels
                mgc = np.reshape(mgc, (N, hparams.num_mels))
                c0 = _normalize(_amp_to_db(np.exp(mgc[0:len(linguistic_features),0:1])))

                # combine linguistic + pow (c0)
                context = np.hstack((linguistic_features, c0))

            context_filename = re.sub('.lab$','.npy', basename(labfn))
            out_path = join(out_dir,condid,speaker)
            os.makedirs(out_path, exist_ok=True)
            np.save(join(out_path, context_filename),
                    context.astype(np.float32), allow_pickle=False)

