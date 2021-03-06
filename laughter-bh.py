from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
from nnmnkwii.io import hts
from nnmnkwii import preprocessing as P
import merlin as fe # <- modified by hiroki
from nnmnkwii.datasets import FileDataSource
from hparams import hparams
from os.path import exists, join
from glob import glob
import librosa

available_speakers = [ "04_MSY", "06_FWA" ]

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    speakers = available_speakers

    wd = WavFileDataSource(in_dir, speakers=speakers)
    td = TranscriptionFileDataSource(in_dir, speakers=speakers)
    wav_paths = wd.collect_files()
    lab_paths = td.collect_files()
    speaker_ids = wd.labels
    binary_dict, continuous_dict = hts.load_question_set(join(in_dir, "questions", hparams.question_fn))

    result = []
    for index, (speaker_id, wav_path, lab_path) in enumerate(
            zip(speaker_ids, wav_paths, lab_paths)):
        result.append(_process_utterance(out_dir, index + 1, speaker_id, wav_path, lab_path, binary_dict, continuous_dict, "N/A"))
    return result

def _process_utterance(out_dir, index, speaker_id, wav_path, lab_path, binary_dict, continuous_dict, text):
    # Load the audio to a numpy array. Resampled if needed
    wav = audio.load_wav(wav_path)

    # determine sessionID and uttID
    wavbn = os.path.basename(wav_path)
    uttID = os.path.splitext(wavbn)[0]

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # [0, quantize_channels)
        out = P.mulaw_quantize(wav, hparams.quantize_channels)
        constant_values = P.mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16
    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        out = P.mulaw(wav, hparams.quantize_channels)
        constant_values = P.mulaw(0.0, hparams.quantize_channels)
        out_dtype = np.float32
    else:
        # [-1, 1]
        out = wav
        constant_values = 0.0
        out_dtype = np.float32

    # time-aligned context
    if hparams.frame_shift_ms is None:
        frame_shift_in_micro_sec = (hparams.hop_size * 10000000) // hparams.sample_rate
    else:
        frame_shift_in_micro_sec = hparams.frame_shift_ms * 10000
    labels = hts.HTSLabelFile(frame_shift_in_micro_sec)
    labels.load(lab_path)
    linguistic_features = fe.linguistic_features(labels, binary_dict, continuous_dict, add_frame_features=True, frame_shift_in_micro_sec = frame_shift_in_micro_sec)

    Nwav = len(out) // audio.get_hop_size()
    out = out[:Nwav * audio.get_hop_size()]

    timesteps = len(out)

    context = linguistic_features

    # Write the spectrograms to disk:
    audio_filename = 'audio-' + uttID + '.npy'
    context_filename = 'context-' + uttID + '.npy'
    np.save(os.path.join(out_dir, audio_filename),
            out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, context_filename),
            context.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return (audio_filename, context_filename, timesteps, text, speaker_id)

class _LaughterBaseDataSource(FileDataSource):
    def __init__(self, data_root, speakers, labelmap, max_files):
        self.data_root = data_root
        self.speakers = speakers
        if labelmap is None:
            labelmap = {}
            for idx, speaker in enumerate(speakers):
                labelmap[speaker] = idx
        self.labelmap = labelmap
        self.labels = None
        self.max_files = max_files

    def collect_files(self, is_wav):
        if is_wav:
            root = join(self.data_root, "training", "wav")
            ext = ".wav"
        else:
            root = join(self.data_root, "training", "labels", "full-timealign")
            ext = ".lab"

        paths = []
        labels = []

        if self.max_files is None:
            max_files_per_speaker = None
        else:
            max_files_per_speaker = self.max_files // len(self.speakers)
        for idx, speaker in enumerate(self.speakers):
            files = sorted(glob(join(root, speaker, "{}_*{}".format(speaker, ext))))
            files = files[:max_files_per_speaker]
            for f in files:
                paths.append(f)
                labels.append(self.labelmap[self.speakers[idx]])
        self.labels = np.array(labels, dtype=np.int16)

        return paths


class TranscriptionFileDataSource(_LaughterBaseDataSource):
    def __init__(self, data_root, speakers=available_speakers, labelmap=None, max_files=None):
        super(TranscriptionFileDataSource, self).__init__(
            data_root, speakers, labelmap, max_files)

    def collect_files(self):
        return super(TranscriptionFileDataSource, self).collect_files(False)

class WavFileDataSource(_LaughterBaseDataSource):
    def __init__(self, data_root, speakers=available_speakers, labelmap=None, max_files=None):
        super(WavFileDataSource, self).__init__(
            data_root, speakers, labelmap, max_files)

    def collect_files(self):
        return super(WavFileDataSource, self).collect_files(True)
