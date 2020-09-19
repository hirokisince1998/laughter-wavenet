# Trial

この節では、laughter-wavenet-startupkit で提供しているサンプルデータからモデル学習および合成までを行う手順を述べる。独自データからモデル学習および合成を行う手順については[HowTo](#HowTo)まで飛ばすべし。

## Dockerコンテナの起動

laughter-wavenetの実行環境一式は Docker Hub からダウンロードできる。

```
docker pull hirokisince1998/laughter-wavenet
```

一度ダウンロードしたら、後は以下のコマンドで実行できる。今現在いるディレクトリが Docker 上では /playground というディレクトリにマウントされるので、データ等のあるディレクトリに移動してから実行するとよい。

```
docker run --gpus='"device=0"' --rm -it --ipc=host -v $PWD:/playground -w /playground --env WNPATH=/work/laughter-wavenet hirokisince1998/laughter-wavenet bash
```

## 前処理 (Preprocessing)

まず、b,h等のラベルファイルから補助特徴量のファイルを生成する必要がある。次のスクリプトを実行すると、data ディレクトリの中にある訓練用データのラベルファイルから補助特徴量に変換し、音声波形ファイルと合わせて、train.py が読み込める numpy 形式のファイルが生成される。

```
bash scripts/preprocess-bh.sh
```

## 学習 (Training)

最初から学習を始めるためには次を実行する。

```
bash scripts/04_MSY-bh-train.sh
```

また、以前に保存したチェックポイントから学習を再開するためには次のようにする。

```
checkpointoption="--checkpoint=checkpoints/04_MSY-bh/checkpoint_step000031072.pth " bash scripts/04_MSY-bh-train.sh
```

## 合成 (Synthesis)

### ラベルファイルから補助特徴量ファイルを生成する

laughter-wavenet-startupkit では、test/fromlabel の下に合成テスト用のデータを用意してある。データは labels ディレクトリと questions ディレクトリから成る。labels ディレクトリの下にはHTK形式ラベルファイルを配置する。

```:04_MSY_181.lab
0 1010000 b
1010000 3160000 H
3160000 12250000 b
12250000 15240000 h
```

```:04_MSY_386-2.lab
0 8180000 b
8180000 10110000 H
10110000 14010000 b
14010000 15590000 H
15590000 23400000 H
23400000 25610000 h
```

次のコマンドで、test/fromlabel に格納されているデータから補助特徴量ファイルが生成され、 test/fromlabel の下の条件ID (ここではbh) の名前のディレクトリに保存される。

```
python $WNPATH/makecontext.py --preset=presets/laughter-bh.json bh test/fromlabel test/fromlabel
```

### WaveNet合成

次のコマンドで、gen の下の 04_MSY-bh の下に合成された wav ファイルができる。このシェルスクリプトでは、laughter-wavenet-startupkit で提供している pretrained の下にある学習済モデル(590000ステップ)と test/fromlabel/lh の下のデータから合成するよう指定している。

```
bash scripts/04_MSY-bh-fromlabel-gen.sh
```

# HowTo

手元のデータからモデル学習および合成を行うためのファイル準備、設定およびコマンドのガイド。

## 学習に必要なファイル

```
data
|-- questions
|   `-- questions_laugh-bh.hed
`-- training
    |-- labels
    |   `-- full-timealign
    |       |-- (話者ID)
    |       |   |-- 1番目ラベルファイル
    |       |   |-- ...
    |       |   `-- n番目ラベルファイル
    |       `-- (話者ID)
    |           |-- ...
    `-- wav
        |-- (話者ID)
        |   |-- 1番目wavファイル
        |   |-- ...
        |   `-- n番目wavファイル
        `-- (話者ID)
            |-- ...
```

このほか、C0も使ったモデルを学習する場合にはmgc というディレクトリの中にメルケプストラムのファイルが必要。

## 設定ファイル

前処理、学習、合成の各行程で同じ設定ファイルを指定する必要がある。
b,hのラベルだけから合成する場合にはpresets/laughter-bh.json を指定する。(サンプルスクリプトでは指定済)

以下に有用なパラメータを示す。

- question_fn: HTK HHEd 形式のコンテキストクラスタリング質問ファイルの場所。in_dir からの相対パスで指定する。デフォルトは questions_laugh-bh.hed 。ラベルの種類を増やす場合には質問ファイルを自作する必要がある。
- nepochs: 学習エポック数。デフォルトのままにしておいて、手動で強制終了させてもよい。
- test_eval_epoch_interval: 何エポック毎にテスト音声の合成を行うか。デフォルトは5だが、笑い声の場合は学習データが少ないので、この数字をもっと大きくしないと合成にかかる時間が占める割合が大きすぎてもったいない。

### コマンドライン

#### preprocess.py

```
Preprocess dataset

usage: preprocess.py [options] <name> <in_dir> <out_dir>

options:
    --num_workers=<n>        Num workers.
    --hparams=<parmas>       Hyper parameters [default: ].
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
```

name は laughter-bh (補助特徴量はb, h, H 情報)か laughter-c0 (補助特徴量は b, h, H に加えてC0) のどちらか。

in_dir はデータパス(上の例ではdata)。

out_dir は補助特徴量ファイルを書き出すディレクトリ。

(データパス)の下に labels, mgc, questions が必要。questions は、HTK形式のコンテキストクラスタリング質問ファイル *.hed を格納したディレクトリ。

**** train.py

```
Trainining script for WaveNet vocoder

usage: train.py [options]

options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --restore-parts=<path>       Restore part of the model.
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    -h, --help                   Show this help message and exit
```

**** synthesis.py

```
Synthesis waveform from trained WaveNet.

usage: synthesis.py [options] <checkpoint> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --length=<T>                      Steps to generate [default: 32000].
    --initial-value=<n>               Initial value for the WaveNet decoder.
    --conditional=<p>                 Conditional features path.
    --symmetric-mels                  Symmetric mel.
    --max-abs-value=<N>               Max abs value [default: -1].
    --file-name-suffix=<s>            File name suffix [default: ].
    --speaker-id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
```

--conditional-path には合成に使う補助特徴量ファイルを与える。
