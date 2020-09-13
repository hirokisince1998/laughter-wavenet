# 笑い声WaveNet

# Dockerコンテナの起動

```
docker run --gpus='"device=0"' --rm -it --ipc=host -v $PWD:/playground -w /playground --env WNPATH=/work/laughter-wavenet hirokisince1998/laughter-wavenet bash
```

# Preprocessing

```
bash scripts/preprocess-bh.sh
```

# Training

```
bash scripts/04_MSY-bh.train.sh
```

## チェックポイントからの再開

```
checkpointoption="--checkpoint=checkpoints/04_MSY-bh/checkpoint_step000031072.pth " bash scripts/04_MSY-bh-train.sh
```

# 合成

## コンテキストラベルを生成する

(データパス)の下に labels, mgc が必要。(データパス)と同じレベルに questions ディレクトリ(HTK形式のコンテキストクラスタリング質問ファイル *.hed を格納)が必要。

### b, Hラベルだけから生成 (loose conditioning用)

```
python makecontext.py --preset=presets/laughter-bh.json bh (データパス) (出力パス)
```

### b, HラベルおよびC0から生成 (conditioning by power用)

```
python makecontext.py --preset=presets/laughter-c0.json c0 (データパス) (出力パス)
```

# b, Hラベルからの合成
```
bash scripts/04_MSY-bh-resynth-gen.sh
```
