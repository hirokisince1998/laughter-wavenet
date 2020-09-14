# Dockerコンテナの起動

laughter-wavenetの実行環境一式は Docker Hub からダウンロードできる。

```
docker pull hirokisince1998/laughter-wavenet
```

一度ダウンロードしたら、後は以下のコマンドで実行できる。今現在いるディレクトリが Docker 上では /playground というディレクトリにマウントされるので、データ等のあるディレクトリに移動してから実行するとよい。

```
docker run --gpus='"device=0"' --rm -it --ipc=host -v $PWD:/playground -w /playground --env WNPATH=/work/laughter-wavenet hirokisince1998/laughter-wavenet bash
```

# 前処理 (Preprocessing)

(dataディレクトリの構造を書く)

まず、b,h等のラベルファイルから補助特徴量のファイルを生成する必要がある。次のスクリプトを実行すると、data ディレクトリの中にある訓練用データのラベルファイルから補助特徴量に変換し、音声波形ファイルと合わせて、train.py が読み込める numpy 形式のファイルが生成される。

```
bash scripts/preprocess-bh.sh
```

preprocess-bh.sh のヘルプ
```



# 学習 (Training)

```
bash scripts/04_MSY-bh-train.sh
```

## チェックポイントからの再開

```
checkpointoption="--checkpoint=checkpoints/04_MSY-bh/checkpoint_step000031072.pth " bash scripts/04_MSY-bh-train.sh
```

# 合成 (Synthesis)

## コンテキストラベルを生成する

(データパス)の下に labels, mgc, questions が必要。questions は、HTK形式のコンテキストクラスタリング質問ファイル *.hed を格納したディレクトリ。

### b, Hラベルだけから生成 (loose conditioning用)

```
python $WNPATH/makecontext.py --preset=presets/laughter-bh.json bh (データパス) (出力パス)
```

### b, HラベルおよびC0から生成 (conditioning by power用)

```
python $WNPATH/makecontext.py --preset=presets/laughter-c0.json c0 (データパス) (出力パス)
```

## b, Hラベルからの合成
```
bash scripts/04_MSY-bh-fromlabel-gen.sh
```

# TODO
- JSONファイルのどこを書き換えればよいのか
- ファイルの依存関係の説明を加筆 (questions_laugh-bh.hed という名前は json ファイルの中で指定されている、とか)
