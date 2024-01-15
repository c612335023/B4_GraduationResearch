# MusicTransformerに基づく補間フレーズの生成

## 手順とか
1. 必要なやつのcloneとか
---
    git clone https://github.com/jason9693/midi-neural-processor.git
    mv midi-neural-processor/processor.py midi_processor

    mkdir dataset
    mkdir dataset/midi

    git clone https://github.com/djosix/Performance-RNN-PyTorch

2. datasetのダウンロード
    1. datasetのダウンロードプログラム(ecomp_piano_downloader.sh)のエラー修正  
        #### Performance-RNN-PyTorch/dataset/scripts/ecomp_piano_downloader.shの5行目以降のURLを以下に変更
        ---
            https://www.piano-e-competition.com/midi_2002.asp
            https://www.piano-e-competition.com/midi_2004.asp
            https://www.piano-e-competition.com/midi_2006.asp
            https://www.piano-e-competition.com/midi_2008.asp
            https://www.piano-e-competition.com/midi_2009.asp
            https://www.piano-e-competition.com/midi_2011.asp
            https://www.piano-e-competition.com/midi_2013.asp
            https://www.piano-e-competition.com/midi_2014.asp
            https://www.piano-e-competition.com/midi_2015.asp
            https://www.piano-e-competition.com/midi_2017.asp
            https://www.piano-e-competition.com/midi_2018.asp

    2. datasetのダウンロード
    ---
        sh Performance-RNN-PyTorch/dataset/scripts/ecomp_piano_downloader.sh dataset/midi

3. datasetの前処理
    1. 壊れているファイルの削除（こっちは確定）
    ---
        rm dataset/midi/KaszoS14.MID
    
    2. 前処理がおかしくなるファイルの削除（これは削除しなくて大丈夫かも）
    ---
        rm dataset/midi/LiA01.MID dataset/midi/Jussow07.MID dataset/midi/DANILO04.mid
    3. 実行
    ---
        python3 preprocess.py dataset/midi/ dataset/pickle

4. 学習方法
    1. ハイパーパラメータの設定
        - config/train.yml
        - config/base.yml
        - config/debug_train.yml
    2. 実行
    ---
        python3 advanced/train/dpc.py -c config/train.yml config/base.yml config/debug_train.yml -m dataset/model

5. 生成方法
    1. ハイパーパラメータの設定
        - config/generate.yml
        - config/base.yml
    
    2. 実行
    ---
        python3 generate.py -c config/generate.yml config/base.yml -m dataset/model