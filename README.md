# MamboNet

硬體設備與環境
===
Ubuntu16.04  
ROS Kinetic  
CUDA 10  
Anaconda  
Geforce RTX 2080  

事前準備
===
1. 下載 SemanticKITTI 放到 home 底下
2. 

Anaconda 虛擬環境設定
===
先建一個虛擬環境，環境名為mambonet
```
conda env create -f mambonet_env.yml --name mambonet
```
進入該環境
```
conda activate mambonet
```

再安裝一些遺漏的
```
pip install tb-nightly
pip install future
```

Training
===
```
./train.sh -d ../Semantickitti/dataset/ -a salsanext.yml -l logs -c 0
```

Evaluate
===
```
./eval.sh -d ../Semantickitti/dataset/ -p ./pred -m "pretrained model path" -s valid salsanext -c 30
```
