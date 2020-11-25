
# Attentional Graph Neural Network for Parking Slot Detection

![image](https://github.com/Jiaolong/gcn-parking-slot/blob/main/images/animated.gif)

Repository for the paper ["Attentional Graph Neural Network for Parking Slot Detection"](https://arxiv.org/).
```
@inproceedings{gcn-parking-slot:2020,
  title={Attentional Graph Neural Network for Parking Slot Detection},
  author={M. Chen, J. Xu, L. Xiao, D. Zhao etal},
  booktitle={arxiv},
  year={2020}
}
```

## Requirements

- python 3.6

- pytorch 1.7

- other requirements: `pip install -r requirements.txt`

## Pretrained models

Pre-trained weights can be downloaded [here](https://pan.baidu.com/s/137ZHZnsEfyaO4yaa5YoBIQ) (code:bc0a).

## Prepare data

The original ps2.0 data and label can be found [here](https://github.com/Teoge/DMPR-PS). Extract and organize as follows:

```
├── datasets
│   └── parking_slot
│       ├── ps_json_label 
│       ├── testing
│       └── training
```
## Train & Test

Export current directory to `PYTHONPATH`:

```bash
export PYTHONPATH=`pwd`
```

- demo

```
python3 tools/demo.py -c config/ps_gat.yaml -m cache/ps_gat/100/models/checkpoint_epoch_60.pth
```

- train

```
python3 tools/train.py -c config/ps_gat.yaml
```

- test

```
python3 tools/test.py -c config/ps_gat.yaml -m cache/ps_gat/100/models/checkpoint_epoch_60.pth
```

## References

[1] J. Huang, L. Zhang, Y. Shen, H. Zhang, and Y. Yang, “DMPR-PS: A novel approach for parking-slot detection using directional marking-point regression,” in IEEE International Conference on Multimedia and Expo (ICME), 2019. [code](https://github.com/Teoge/DMPR-PS)
