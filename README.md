
# Modality-Aware Shot Relating and Comparing for Video Scene Detection
This is an official PyTorch Implementation of **Modality-Aware Shot Relating and Comparing for Video Scene Detection**.

``supply.pdf`` is our supplement to the main manuscript.

## Environment

This project runs on Windows10 with one GPU (~12G) and a memory (~32G).

Install the following packages at first:
- python 3.9.2
- PyTorch 1.10.0
- torchvision 0.11.1
- numpy
- scikit-learn
- pickle
- json
- vit_pytorch


## Prepare Dataset
1. Download processed features (entity and place features) for MovieNet Dataset (Backbone is ResNet-50 Pretrained on ImageNet and Place365)
   https://pan.quark.cn/s/8452ff70183d  Code: Ev6E

   (If you are interested in how to process this dataset, please refer to https://github.com/mini-mind/VSMBD ）
2. Download MovieNet dataset label: https://drive.google.com/drive/folders/1F-uqCKnhtSdQKcDUiL3dRcLOrAxHargz

3. Generate the dataset by running **function gen_movienet(fore_path, back_path, lb_path, seg_sz=14, topk=4, have_graph=False, graph_path=None, save_path=None)** in dataProcess\DefineGraphv2.py;

   fore_path and back_path is the saving path of ImageNet_shot.pkl and Place_shot.pkl downloaded from step 1, respectively.

   lb_path is the saving path of the txt file download from step 2.

   If you downloaded graphs from https://pan.quark.cn/s/d2893715c722 Code：9RN3, then have_graph=True, graph_path='movienet_graph.pkl'.
   Else have_graph=False
   
## Train and Test
run main in main.py

## Quote

```
@InProceedings{vsd_masrc,
    author    = {Tan, Jiawei and Wang, Hongxing and Dang, Kang and Jiaxin, Li and Qu, Zhilong},
    title     = {Modality-Aware Shot Relating and Comparing for Video Scene Detection},
    booktitle = {The 39th Annual AAAI Conference on Artificial Intelligence},
    year      = {2025},
}
```
