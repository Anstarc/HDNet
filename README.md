# [HDNet: Hybrid Distance Network for semantic segmentation](https://www.sciencedirect.com/science/article/pii/S0925231221004185)

A deep neural network with attention mechanism for semantic segmentation task.

&nbsp;
![Architecture of HDNet](./images/arch.png "Architecture of HDNet")

&nbsp;
- Segmentation results on PASCAL Context

  ![Segmentation results on PASCAL Context](./images/seg_pcontext.png)

&nbsp;
- Application on skin detection task

  ![Application on skin detection task](./images/seg_skin.png)

&nbsp;

### Requirements
- Install [PASCAL in Detail](https://sites.google.com/view/pasd/dataset)
- Install `requirements.txt`

&nbsp;

### How to use

- To train HDNet:
```bash
python train.py 
```
The model and log are saved in `--out_dir`

&nbsp;

- To test HDNet:
```bash
python test.py
```
&nbsp;

- To get the visual results:
```bash
python vis.py
```

&nbsp;
### Citation
if you find HDNet useful in your research, please consider citing:

```
@article{li2021hdnet,
  title={HDNet: Hybrid Distance Network for semantic segmentation},
  author={Li, Chunpeng and Kang, Xuejing and Zhu, Lei and Ye, Lizhu and Feng, Panhe and Ming, Anlong},
  journal={Neurocomputing},
  volume={447},
  pages={129-144},
  year={2021},
  publisher={Elsevier}
}
```

### Acknowledgement
Thanks for [DANet](https://github.com/junfu1115/DANet), [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), [timm](https://github.com/rwightman/pytorch-image-models)