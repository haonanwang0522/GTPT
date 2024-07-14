# GTPT: Group-based Token Pruning Transformer for Efficient Human Pose Estimation [arxiv](https://arxiv.org/abs/)

> [**GTPT: Group-based Token Pruning Transformer for Efficient Human Pose Estimation**](https://arxiv.org/abs/)<br>
> Accepted by **ECCV 2024**<br>
> [Haonan Wang](https://github.com/haonanwang0522), Jie Liu, Jie Tang, [Gangshan Wu](http://mcg.nju.edu.cn/member/gswu/en/index.html), Bo Xu, Yanbing Chou, Yong Wang

## News!
- [2024.07.14] The pretrained models are released in [Google Drive](https://drive.google.com/drive/folders/188-x9NyIhXnU8k5_b1dRQbRRDg4CxKtI?usp=sharing)!
- [2024.07.13] The codes for SRPose are released!
- [2024.07.01] Our paper ''GTPT: Group-based Token Pruning Transformer for Efficient Human Pose Estimation'' has been accpeted by **ECCV 2024**. If you find this repository useful please give it a star 🌟. 


## Introduction
This is the official implementation of [GTPT: Group-based Token Pruning Transformer for Efficient Human Pose Estimation](https://arxiv.org/abs/). We propose the Group-based Token Pruning  Transformer (GTPT) that fully harnesses the advantages of the Transformer. GTPT alleviates the computational burden by gradually introducing keypoints in a coarse-to-fine manner. It minimizes the computation overhead while ensuring high performance. Besides, GTPT groups keypoint tokens and prunes visual tokens to improve model performance while reducing redundancy. We propose the Multi-Head Group Attention (MHGA) between different groups to achieve global interaction with little computational overhead. We conducted experiments on COCO and COCO-WholeBody. Compared to other methods, the experimental results show that GTPT can achieve higher performance with less computation, especially in whole-body with numerous keypoints. 

<img width="1183" alt="image" src="imgs\overall.png">

## Experiments

### Results on COCO validation set
<table border="1">
  <thead>
    <tr>
      <th>Method</th>
      <th>GFLOPs</th>
      <th>Params (M)</th>
      <th>AP</th>
      <th>AR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5"><strong>CNN-based Methods</strong></td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/1804.06208">SimBa.-Res50</a></td>
      <td>8.9</td>
      <td>34</td>
      <td>70.4</td>
      <td>76.3</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/1804.06208">SimBa.-Res101</a></td>
      <td>12.4</td>
      <td>53</td>
      <td>71.4</td>
      <td>77.1</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/1804.06208">SimBa.-Res152</a></td>
      <td>15.7</td>
      <td>68.6</td>
      <td>72.0</td>
      <td>77.8</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/1902.09212">HRNet-W32</a></td>
      <td>7.1</td>
      <td>28.5</td>
      <td>74.4</td>
      <td>79.8</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/1902.09212">HRNet-W48</a></td>
      <td>14.6</td>
      <td>63.6</td>
      <td>75.1</td>
      <td>80.4</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2104.06403">Lite-HRNet-18</a></td>
      <td>0.2</td>
      <td>1.1</td>
      <td>64.8</td>
      <td>71.2</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2104.06403">Lite-HRNet-30</a></td>
      <td>0.3</td>
      <td>1.8</td>
      <td>67.2</td>
      <td>73.3</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2011.04307">EfficientPose-B</a></td>
      <td>1.1</td>
      <td>3.3</td>
      <td>71.1</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2011.04307">EfficientPose-C</a></td>
      <td>1.6</td>
      <td>5.0</td>
      <td>71.3</td>
      <td>-</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td colspan="5"><strong>Transformer-based Methods</strong></td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2012.14214">TransPose-R-A4</a></td>
      <td>8.9</td>
      <td>6.0</td>
      <td>72.6</td>
      <td>78.0</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2012.14214">TransPose-H-S</a></td>
      <td>10.2</td>
      <td>8.0</td>
      <td>74.2</td>
      <td>79.5</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2104.03516">TokenPose-S-v1</a></td>
      <td>2.4</td>
      <td>6.6</td>
      <td>72.5</td>
      <td>78.0</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2104.03516">TokenPose-B</a></td>
      <td>6.0</td>
      <td>13.5</td>
      <td>74.7</td>
      <td>80.0</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2304.05548">DistilPose-S</a></td>
      <td>2.4</td>
      <td>5.4</td>
      <td>71.6</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2304.05548">DistilPose-L</a></td>
      <td>10.3</td>
      <td>21.3</td>
      <td>74.4</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2209.08194">PPT-S</a></td>
      <td>2.0</td>
      <td>6.6</td>
      <td>72.2</td>
      <td>77.8</td>
    </tr>
    <tr>
      <td><a href ="https://arxiv.org/abs/2209.08194">PPT-B</a></td>
      <td>5.6</td>
      <td>13.5</td>
      <td>74.4</td>
      <td>79.6</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td>GTPT-T</td>
      <td>0.7</td>
      <td>2.4</td>
      <td>71.1</td>
      <td>76.6</td>
    </tr>
    <tr>
      <td>GTPT-S</td>
      <td>1.6</td>
      <td>5.4</td>
      <td>73.6</td>
      <td>78.9</td>
    </tr>
    <tr>
      <td>GTPT-B</td>
      <td>3.6</td>
      <td>8.3</td>
      <td>74.9</td>
      <td>80.0</td>
    </tr>
  </tfoot>
</table>

#### Note:
* The resolution of input is 256x192.
* Flip test is used.
* Person detector has person AP of 56.4 on COCO val2017 dataset for top-down methods.

### Results on COCO-WholeBody validation set

<table border="1">
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">Input Size</th>
      <th rowspan="2">GFLOPs</th>
      <th colspan="2">Whole-body</th>
      <th colspan="2">Body</th>
      <th colspan="2">Foot</th>
      <th colspan="2">Face</th>
      <th colspan="2">Hand</th>
    </tr>
    <tr>
      <th>AP</th>
      <th>AR</th>
      <th>AP</th>
      <th>AR</th>
      <th>AP</th>
      <th>AR</th>
      <th>AP</th>
      <th>AR</th>
      <th>AP</th>
      <th>AR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href ="https://arxiv.org/abs/1909.13423">SN</a></td>
      <td>N/A</td>
      <td>272.3</td>
      <td>32.7</td>
      <td>45.6</td>
      <td>42.7</td>
      <td>58.3</td>
      <td>9.9</td>
      <td>36.9</td>
      <td>64.9</td>
      <td>69.7</td>
      <td>40.8</td>
      <td>58.0</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1812.08008">OpenPose</a></td>
        <td>N/A</td>
        <td>451.1</td>
        <td>44.2</td>
        <td>52.3</td>
        <td>56.3</td>
        <td>61.2</td>
        <td>53.2</td>
        <td>64.5</td>
        <td>76.5</td>
        <td>84.0</td>
        <td>38.6</td>
        <td>43.3</td>
    </tr>
    <tr>
        <td><a href ="https://arxiv.org/abs/1611.08050">PAF</a></td>
        <td>512x512</td>
        <td>329.1</td>
        <td>29.5</td>
        <td>40.5</td>
        <td>38.1</td>
        <td>52.6</td>
        <td>5.3</td>
        <td>27.8</td>
        <td>65.6</td>
        <td>70.1</td>
        <td>35.9</td>
        <td>52.8</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1611.05424">AE</a></td>
        <td>512x512</td>
        <td>212.4</td>
        <td>44.0</td>
        <td>54.5</td>
        <td>58.0</td>
        <td>66.1</td>
        <td>57.7</td>
        <td>72.5</td>
        <td>58.8</td>
        <td>65.4</td>
        <td>48.1</td>
        <td>57.4</td>
    </tr>
    <tr>
        <td><a href ="https://arxiv.org/abs/1312.4659">DeepPose</a></td>
        <td>384x288</td>
        <td>17.3</td>
        <td>33.5</td>
        <td>48.4</td>
        <td>44.4</td>
        <td>56.8</td>
        <td>36.8</td>
        <td>53.7</td>
        <td>49.3</td>
        <td>66.3</td>
        <td>23.5</td>
        <td>41.0</td>
    </tr>
    <tr>
        <td><a href ="https://arxiv.org/abs/1804.06208">SimBa.</a></td>
        <td>384x288</td>
        <td>20.4</td>
        <td>57.3</td>
        <td>67.1</td>
        <td>66.6</td>
        <td>74.7</td>
        <td>63.5</td>
        <td>76.3</td>
        <td>73.2</td>
        <td>81.2</td>
        <td>53.7</td>
        <td>64.7</td>
    </tr>
    <tr>
        <td><a href ="https://arxiv.org/abs/1902.09212">HRNet</a></td> 
        <td>384x288</td>
        <td>16.0</td>
        <td>58.6</td>
        <td>67.4</td>
        <td>70.1</td>
        <td>77.3</td>
        <td>58.6</td>
        <td>69.2</td>
        <td>72.7</td>
        <td>78.3</td>
        <td>51.6</td>
        <td>60.4</td>
    </tr>
    <tr>
        <td><a href ="https://arxiv.org/abs/2102.12122">PVT</a></td>
        <td>384x288</td>
        <td>19.7</td>
        <td>58.9</td>
        <td>68.9</td>
        <td>67.3</td>
        <td>76.1</td>
        <td>66.0</td>
        <td>79.4</td>
        <td>74.5</td>
        <td>82.2</td>
        <td>54.5</td>
        <td>65.4</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2211.03375">FastPose50-dcn-si</a></td>
        <td>256x192</td>
        <td>6.1</td>
        <td>59.2</td>
        <td>66.5</td>
        <td>70.6</td>
        <td>75.6</td>
        <td>70.2</td>
        <td>77.5</td>
        <td>77.5</td>
        <td>82.5</td>
        <td>45.7</td>
        <td>53.9</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2007.11858">ZoomNet</a></td>
        <td>384x288</td>
        <td>28.5</td>
        <td>63.0</td>
        <td>74.2</td>
        <td>74.5</td>
        <td>81.0</td>
        <td>60.9</td>
        <td>70.8</td>
        <td>88.0</td>
        <td>92.4</td>
        <td>57.9</td>
        <td>73.4</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2208.11547">ZoomNAS</a></td>
        <td>384x288</td>
        <td>18.0</td>
        <td>65.4</td>
        <td>74.4</td>
        <td>74.0</td>
        <td>80.7</td>
        <td>61.7</td>
        <td>71.8</td>
        <td>88.9</td>
        <td>93.0</td>
        <td>62.5</td>
        <td>74.0</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2212.04246">ViTPose+-S</a></td> 
        <td>256x192</td>
        <td>5.4</td>
        <td>54.4</td>
        <td>-</td>
        <td>71.6</td>
        <td>-</td>
        <td>72.1</td>
        <td>-</td>
        <td>55.9</td>
        <td>-</td>
        <td>45.3</td>
        <td>-</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2212.04246">ViTPose+-H</a></td> <!-- 替换链接为实际的arXiv链接 -->
        <td>256x192</td>
        <td>122.9</td>
        <td>61.2</td>
        <td>-</td>
        <td>75.9</td>
        <td>-</td>
        <td>77.9</td>
        <td>-</td>
        <td>63.3</td>
        <td>-</td>
        <td>54.7</td>
        <td>-</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2303.07399">RTMPose-m</a></td>
        <td>256x192</td>
        <td>2.2</td>
        <td>58.2</td>
        <td>67.4</td>
        <td>67.3</td>
        <td>75.0</td>
        <td>61.5</td>
        <td>75.2</td>
        <td>81.3</td>
        <td>87.1</td>
        <td>47.5</td>
        <td>58.9</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2303.07399">RTMPose-l</a></td> <!-- 替换链接为RTMPose-l论文的实际arXiv链接 -->
        <td>256x192</td>
        <td>4.5</td>
        <td>61.1</td>
        <td>70.0</td>
        <td>69.5</td>
        <td>76.9</td>
        <td>65.8</td>
        <td>78.5</td>
        <td>83.3</td>
        <td>88.7</td>
        <td>51.9</td>
        <td>62.8</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td>GTPT-T</td>
      <td>256x192</td>
      <td>0.8</td>
      <td>54.9</td>
      <td>65.6</td>
      <td>67.6</td>
      <td>75.9</td>
      <td>64.9</td>
      <td>77.4</td>
      <td>75.4</td>
      <td>84.1</td>
      <td>38.3</td>
      <td>49.9</td>
    </tr>
    <tr>
        <td>GTPT-S</td>
        <td>256x192</td>
        <td>2.0</td>
        <td>59.6</td>
        <td>69.9</td>
        <td>71.0</td>
        <td>78.7</td>
        <td>70.4</td>
        <td>82.2</td>
        <td>81.0</td>
        <td>87.6</td>
        <td>45.4</td>
        <td>57.0</td>
    </tr>
    <tr>
        <td>GTPT-B</td>
        <td>256x192</td>
        <td>4.0</td>
        <td>61.7</td>
        <td>71.4</td>
        <td>72.0</td>
        <td>79.5</td>
        <td>73.0</td>
        <td>84.0</td>
        <td>84.2</td>
        <td>89.6</td>
        <td>47.9</td>
        <td>59.3</td>
    </tr>
  </tfoot>
</table>

#### Note:
* Flip test is used.


## Start to use
### 1. Dependencies installation & data preparation
Please refer to [THIS](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) to prepare the environment step by step.

### 2. Model Zoo
Pretrained models are provided in our [model zoo](https://drive.google.com/drive/folders/188-x9NyIhXnU8k5_b1dRQbRRDg4CxKtI?usp=sharing).

### 3. Trainging
```bash=
CUDA_VISIBLE_DEVICES=<GPUs> python tools/train.py --cfg <Config PATH>
```

### 4. Testing
To test the pretrained models performance, please run 

```bash
CUDA_VISIBLE_DEVICES=<GPUs> python tools/test.py --cfg <Config PATH>
```

## Acknowledgement
We acknowledge the excellent implementation from [SimCC](https://github.com/leeyegy/SimCC), [TokenPose
](https://github.com/leeyegy/TokenPose), [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) and [HRFormer](https://github.com/HRNet/HRFormer).

## Citations
If you use our code or models in your research, please cite with:
<!-- ```
@article{wang2023lightweight,
  title={Lightweight Super-Resolution Head for Human Pose Estimation},
  author={Wang, Haonan and Liu, Jie and Tang, Jie and Wu, Gangshan},
  journal={arXiv preprint arXiv:2307.16765},
  year={2023}
}
``` -->
