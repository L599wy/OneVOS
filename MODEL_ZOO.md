## Model Zoo and Results

### Stages:
- `PRE`: the pre-training stage with static images.
- `PRE_YTB_DAV`: the main-training stage with YouTube-VOS and DAVIS. 
- `PRE_YTB_DAV_MOSE`: the main-training stage with YouTube-VOS and DAVIS and MOSE.

### Pre-trained Models
We also provide our trained models you can use directly for inference and evaluation.  The inference script can be referenced [eval_examples_pre_ytb_dav.sh](./eval_examples_pre_ytb_dav.sh) and [eval_examples_pre_ytb_dav_mose.sh](./eval_examples_pre_ytb_dav_mose.sh) 

| Model      | Checkpoints_download | 
|:---------- |:---------:|
| OneVOS (PRE_YTB_DAV)      |    [cks_download](https://pan.baidu.com/s/1Xae6n2KDHfoYHwnefkEdKA?pwd=7867) |
| OneVOS (PRE_YTB_DAV_MOSE)      |    [cks_download](https://pan.baidu.com/s/1YRl1vlKaaw3aJIW5z4IC2w?pwd=7867) |

Note that in order to keep the hyperparameters as consistent as possible, the results we provide may be slightly different from those in the paper, but we guarantee that J&F is consistent. In some datasets, the results are even improved.

Besides, we didn't thoroughly search for hyperparameters, so there may be better choices to get better performance than reported.


### Prediction Results
You can download their predictions for all datasets directly here:
  [OneVOS (PRE_YTB_DAV) All Predictions](https://pan.baidu.com/s/1nGPZOGwW8gS4MCbsCtHP4g?pwd=7867) 
  [OneVOS (PRE_YTB_DAV_MOSE) All Predictions](https://pan.baidu.com/s/1hWiKfckcLbnPRAf0e0MXgQ?pwd=7867)
  or you can download the Predictions for each dataset separately in the table below.


| Datasets    | OneVOS Predictions (PRE_YTB_DAV) | OneVOS Predictions  (PRE_YTB_DAV_MOSE)| 
|:-------------- |:---------:|:---------:|
| DAVIS 16 val   | [predictions](https://pan.baidu.com/s/12oGirufYWNZ8i1hYDUwFJQ?pwd=7867) | [predictions](https://pan.baidu.com/s/1cHYOUrPGGlE6ZbGgHnJ-DA?pwd=7867)| 
| DAVIS 17 val   | [predictions](https://pan.baidu.com/s/1TvzygfKVPz_PkWfTafMm8Q?pwd=7867 ) | [predictions](https://pan.baidu.com/s/1oLbYyoWBJDeaxBHHeZgZzA?pwd=7867)| 
| DAVIS 17 test   | [predictions](https://pan.baidu.com/s/1QjCmJk_zxi-rivP1DIjkUA?pwd=7867) | [predictions](https://pan.baidu.com/s/1X1ZXSY-Ihe3QzSuVSsuxLw?pwd=7867)| 
| YouTubeVOS 19 val   |[predictions](https://pan.baidu.com/s/1J4babgE7DXRTmVzGWn9u-A?pwd=7867) | [predictions](https://pan.baidu.com/s/1Y6LMbXpLv7rFeqMJsi1MyA?pwd=7867)| 
| MOSE |[predictions](https://pan.baidu.com/s/11vgUJTOXGiHRC-Mb-82kpQ?pwd=7867) | [predictions](https://pan.baidu.com/s/1Ql5TFSlbguTYausEfUuGYw?pwd=7867)| 
| LVOS val  |[predictions](https://pan.baidu.com/s/1GJMWCfDxSM1Ek7rIk0kZbQ?pwd=7867)| [predictions](https://pan.baidu.com/s/1x-mOCbLlVLpgqVU7O0oM_w?pwd=7867)| 

Rusults (J&F): 
| Datasets    | OneVOS (PRE_YTB_DAV) | OneVOS (PRE_YTB_DAV_MOSE)| 
|:-------------- |:---------:|:---------:|
| DAVIS 16 val   | 92.7 | 92.8 | 
| DAVIS 17 val   | 88.8 | 88.4 | 
| DAVIS 17 test   | 84.8 | 85.5 | 
| YouTubeVOS 19 val   | 86.1 | 86.3| 
| MOSE | 62.3 | 67.3 | 
| LVOS val  | 67.4 | 71.4 | 