# 运行固定A–C模型

在ubuntu-server现有项目和已核验环境执行；以下预测不需要标签或A–E等级。默认候选按val冻结，未证明优于CNN。

```bash
ssh ubuntu-server
cd /home/login/data/cle/auto/GPT
/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/predict_ac_model.py /absolute/path/to/image.jpg --kind dino
/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/predict_ac_model.py /absolute/path/to/image_or_folder --kind cnn
/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/predict_ac_model.py /absolute/path/to/image.jpg --kind dino --device cpu
```

模型预处理固定为RGB→双线性256×256→中心224×224→ImageNet normalization；GPU推理batch32/AMP，CPU为FP32。输出class_names、类别索引与三类未校准概率。
现有环境torch2.12.0+cu130/timm1.0.27。权重完整包含encoder，接口不会下载原预训练权重。两模型固定seed45，所有验证/测试原始结果另存，不能只按单seed分数宣称优越。

| kind | 服务器checkpoint | SHA256 |
|---|---|---|
| dino | /home/login/data/cle/auto/GPT/weights/ac_final_evaluation/deployment/dino_s45.pth | 439a9c08765ada39e11962f0763591a31d1b144d31081d88d744f520a2a4b4f7 |
| cnn | /home/login/data/cle/auto/GPT/weights/ac_final_evaluation/deployment/cnn_s45.pth | 11579eda9d0360df558de83cbbcdf72d0ec10c17a0175428bdd738c5958d33d0 |

预测源码在本目录predict_ac_model.py，服务器同名文件位于experiments/；CNN依赖原models/Resnet.py，源码副本在reproducibility_sources/models/。不要将聊天目录当完整服务器项目直接运行而忽略项目根目录/权重路径。
完整部署清单deployment_manifest.json锁两模型。若复制到其他机器，需同时提供匹配运行库、项目模型源码及两权重，并据新根目录调整路径；不需要原始训练图片。

研究模型尚无临床校准/外部独立验证，本脚本不输出临床诊断或自动改标签。
