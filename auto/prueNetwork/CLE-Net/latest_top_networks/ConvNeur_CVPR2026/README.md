<h1 align="center">ConvNeur</h1>

<h3 align="center">Efficiency Follows Global-Local Decoupling</h3>

## 🔭Overview

<img src="ConvNeur.svg" alt="ConvNeur" style="zoom:200%;" />

## 🗝️Quick start on ImageNet-1K dataset

1. Data preparation:

   - Download ImageNet-1K dataset from [Here](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

   - Arrange them into the following format:

     ```
     data
     |- imagenet
        |- train
           |- n01440764
              |- n01440764_18.JPEG
              |- ...
           |- n01443537
           |- ...
        |- val
           |- (same as above)
     ```
     
   
2. Installation:

   - Create and activate a new conda environment:

     ```bash
     conda create -n convneur python=3.10
     conda activate convneur
     ```

   - Install PyTorch according to CUDA or ROCm version (taking CUDA 13.0 as an example here): 

     ```bash
     pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
     ```
   
     👉 **Note that** we do not recommend you to use torch versions lower than **2.0.0**.
   
   - Install other dependencies:
   
     ```bash
     pip install -r requirements.txt
     ```
   
3. Train:

   ```bash
   python -m torch.distributed.run --nproc_per_node=8 main.py --model_scale m1 --batch_size 64 --lr 4e-3 --update_freq 8 --data_path data/imagenet --output_dir ./work_dir
   ```

5. Benchmark:

   - Please download the weight of our **ConvNeur** and put it under **project_path/ckpts/**.

   - Test:

     ```bash
     python main.py --eval True --model_scale m1 --finetune ckpts/convneur_m1_in1k_ema.pth --batch_size 32 --data_path data/imagenet
     ```

## 🤝Acknowledgments
This project is based on [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [Titans](https://github.com/lucidrains/titans-pytorch). Thanks for their excellent works!!
