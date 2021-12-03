# Xray image caption generation

1. Create conda environment:
```
    conda create -n xray_caption python=3.8
    conda activate xray_caption
    pip3 install -r requirements.txt
```
2. Download dataset and extract all images to dataset/images folder
  Download from here: https://drive.google.com/file/d/16yDwrINwgOVPUgWbs3JdHqioPo1NsTrt/view?usp=sharing
3. Train the model:
```
  python main.py --mode=train --use_bert=True --batch_size=32 --n_epochs=10
```
