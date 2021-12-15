# Xray image caption generator

I. Model Architecture
![Alt text](https://ibb.co/rkmcCP2 "Model architecture")

II. Running step 
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
  # pass --use_bert to use embeddings vectors from pretrained Biobert
  
  python main.py --mode=train --use_bert --batch_size=32 --n_epochs=40
```
4. Validate model:
```
  python main.py --mode=val --use_bert
```
