# LLM_Lora_Classification_In_Google_Colab
## Install
```
!pip install torch transformers numpy pandas more-itertools scikit-learn tqdm typed-argument-parser accelerate bitsandbytes peft
```

## Prepare Dataset
```
python src/prepare.py --text_column text --label_column label
```

## Train 
```
!accelerate launch src/train.py --model_name cyberagent/open-calm-1b
```