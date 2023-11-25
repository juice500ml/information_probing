# Information Probing

## Environment setup
```bash
# Conda env
conda create -p ./envs
conda activate ./envs
conda install python=3.9

# Pip install
pip install -r requirements.txt
```

## Prepare CommonPhone dataset
```bash
wget -O /path/to/save/cp-1-0.tgz --quiet \
    'https://zenodo.org/record/5846137/files/cp-1-0.tgz?download=1'
tar -xzf /path/to/save/cp-1-0.tgz

# We use --min_classwise_count to filter out classes with too little samples.
# We tested 100, 400, and 1600.
python3 prepare_commonphone.py \
    --dataset_path /untarred/path/CP \
    --output_path /path/to/save/csvs \
    --min_classwise_count INTEGER
```

## Run experiments
```bash
# Word classification + Full finetuning
python3 classification_probing.py \
    --dataset_path commonphone_words.csv.gz \
    --tuning_type finetune --layer_index 0

# Phoneme classification + Partial finetuning
python3 classification_probing.py \
    --dataset_path commonphone_phonemes.csv.gz \
    --tuning_type finetune --layer_index 6

# Word classification + Linear probing
python3 classification_probing.py \
    --dataset_path commonphone_words.csv.gz \
    --tuning_type linear --layer_index 0

# Word classification + Linear probing with MLP
python3 classification_probing.py \
    --dataset_path commonphone_words.csv.gz \
    --tuning_type linear --layer_index 0 \
    --probe_type mlp1 --probe_hidden_dim 100
```
