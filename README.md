# aim-v2-training
Training an AIMv2 Model using just the public datasets COYO and DFN-2B making use of the image alt-texts and some sort of synthetic caption.

## Environment Setup
```
conda create -n aim-v2-training
conda activate aim-v2-training
```

```
git clone git@github.com:apple/ml-aim.git
pip install -r requirements.txt
pip install 'git+https://github.com/apple/ml-aim.git#subdirectory=aim-v1'
pip install 'git+https://github.com/apple/ml-aim.git#subdirectory=aim-v2'
```

## Downloading the Datasets

Download datasets to `/mnt/shared-slurm/datasets` or wherever it is that you've decided to store datasets for all your nodes to access.

We'll only be using DFN-2B and COYO. You can manually follow the instructions to download them here:
```
https://huggingface.co/datasets/apf1/datafilteringnetworks_2b
https://github.com/kakaobrain/coyo-dataset/tree/main/download
```

Or you can use `img2dataset` to conveniently download them. We will convert these files to `webdataset` format.

Download the COYO parquet files likeso:
```
cd /mnt/shared-slurm/datasets
mkdir coyo-700m && cd coyo-700m
for i in {00000..00127}; do wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-$i-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet; done
cd ..
```

Convert the parquet files to webdataset format:
```
img2dataset --url_list coyo-700m --input_format "parquet"\
         --url_col "url" --caption_col "text" --output_format webdataset\
           --output_folder coyo-700m-webdataset --processes_count 16 --thread_count 64 --image_size 384\
            --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
             --save_additional_columns '["clip_similarity_vitb32","clip_similarity_vitl14","nsfw_score_opennsfw2","nsfw_score_gantman","watermark_score","aesthetic_score_laion_v2"]' --enable_wandb False
```

Download the [DFN-2B](https://arxiv.org/pdf/2309.17425) Dataset likeso:
```
```
## Launching a Train Job
I'm using Slurm in order to orchestrate my nodes and GPU's. If you have you're own thing you'll have to create a launcher your own way.