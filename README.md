# CorpusKorea_KNUAI

## Instructions
![image](https://user-images.githubusercontent.com/31720981/140687490-9adafed7-f08a-40c2-91a6-51b85a464bb2.png)
1. Clone this repo.
2. Run the docker image.
3. `sh ./run.sh` to execute evaluation.

## Notice
- We have secondary link for [Weight files](https://drive.google.com/drive/folders/1P8H6tUI0uEV3wK9ZM_NIvhHYsn8c600s?usp=sharing), you can download weight via the link.
- Make sure the name format of the `test` file **MUST** include one of **(CoLA|cola), (COPA|copa), (WiC|wic), (Boolq|boolq)**.
  - eg) `test_cola.tsv` `CoLA_test.tsv` `foo_bar_CoLA.tsv` (O)
  - eg) `test1.tsv` `evaluation_1.tsv` (X)
- We recommand **not to change** the name of `state files`, if you want to, please leave `task_prefix` of files.
- See [`evaluation.py`](https://github.com/KNUAI2021/CorpusKorea_KNUAI/blob/main/evaluation.py#L373) to check implementation.

Before you run, install nvidia-docker&&docker!!!

Just following this command line

# 1st: git down
```
git clone https://github.com/KNUAI2021/CorpusKorea_KNUAI
```
# 2nd: directory setting
```
cd CorpusKorea_KNUAI
```
# 3rd: put your test_dataset in "CorpusKorea_KNUAI/corpus" directory!!!

# 4th: build docker image
```
docker build -t knuai .
```
# 5th: make container
```
nvidia-docker run -it --name knuai knuai
```
# 6th: run python evaluation.py --device cuda:0 --state_dir ./weights --test_file_dir ./corpus --output_dir ./
```
sh ./run.sh
```
