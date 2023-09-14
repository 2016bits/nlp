# Fact checking with LLM and tools

This code based on ["Fact-Checking Complex Claims with Program-Guided Reasoning"](https://github.com/mbzuai-nlp/ProgramFC/tree/main)

## Installation

It is recommended to install Python 3.8.0

Install all required Python packages using:
```bash
pip install -r requirements.txt
```

## Data Preparation

### FEVER dataset

First, create a folder named 'data/FEVER'
Second, download FEVER dataset from [FEVER](https://fever.ai/dataset/fever.html)
Third, preprocess data on root directory:
```bash
sh scripts/process.sh
```

### 2017 wikipedia dumps

First, create a folder named 'data/Wikipedia'
Second, download wikipedia dumps from [Wikipedia dumps](https://github.com/sheffieldnlp/naacl2018-fever/tree/master)
Download raw wikipedia files:
```bash
sh scripts/download-raw-wiki.sh
```

Process wikipedia data and store in database:
```bash
sh scripts/process-wiki.sh
```

### CHEF dataset

First, create a folder named 'data/CHEF'
Second, download CHEF dataset from [CHEF](https://github.com/THU-BPM/CHEF)

## Model Preparation

download LLM model Aquila and Flan-T5-xl

## Main experiment

First, generate reasoning programs using LLM. Second, parse generated programs and execute programs line by line.

### Generate programs

On the root directory
Create folder named 'logs' used for storing log files
Create folder named 'results' used for storing generated programs
Run:
```bash
sh generate.sh
```

Example of each sample with generated programs:
```json

```

### Execute programs

Run:
```bash
sh execute.sh
```

The codes, logs and results will be stored in directory 'outputs'.

## Ablation exeperiments

There are 5 ablation exeperiments:
1. Directly use T5 to verify the claims. (only T5 without evidence)
2. Use NER and similarity match to retrieve evidences, then use T5 to verify.   (Retrieval tools + T5)
3. Directly use LLM to verify the claims.    (only LLM without evidence)
4. Use LLM to generate evidence, then use LLM to verify.    (only LLM with evidence)
5. Use LLM to generate evidence, then use T5 to verify. (only LLM with evidence)

### Only T5 without evidence

```

```