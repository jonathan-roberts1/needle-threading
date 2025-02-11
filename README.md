# Needle Threading: Can LLMs Follow Threads through Near-Million-Scale Haystacks?

### [Project Page](https://needle-threading.github.io) | [Paper](https://arxiv.org/abs/2411.05000) | [Data](https://huggingface.co/datasets/jonathan-roberts1/needle-threading) | [Code](https://github.com/jonathan-roberts1/needle-threading/)

### Jonathan Roberts, Kai Han, Samuel Albanie

As the context limits of Large Language Models (LLMs) increase, the range of possible applications and downstream functions broadens. Although the development of longer context models has seen rapid gains recently, our understanding of how effectively they use their context has not kept pace. 

To address this, we conduct a set of retrieval experiments designed to evaluate the capabilities of 17 leading LLMs, such as their ability to follow threads of information through the context window. 

Strikingly, we find that many models are remarkably thread-safe: capable of simultaneously following multiple threads without significant loss in performance. Still, for many models, we find the effective context limit is significantly shorter than the supported context length, with accuracy decreasing as the context window grows.

#### This repository contains a summary of the key findings from our paper, as well as instructions and example code to both access our experimental data and evaluate on it.
### Contents
  - [News](#news)
  - [Key Insights](#key-insights)
  - [Overview](#overview)
  - [Data](#data)
  - [Evaluation](#evaluation)

## News
🎉 **[08/11/24]** Initial release!

## Key Insights
- We find significant differences in the tokenization of long-context LLMs, especially when processing abstract text (numbers or UUIDs)
- Like previous works, at long context lengths, the retrieval precision of frontier LLMs decreases towards the middle of the context
- Clustering needles has little effect on performance when tasked with retrieving specific needles
- But noticeably increases performance when retrieving all needles meeting a condition
- Most LLMs achieve higher accuracies when retrieving threads moving forwards through the context versus backwards directed threads
- The evaluated LLMs show proficiency at keeping track of multiple threads simultaneously
- Across all our tasks, the retrieval performance of frontier models decreases at longer context lengths

## Overview

### Context Length Comparison
<img src="images/context_length_comparison_llama_essay.png" alt="Context Length Comparison" width="75%" />

### Tasks
<img src="images/schematic.png" alt="Tasks" width="75%" />

### Frontier models are remarkably thread-safe
<img src="images/thread_safe.png" alt="Frontier models are remarkably thread-safe" width="75%" />

### Aggregated Results
<img src="images/table.png" alt="Aggregated Results" width="75%" />

### Effective Context Length
**Single Needle** left, **Multiple Needles** right
<div style="display: flex; justify-content: space-between;">
    <img src="images/single_needle_contours.png" alt="Single Needle Contours" width="49%" />
    <img src="images/multi_needle_contour.png" alt="Multi Needle Contour" width="49%" />
</div>


## Data

Our experimental data can be accessed either using the HuggingFace datasets library or by manual download.

### Option 1: HuggingFace datasets
```python
from datasets import load_dataset

# task splits can be downloaded separately:
# splits = ['Single_Needle', 'Multi_Needle', 'Conditional_Needle', 'Single_Thread', 'Multi_Thread']
single_needle_dataset = load_dataset("jonathan-roberts1/needle-threading", split='Single_Needle')

"""
Dataset({
    features: ['id', 'haystack', 'keys', 'values', 'question', 'context_length', 'num_kv_pairs',
    'repeat_number', 'needle_depth', 'num_needles', 'needle_placement', 'conditional_character',
    'thread_length', 'thread_direction', 'num_threads'],
    num_rows: 660
})
Note the units of context_length are number of characters.
"""

# query individual questions
single_needle_dataset[5] # e.g., the 6th element
"""
{'id': 5, 'haystack': '{"e3e70682-c209-4cac-629f-6fbed82c07cd": "f728b4fa-4248-5e3a-0a5d-2f346baa9455",
"eb1...": "964a870c-7c87-9b74-1d87-8f9f9cdf5a86"}', 'keys': '247a8333-f7b0-b7d2-cda8-056c3d15eef7',
'values': '1759edc3-72ae-2244-8b01-63c1cd9d2b7d', 'question': 'Extract the value corresponding to
the specified key in the JSON object. Key: "247a83...-cda8-056c3d15eef7"\n Corresponding value: ',
'context_length': 2000, 'num_kv_pairs': 25, 'repeat_number': 0, 'needle_depth': '50', 'num_needles': 1,
'needle_placement': 'depthwise', 'conditional_character': 'N/A', 'thread_length': 1,
'thread_direction': 'N/A', 'num_threads': 0}
"""
```


### Option 2: Manual download

Directly downloading image files and question data from the needle-threading HuggingFace repository into the ```data``` directory in this repo.
```
cd data
wget "https://huggingface.co/datasets/jonathan-roberts1/needle-threading/resolve/main/json_data.zip?download=true" -O json_data.zip
unzip json_data.zip && rm json_data.zip
```
#### Expected structure
```
├── data
    ├── json_data
        ├── Single_Needle.json
        ├── Multiple_Needles.json
        ├── Conditional_Needles.json
        ├── Single_Threads.json
        ├── Multi_Threads.json
```

Note: ```data_json/``` needs to be downloaded.


## Evaluation

### Single Needle example inference using Gemini 1.5 Flash
```python
from datasets import load_dataset
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from tqdm import tqdm
import pandas as pd
from ast import literal_eval

project_id = "YOUR_PROJECT_ID"
region = "REGION" # eg "us-central1"
model_name = "gemini-1.5-flash-preview-0514"

dataset = load_dataset("jonathan-roberts1/needle-threading", 
    split="Single_Needle") # optional: set cache_dir="PATH/TO/MY/CACHE/DIR"

# dataframe to store results
output_df = pd.DataFrame(columns=["Question_ID", "Output", "Answer", "Correct?"])

# Initialise generative multimodal model
vertexai.init(project=project_id, location=region)
generative_multimodal_model = GenerativeModel(model_name)
config = {
        "max_output_tokens": 100,
        "temperature": 0,
        "top_k": 1
    }

# Iterate over questions
for idx, item in tqdm(enumerate(dataset)):

    if idx == 10:
        break

    question = item['question']
    haystack = item['haystack']
    # see our paper the specific prompt structure we use
    prompt = haystack + '\n' + question
    model_response = generative_multimodal_model.generate_content(contents=prompt,
                                                            generation_config=config).text
    model_response.strip()
    model_response = literal_eval(model_response)

    answer = item['values']

    # evaluate answer
    model_answer = model_response[0:len(str(answer))]
    correct = model_answer == str(answer)

    # store results
    results_row = {"Question_ID": item['id'], "Output": model_response,
                    "Answer": answer, "Correct?": correct}
    output_df = pd.concat([output_df, pd.DataFrame([results_row])], ignore_index=True)

    # save output
    #output_df.to_csv("PATH/TO/SAVE/DIR", index=False)

# compute accuracy
accuracy = output_df["Correct?"].mean()
    
print(f"{model_name} Single_Needle: {100 * accuracy:.2f}%")
```

## Citation
If you found our work useful in your own research, please consider citing our paper:
```latex
@article{roberts2024needle,
  title={Needle Threading: Can LLMs Follow Threads through Near-Million-Scale Haystacks?},
  author={Roberts, Jonathan and Han, Kai and Albanie, Samuel},
  journal={arXiv preprint arXiv:2411.05000},
  year={2024}
}
```










