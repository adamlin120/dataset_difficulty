from datasets import load_dataset

dataset = load_dataset("banking77")['train']
# Dataset({
#     features: ['text', 'label'],
#     num_rows: 10003
# })

# get only the first 5 labels
dataset = dataset.filter(lambda x: x['label'] < 5)

# random sample 3 examples from each label no group by
data = [
    dataset.filter(lambda x: x['label'] == i).shuffle().select(range(3))
    for i in range(5)
]

# concat all the examples
data = [(d['text'], dataset.features['label'].int2str(d['label']).replace("_", " ")) for ds in data for d in ds]

# shuffle the data
import random
random.shuffle(data)

# Format to "Exapmle {i}: {text}\nLabel {i}: {label}\n""
prompt = ""
null_prompt = ""
null_sentence = "I don't know"
for i, (text, label) in enumerate(data):
    i += 1
    prompt += f"Example {i}: {text}\nLabel {i}: {label}\n"
    null_prompt += f"Example {i}: {null_sentence}\nLabel {i}: {label}\n"

test = load_dataset("banking77")['test']
# random sample 1
test = test.filter(lambda x: x['label'] < 5).shuffle().select(range(1))
text = test['text'][0]
label = test.features['label'].int2str(test['label'][0]).replace('_', ' ')

prompt += f"Example {i+1}: {text}\nLabel {i+1}: {label}\n"
prompt += f"Confidence of the label {i+1} on a scale of 0.0 to 10.0 (Numeric output only):"
null_prompt += f"Example {i+1}: {null_sentence}\nLabel {i+1}: {label}\n"
null_prompt += f"Confidence of the label {i+1} on a scale of 0.0 to 10.0: (Numeric output only):"

print(prompt)
print(null_prompt)

