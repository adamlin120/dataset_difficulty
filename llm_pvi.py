import os
import pprint
import colorama
import json
import random

import anthropic
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

random.seed(42)

single_exempler_prompt = "Premise: {}\nHypothesis: {}\nLabel: {}"


def create_prompt(data, test_sample, null_sentence="N/A"):
    prompt = ""
    null_prompt = ""
    # random sample examples from the training set
    for i, (_, datum) in enumerate(data.sample(8).iterrows()):
        i += 1
        premise = datum["premise"]
        hypothesis = datum["hypothesis"]
        label = datum["gold"]
        prompt += f"Example {i}:\n{single_exempler_prompt.format(premise, hypothesis, label)}\n"
        null_prompt += f"Example {i}:\n{single_exempler_prompt.format(null_sentence, null_sentence, label)}\n"

    premise = test_sample["premise"]
    hypothesis = test_sample["hypothesis"]
    prompt += f"Example {i + 1}:\nPremise: {premise}\nHypothesis: {hypothesis}\n"
    null_prompt += (
        f"Example {i + 1}:\nPremise: {null_sentence}\nHypothesis: {null_sentence}\n"
    )

    instruction_prompt = """What is the predict label and probability distribution for each class for the example 9.\nPlease answer in a json format of {"class": "...", "probs": {"entailment": float, "neutral": float, "contradiction": float}}"""

    prompt += f"{instruction_prompt}\n"
    null_prompt += f"{instruction_prompt}\n"

    return prompt.strip(), null_prompt.strip()


def get_likelihood(prompt, gold):
    full_prompt = f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}{{\"class\":"

    temperature = 0.0

    while True:
        resp = c.completion(
            prompt=full_prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT, "\n"],
            model="claude-v1.3",
            max_tokens_to_sample=100,
            temperature=temperature,
        )
        prediction = resp["completion"].strip()
        prediction = "{\"class\":" + prediction
        try:
            prediction = json.loads(prediction)
            return prediction
        except json.decoder.JSONDecodeError:
            print(f"JSONDecodeError: {prediction}")
            temperature += 0.1
            print(f"Retrying with temperature {temperature}")


def calculate_pvi(few_shot_gold_likelihood, null_likelihood):
    pvi = -np.log2(null_likelihood) + np.log2(few_shot_gold_likelihood)
    return pvi


def main():
    dataset_name = "alisawuffles/WANLI"

    data = load_dataset(dataset_name)
    train = data["train"].to_pandas()
    test = data["test"].to_pandas()

    for index, test_sample in tqdm(test.iterrows(), total=len(test), ncols=0):
        prompt, null_prompt = create_prompt(train, test_sample)

        prob_dist = get_likelihood(
            prompt, test_sample["gold"]
        )
        null_prob_dist = get_likelihood(
            null_prompt, test_sample["gold"]
        )

        gold_label = test_sample["gold"]
        gold_likelihood = prob_dist["probs"][gold_label]
        null_gold_likelihood = null_prob_dist["probs"][gold_label]

        pvi = calculate_pvi(gold_likelihood, null_gold_likelihood)

        # Green
        print(
            colorama.Fore.GREEN,
            f"Gold label: {gold_label} | Gold likelihood: {gold_likelihood} | Null likelihood: {null_gold_likelihood} | PVI: {pvi}"
        )
        # MAGENTA
        print(colorama.Fore.MAGENTA, prob_dist)
        # CYAN
        print(colorama.Fore.CYAN, null_prob_dist)
        # RESET
        print(colorama.Fore.RESET)

        # test.loc[index, "prob_dist"] = prob_dist
        # test.loc[index, "null_prob_dist"] = null_prob_dist
        test.loc[index, "PVI"] = pvi
        # save periodically
        if index % 10 == 0:
            test.to_csv(f"claudev1.3_wanli.csv", index=False)
    test.to_csv(f"claudev1.3_wanli.csv", index=False)


if __name__ == "__main__":
    main()
