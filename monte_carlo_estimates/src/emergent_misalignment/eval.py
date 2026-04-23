"""Usage:
python eval.py --model meta-llama/Llama-3.2-1B-Instruct --questions ../../data/emergent_misalignment/first_plot_questions.yaml
"""

import asyncio
import yaml
import json
import torch
import pandas as pd
import random
from joblib import Parallel, delayed

from vllm import LLM, SamplingParams

from judge import OpenAiJudge



def sample(
    llm, conversations, n=100, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1
):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens,
    )

    texts = []
    for messages in conversations:
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    generate_kwargs = {"sampling_params": sampling_params, "use_tqdm": True}

    print("Sampling params: ", generate_kwargs)

    completions = llm.generate(texts, **generate_kwargs)
    answers = [o.text for completion in completions for o in completion.outputs]

    return answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict,
        temperature: float = 1,
        system: str = None,
        judge: str = "gpt-4o",
        **ignored_extra_args,
    ):
        assert len(paraphrases) == 1
        assert system is None

        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {
            metric: OpenAiJudge(judge, prompt)
            for metric, prompt in judge_prompts.items()
        }


    #def get_input(self, n_per_question):
    #    paraphrases = random.choices(self.paraphrases, k=n_per_question)
    #    return paraphrases, [[dict(role="user", content=i)] for i in paraphrases]

    def eval(self, llm, n_per_question):
        paraphrase = self.paraphrases[0]
        batch_size = 100
        conversations = [[dict(role="user", content=paraphrase)] for _ in range(n_per_question // batch_size)]
        answers = sample(llm, conversations, n=batch_size)
        df = pd.DataFrame(
            [
                dict(system=self.system, question=paraphrase, answer=answer, question_id=self.id)
                for answer in answers
            ]
        )
        for score, judge in self.judges.items():
            judge_fn = lambda question, answer : asyncio.run(judge(question=question, answer=answer))
            scores = Parallel(n_jobs=100, verbose=10)(
                delayed(judge_fn)(paraphrase, answer)
                for answer in answers
            )
            df[score] = scores

        return df


def load_model(model):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=False,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=2048,
        max_num_batched_tokens=32000,
    )
    return LLM(**load_kwargs)


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    for question in data:
        if "json" in question["id"]:
            print(f"Skipping {question['id']}...")
            continue
        assert (
            question["type"] == "free_form_judge_0_100"
        ), "We currently only open sourced the judge for free_form_judge_0_100 questions"
        questions.append(Question(**question))
    return questions


def main(model, questions, n_per_question=100, output="eval_result.csv"):
    """Evaluate a model on all questions form the evaluation yaml file"""
    llm = load_model(model)
    questions = load_questions(questions)

    outputs = []
    for i, question in enumerate(questions):
        print(f"Question idx: {i}, Question obj: {question.__dict__}")
        outputs.append(question.eval(llm, n_per_question))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
