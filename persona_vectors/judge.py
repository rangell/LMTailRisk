import os
import json
import math
import time
from pathlib import Path
import uuid
from typing import List
from joblib import Parallel, delayed

from persona_vectors.config import setup_credentials


MAX_BATCH_SIZE = 5000

# Set up credentials and environment
config = setup_credentials()


class OpenAiJudge:
    """OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""

    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model = model
        assert eval_type in ["0_100", "0_10", "binary", "binary_text"], (
            "eval_type must be either 0_100 or binary"
        )
        self.eval_type = eval_type

        if self.eval_type == "0_100":
            self.aggregate_score = self._aggregate_0_100_score
        elif self.eval_type == "0_10":
            self.aggregate_score = self._aggregate_0_10_score
        elif self.eval_type == "binary":
            self.aggregate_score = self._aggregate_binary_score
        elif self.eval_type == "binary_text":
            self.aggregate_score = self._aggregate_binary_text_score
        else:
            raise ValueError(f"Invalid eval_type: {self.eval_type}")

        self.prompt_template = prompt_template

    async def judge(self, client, **kwargs):
        messages = [dict(role="user", content=self.prompt_template.format(**kwargs))]
        if self.eval_type == "binary_text":
            response_text = await self.query_full_text(client, messages)
            score = self.aggregate_score(
                response_text
            )  # aggregate_score is _aggregate_binary_text_score
        else:
            logprobs = await self.logprob_probs(client, messages)
            score = self.aggregate_score(
                logprobs
            )  # aggregate_score is one of the other three
        return score

    async def logprob_probs(self, client, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        while True:
            try:
                completion = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=10000,
                    temperature=0,
                    logprobs=True,
                    top_logprobs=20,
                    seed=0,
                    timeout=60.0,
                )
                break
            except Exception as e:
                continue
                print(f"Exception: {e}")

        try:
            ## for instruct models
            # logprobs = completion.choices[0].logprobs.content[0].top_logprobs

            # for reasoning models
            logprobs = completion.choices[0].logprobs.content[-2].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))

        return result

    async def query_full_text(self, client, messages) -> str:
        """Requests a full text completion. Used for binary_text eval_type."""
        completion = await client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, seed=0
        )
        try:
            return completion.choices[0].message.content
        except (IndexError, AttributeError):
            return ""

    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explcitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    def _aggregate_0_10_score(self, score: dict) -> float:
        #   NOTE: we check for refusals explcitly.
        #   If "REFUSAL" has the highest probability, we return None.
        #   Otherwise, we aggregate scores from 0-9.
        if "REFUSAL" in score and score["REFUSAL"] > max(
            score.get(str(i), 0) for i in range(10)
        ):
            return None

        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 9:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:  # Similar threshold as 0_100
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    def _aggregate_binary_score(self, score: dict) -> float:
        #   NOTE: we check for refusals explicitly.
        #   If "REFUSAL" has a higher probability than both "YES" and "NO", we return None.
        #   Otherwise, calculates a score P(YES) / (P(YES) + P(NO)).
        #   Returns None if the combined probability of YES and NO is below a threshold (0.25).

        yes_prob = score.get("YES", 0.0)
        no_prob = score.get("NO", 0.0)
        refusal_prob = score.get("REFUSAL", 0.0)

        # If REFUSAL has a higher probability than both YES and NO, consider it a refusal.
        if refusal_prob > yes_prob and refusal_prob > no_prob:
            return None

        denominator = yes_prob + no_prob

        # If the combined probability of YES and NO is too low (e.g., model outputted something else,
        # or was not confident in YES/NO), return None.
        if (
            denominator < 0.25
        ):  # Using 0.25 to be consistent with other aggregation methods
            return None

        return yes_prob / denominator

    def _aggregate_binary_text_score(self, response_text: str) -> bool:
        if "<answer>REFUSAL</answer>" in response_text:
            return None
        elif "<answer>NO</answer>" in response_text:
            return 0
        elif "<answer>YES</answer>" in response_text:
            return 1
        return None  # Invalid response

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)

    def batch_judge(self, questions: List[str], answers: List[str]) -> List[float]:
        assert len(questions) == len(answers)
        if len(questions) <= MAX_BATCH_SIZE:
            return self._batch_judge(questions, answers)
        else:
            num_batches = ((len(questions) - 1) // MAX_BATCH_SIZE) + 1
            chunk_size = math.ceil(len(questions) / num_batches)
            batched_judge_scores = Parallel(n_jobs=10, prefer="threads")(
                delayed(self._batch_judge)(
                    questions[i * chunk_size : (i + 1) * chunk_size],
                    answers[i * chunk_size : (i + 1) * chunk_size],
                )
                for i in range(num_batches)
            )

            judge_scores = [s for sublist in batched_judge_scores for s in sublist]
            assert len(judge_scores) == len(questions)
            return judge_scores

    def _batch_judge(self, questions: List[str], answers: List[str]) -> List[float]:
        try:
            # Evaluate this judge template on all question answer pairs
            uid, infile, outfile = self._create_tmp_file()
            custom_ids = self._write_prompts_to_file(uid, infile, questions, answers)

            # Stores the responses in outfile
            responses = self._get_responses(uid, infile, outfile)

            # Order judge scores with question-answer pairs
            judge_scores = [responses.get(k, 0) for k in custom_ids]

            assert len(judge_scores) == len(questions)
        finally:
            os.remove(outfile)
            os.remove(infile)

        return judge_scores

    def _create_tmp_file(self) -> str:
        while True:
            uid = str(uuid.uuid4())
            infile = f"batch_input_{uid}.jsonl"
            file_path = Path(infile)
            outfile = f"batch_output_{uid}.jsonl"
            if not file_path.exists():
                file_path.touch()
                break
        return uid, infile, outfile

    def _write_prompts_to_file(
        self, uid: str, infile: str, questions: List[str], answers: List[str]
    ) -> List[str]:
        custom_ids = []
        with open(infile, "w", encoding="utf-8") as f:
            for idx, (q, a) in enumerate(zip(questions, answers)):
                body = {
                    "model": self.model,
                    "temperature": 0.0,
                    "messages": [
                        {
                            "role": "user",
                            "content": self.prompt_template.format(
                                question=q, answer=a
                            ),
                        }
                    ],
                    "min_tokens": 1,
                    "max_tokens": 1,
                }
                line = {
                    "custom_id": f"{idx}",
                    "body": body,
                }
                f.write(json.dumps(line) + "\n")
                custom_ids.append(f"{idx}")
        return custom_ids

    def _get_responses(self, uid: str, infile: str, outfile: str) -> None:
        together_client = Together()  # uses TOGETHER_API_KEY env var
        file_resp = together_client.files.upload(
            file=infile,
            purpose="batch-api",  # important for Batch API
        )

        print("Uploaded file:", file_resp.id)

        batch = together_client.batches.create_batch(
            file_id=file_resp.id,
            endpoint="/v1/chat/completions",
        )
        print("Created batch:", batch.id)

        while True:
            try:
                batch_status = together_client.batches.get_batch(batch.id)
                print(f"Status ({uid}): {batch_status.status}")

                if batch_status.status in (
                    "COMPLETED",
                    "CANCELLED",
                    "FAILED",
                    "EXPIRED",
                ):
                    break
                time.sleep(15)  # tune based on how spammy you want to be
            except Exception as e:
                print(f"Exception raised: {e}")

        if batch_status.status != "COMPLETED":
            raise RuntimeError(
                f"Batch did not complete successfully: {batch_status.status}"
            )

        together_client.files.retrieve_content(
            id=batch_status.output_file_id,
            output=outfile,
        )

        print("Downloaded results to", outfile)

        judge_responses = {}

        with open(outfile, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)
                cid = obj.get("custom_id")
                try:
                    resp = int(
                        obj["response"]["body"]["choices"][0]["message"]["content"]
                    )
                except Exception as e:
                    print("Exception!: ", e, obj)
                    print("Replacing with: '0'")
                    resp = 0

                judge_responses[cid] = resp

        return judge_responses
