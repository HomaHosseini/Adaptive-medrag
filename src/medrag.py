import os
import re
import json
import tqdm
import torch
import time, math
import argparse
import requests
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
sys.path.append("src")
from utils import RetrievalSystem, DocExtracter
from template import *

from utils import robust_parse_json
from config import config

openai.api_type = openai.api_type or os.getenv("OPENAI_API_TYPE") or config.get("api_type")
openai.api_version = openai.api_version or os.getenv("OPENAI_API_VERSION") or config.get("api_version")
openai.api_key = openai.api_key or os.getenv('OPENAI_API_KEY') or config["api_key"]

# ------------------------------------------------------------------
# Custom base URL support (needed for OpenAI-SDK-compatible gateways like
# Liara/Chabokan/Arvan's OpenAI-style routes -- base_url points at their
# endpoint instead of OpenAI's own). IMPORTANT: which env var actually
# takes effect depends on which openai package version is installed --
# this project intentionally stays on the legacy (openai<1.0) package, so
# set OPENAI_API_BASE in your .env, NOT OPENAI_BASE_URL:
#
#   openai<1.0 (the branch this project actually uses):
#       reads OPENAI_API_BASE (see line below, unchanged from upstream)
#   openai>=1.0 (NOT used here, kept only for forward-compatibility):
#       reads OPENAI_BASE_URL (via _openai_base_url just below)
#
# Setting the wrong one of these two variables will silently fall back to
# OpenAI's real endpoint instead of erroring -- if requests seem to be
# hitting OpenAI instead of your gateway, this is the first thing to check.
# ------------------------------------------------------------------
_openai_base_url = os.getenv("OPENAI_BASE_URL") or config.get("base_url")  # only read by the openai>=1.0 branch below

if openai.__version__.startswith("0"):
    # ACTIVE BRANCH for this project. Reads OPENAI_API_BASE (not OPENAI_BASE_URL).
    openai.api_base = openai.api_base or os.getenv("OPENAI_API_BASE") or config.get("api_base")
    if openai.api_type == "azure":
        openai_client = lambda **x: openai.ChatCompletion.create(**{'engine' if k == 'model' else k: v for k, v in x.items()})["choices"][0]["message"]["content"]
        # Raw variant: returns the full response object (not just the content
        # string) so callers can also read .usage and .choices[0].logprobs.
        openai_raw_client = lambda **x: openai.ChatCompletion.create(**{'engine' if k == 'model' else k: v for k, v in x.items()})
    else:
        openai_client = lambda **x: openai.ChatCompletion.create(**x)["choices"][0]["message"]["content"]
        openai_raw_client = lambda **x: openai.ChatCompletion.create(**x)
else:
    # NOT the active branch for this project (staying on openai<1.0 to avoid
    # breaking other call sites) -- kept here only in case of a future
    # upgrade. Reads OPENAI_BASE_URL, not OPENAI_API_BASE.
    if openai.api_type == "azure":
        openai.azure_endpoint = openai.azure_endpoint or os.getenv("OPENAI_ENDPOINT") or config.get("azure_endpoint")
        openai_client = lambda **x: openai.AzureOpenAI(
            api_version=openai.api_version,
            azure_endpoint=openai.azure_endpoint,
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content
        openai_raw_client = lambda **x: openai.AzureOpenAI(
            api_version=openai.api_version,
            azure_endpoint=openai.azure_endpoint,
            api_key=openai.api_key,
        ).chat.completions.create(**x)
    else:
        openai_client = lambda **x: openai.OpenAI(
            api_key=openai.api_key,
            base_url=_openai_base_url,  # None = OpenAI's default endpoint; set = Liara/Chabokan/other gateway
        ).chat.completions.create(**x).choices[0].message.content
        openai_raw_client = lambda **x: openai.OpenAI(
            api_key=openai.api_key,
            base_url=_openai_base_url,  # NEW
        ).chat.completions.create(**x)

class MedRAG:

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, follow_up=False, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None, corpus_cache=False, HNSW=False):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None
        self.llm_call_count = 0
        self.llm_call_history = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW)
        else:
            self.retrieval_system = None
        self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}
        if self.llm_name.split('/')[0].lower() == "arvan":
            self.model = '/'.join(self.llm_name.split('/')[1:])
            self.api_key = os.environ["ARVAN_API_KEY"]
            self.api_endpoint = os.environ["ARVAN_ENDPOINT"]

            self.max_length = 32768
            self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif self.llm_name.split('/')[0].lower() == "openai":
            # Was: self.llm_name.split('/')[-1] -- only kept the LAST path
            # segment, silently dropping vendor prefixes like "meta-llama/"
            # in multi-segment model IDs (needed for gateways such as Liara/
            # Chabokan, e.g. "OpenAI/meta-llama/llama-3.1-8b-instruct").
            # Matches the Arvan branch's (correct) handling above.
            self.model = '/'.join(self.llm_name.split('/')[1:])
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-3" in self.model.lower():
                # Matches the local-model branch's numbers below. Llama-3.1
                # and later support a 128k context window; earlier 3.x
                # variants (plain "llama-3") are 8k.
                if any(v in self.model for v in (".1", ".2", ".3")):
                    self.max_length = 131072
                    self.context_length = 128000
                else:
                    self.max_length = 8192
                    self.context_length = 7168
            else:
                # SAFETY NET: this branch previously left max_length/
                # context_length completely UNSET for any model whose name
                # didn't match "gpt-3.5"/"gpt-4" -- causing a downstream
                # AttributeError the first time a different model (e.g. a
                # Llama variant routed through an OpenAI-compatible gateway)
                # was used. Always set something conservative rather than
                # nothing, so a new/unrecognized model degrades gracefully
                # (shorter context) instead of crashing.
                print(f"NOTE: no known context-length profile for model "
                      f"'{self.model}' in the openai branch -- defaulting to "
                      f"a conservative 8k context. Add a case above if this "
                      f"model supports more.")
                self.max_length = 8192
                self.context_length = 7168
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif "gemini" in self.llm_name.lower():
            import google.generativeai as genai
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.model = genai.GenerativeModel(
                model_name=self.llm_name.split('/')[-1],
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                }
            )
            if "1.5" in self.llm_name.lower():
                self.max_length = 1048576
                self.context_length = 1040384
            else:
                self.max_length = 30720
                self.context_length = 28672
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.max_length = 2048
            self.context_length = 1024
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
            if "mixtral" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.context_length = 3072
            elif "llama-3" in llm_name.lower():
                self.max_length = 8192
                self.context_length = 7168
                if ".1" in llm_name or ".2" in llm_name:
                    self.max_length = 131072
                    self.context_length = 128000
            elif "meditron-70b" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/meditron.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
                self.context_length = 3072
                self.templates["cot_prompt"] = meditron_cot
                self.templates["medrag_prompt"] = meditron_medrag
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
                self.context_length = 1024
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                # torch_dtype=torch.float16,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                model_kwargs={"cache_dir":self.cache_dir},
            )
        
        self.follow_up = follow_up
        if self.rag and self.follow_up:
            self.answer = self.i_medrag_answer
            self.templates["medrag_system"] = simple_medrag_system
            self.templates["medrag_prompt"] = simple_medrag_prompt
            self.templates["i_medrag_system"] = i_medrag_system
            self.templates["follow_up_ask"] = follow_up_instruction_ask
            self.templates["follow_up_answer"] = follow_up_instruction_answer
        else:
            self.answer = self.medrag_answer

    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria

    def generate(self, messages, **kwargs):
        '''
        generate response given messages
        '''
        if "arvan" in self.llm_name.lower():
            max_retries = 3
            base_delay = 2

            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.api_endpoint}/chat/completions",
                        headers={
                            "Authorization": f"apikey {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": kwargs.get("max_tokens", 2048),
                            "temperature": kwargs.get("temperature", 0.0),
                        },
                        timeout=(60, 180),
                    )

                    response.raise_for_status()
                    data = response.json()
                    usage = data.get("usage", {})
                    self.llm_call_count += 1

                    prompt_tokens = usage.get(
                        "prompt_tokens",
                        0
                    )

                    completion_tokens = usage.get(
                        "completion_tokens",
                        0
                    )

                    total_tokens = usage.get(
                        "total_tokens",
                        prompt_tokens + completion_tokens
                    )

                    self.llm_call_history.append({
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    })
                    self.prompt_tokens += prompt_tokens
                    self.completion_tokens += completion_tokens
                    self.total_tokens += total_tokens

                    return data["choices"][0]["message"]["content"]

                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        print(f"CRITICAL: Arvan API failed after {max_retries} attempts. Error: {e}")
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    print(f"Network error: {e}. Retrying Arvan API in {delay} seconds...")
                    time.sleep(delay)
        elif "openai" in self.llm_name.lower():

            # Was: openai_client(...), which already unwraps to
            # .choices[0].message.content (a plain string) -- but this
            # branch also needs .usage below, so it must use the raw,
            # full-response variant instead.
            response = openai_raw_client(
                model=self.model,
                messages=messages,
                temperature=0.0,
                **kwargs
            )

            self.llm_call_count += 1

            usage = response.usage

            prompt_tokens = (
                getattr(usage, "prompt_tokens", 0)
                or 0
            )

            completion_tokens = (
                getattr(usage, "completion_tokens", 0)
                or 0
            )

            total_tokens = (
                getattr(usage, "total_tokens", 0)
                or (
                    prompt_tokens
                    + completion_tokens
                )
            )

            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens

            self.llm_call_history.append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            })

            ans = response.choices[0].message.content
            
        elif "gemini" in self.llm_name.lower():

            response = self.model.generate_content(
                messages[0]["content"]
                + "\n\n"
                + messages[1]["content"],
                **kwargs
            )

            self.llm_call_count += 1

            usage = getattr(
                response,
                "usage_metadata",
                None
            )

            prompt_tokens = getattr(
                usage,
                "prompt_token_count",
                0
            ) if usage else 0

            completion_tokens = getattr(
                usage,
                "candidates_token_count",
                0
            ) if usage else 0

            total_tokens = getattr(
                usage,
                "total_token_count",
                prompt_tokens + completion_tokens
            ) if usage else (
                prompt_tokens + completion_tokens
            )

            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens

            self.llm_call_history.append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            })

            ans = (
                response.candidates[0]
                .content.parts[0]
                .text
            )
        else:
            stopping_criteria = None
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if "meditron" in self.llm_name.lower():
                # stopping_criteria = custom_stop(["###", "User:", "\n\n\n"], self.tokenizer, input_len=len(self.tokenizer.encode(prompt_cot, add_special_tokens=True)))
                stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))

            # Same bug as the OpenAI/Arvan branches had: do_sample was
            # hardcoded to False, so a passed-in temperature was silently
            # ignored (greedy decoding regardless). Derive do_sample from
            # whether a real temperature was requested, and only pass
            # temperature through when actually sampling -- HF's generate()
            # warns/ignores it otherwise anyway.
            gen_temperature = kwargs.pop("temperature", 0.0)
            do_sample = gen_temperature is not None and gen_temperature > 0.0
            sampling_kwargs = {"do_sample": do_sample}
            if do_sample:
                sampling_kwargs["temperature"] = gen_temperature

            if "llama-3" in self.llm_name.lower():
                response = self.model(
                    prompt,
                    eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    **sampling_kwargs,
                    **kwargs
                )
            else:
                response = self.model(
                    prompt,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    **sampling_kwargs,
                    **kwargs
                )
            # ans = response[0]["generated_text"]
            generated_text = response[0]["generated_text"]

            ans = generated_text[len(prompt):]

            self.llm_call_count += 1

            prompt_tokens = len(
                self.tokenizer.encode(
                    prompt,
                    add_special_tokens=False
                )
            )

            completion_tokens = len(
                self.tokenizer.encode(
                    ans,
                    add_special_tokens=False
                )
            )

            total_tokens = (
                prompt_tokens
                + completion_tokens
            )

            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens

            self.llm_call_history.append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            })
        return ans

    def generate_answer_with_letter_probs(self, messages, option_letters, max_tokens=512, temperature=0.0):
        """
        Single call that returns a continuous 0-1 probability for each MCQ
        option letter, extracted from token-level logprobs -- WITHOUT
        forcing a bare-letter completion. `messages` should use your normal
        reasoning-permitting prompt/template (same as your n_trials answer
        generation), and this scans across all generated tokens for wherever
        the model actually commits to its final letter (dominance-based,
        matching generate_answer_with_letter_probs_local's approach), rather
        than requiring the answer to be the very first token. This avoids
        the accuracy cost we found when forcing an immediate bare-letter
        answer (see the gpt-4o-mini findings) -- confirmed working via
        Chabokan/OpenRouter-routed Llama-3.1-8B-Instruct with logprobs=True.

        Supports OpenAI-schema gateways (llm_name="OpenAI/...", including
        Liara/Chabokan/etc. via a custom base_url) and Arvan
        (llm_name="Arvan/...") -- IMPORTANT: whether a given gateway/model
        actually returns logprobs is not guaranteed in general and should be
        verified with a raw test call first (see test_arvan_logprobs.py /
        test_chabokan_logprobs.py) before relying on this for a new provider.

        `option_letters` is the list of valid letters for this question,
        e.g. ["A", "B", "C", "D"].

        Returns: (raw_text, {letter: probability, ...})
        """
        is_arvan = "arvan" in self.llm_name.lower()
        is_openai = "openai" in self.llm_name.lower()

        if not (is_arvan or is_openai):
            raise ValueError(
                f"generate_answer_with_letter_probs requires an OpenAI- or "
                f"Arvan-backed MedRAG instance, got '{self.llm_name}'."
            )

        request_kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=20,
        )

        if is_arvan:
            resp = requests.post(
                f"{self.api_endpoint}/chat/completions",
                headers={
                    "Authorization": f"apikey {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_kwargs,
                timeout=(60, 180),
            )
            resp.raise_for_status()
            data = resp.json()

            self.llm_call_count += 1
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.llm_call_history.append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            })

            choice = data["choices"][0]
            raw_text = choice["message"]["content"]
            logprobs_obj = choice.get("logprobs")
            # dict-style entries: [{"token": ..., "top_logprobs": [{"token": ..., "logprob": ...}, ...]}, ...]
            token_entries = (logprobs_obj or {}).get("content", []) or []

            def top_list_of(e): return e.get("top_logprobs", [])
            def cand_token(c): return c["token"]
            def cand_logprob(c): return c["logprob"]

        else:  # is_openai (covers any OpenAI-schema gateway via base_url, e.g. Liara/Chabokan)
            response = openai_raw_client(**request_kwargs)

            self.llm_call_count += 1
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.llm_call_history.append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            })

            choice = response.choices[0]
            raw_text = choice.message.content
            token_entries = choice.logprobs.content if choice.logprobs else []

            def top_list_of(e): return e.top_logprobs
            def cand_token(c): return c.token
            def cand_logprob(c): return c.logprob

        # Dominance-based scan across ALL generated tokens (not just the
        # first match) -- with reasoning allowed, an early token could
        # accidentally look like a letter (e.g. the article "A" mid-sentence)
        # without actually being the model's answer commitment. Only a
        # position where option letters combine to >50% of the probability
        # mass is treated as the real answer; otherwise fall back to
        # whichever position had the most combined letter-mass.
        best_mass = -1.0
        best_probs = {letter: 0.0 for letter in option_letters}

        for tok in token_entries:
            candidates = {}
            for c in top_list_of(tok):
                tok_str = cand_token(c).strip().upper()
                candidates[tok_str] = math.exp(cand_logprob(c))
            letter_mass = {L: candidates.get(L, 0.0) for L in option_letters}
            total = sum(letter_mass.values())

            if total > 0.5:
                return raw_text, {L: letter_mass[L] / total for L in option_letters}

            if total > best_mass:
                best_mass = total
                best_probs = {L: (letter_mass[L] / total if total > 0 else 0.0) for L in option_letters}

        if best_mass <= 0 and raw_text:
            print(f"WARNING: none of {option_letters} found in top_logprobs for "
                  f"response {raw_text!r} -- check the prompt/template, or that "
                  f"this gateway actually returns logprobs for this model.")

        return raw_text, best_probs

    def generate_answer_with_letter_probs_local(self, messages, option_letters, max_new_tokens=512, temperature=0.0):
        """
        For LOCALLY-hosted HF models only (e.g. Llama-3.1-8B-Instruct).

        Unlike generate_answer_with_letter_probs (which requires forcing a
        bare-letter completion to get clean logprobs, at a real accuracy
        cost), this runs ONE normal generation call -- full reasoning
        allowed, same prompt/template as your existing n_trials answer
        generation -- and extracts a continuous per-letter probability from
        the model's own logits at the token position where it actually
        commits to its final answer (e.g. "...Answer: D"), not the first
        token. No accuracy trade-off: the model reasons exactly as it
        normally would.

        Returns: (raw_text, {letter: probability, ...})
        """
        underlying_model = getattr(self.model, "model", self.model)
        tokenizer = getattr(self.model, "tokenizer", None) or self.tokenizer

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(underlying_model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = underlying_model.generate(**inputs, **gen_kwargs)

        generated_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        self.llm_call_count += 1
        prompt_tokens = inputs["input_ids"].shape[1]
        completion_tokens = generated_ids.shape[0]
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.llm_call_history.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })

        # Pre-compute the token id(s) each option letter maps to, accounting
        # for a leading-space variant (common with BPE tokenizers -- "B" and
        # " B" are usually different token ids).
        letter_token_ids = {}
        for letter in option_letters:
            for variant in (letter, f" {letter}"):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1:
                    letter_token_ids.setdefault(letter, set()).add(ids[0])

        best_step_mass = -1.0
        best_step_probs = {letter: 0.0 for letter in option_letters}

        for step_logits in out.scores:
            probs = torch.softmax(step_logits[0].float(), dim=-1)
            step_letter_mass = {
                letter: sum(probs[i].item() for i in ids)
                for letter, ids in letter_token_ids.items()
            }
            total = sum(step_letter_mass.values())

            # A step where option letters dominate the distribution (>50%
            # combined mass) is very likely the model committing to its
            # final answer, not a stray letter-shaped token from mid-reasoning.
            if total > 0.5:
                letter_probs = {L: step_letter_mass.get(L, 0.0) / total for L in option_letters}
                return raw_text, letter_probs

            if total > best_step_mass:
                best_step_mass = total
                best_step_probs = {
                    L: (step_letter_mass.get(L, 0.0) / total if total > 0 else 0.0)
                    for L in option_letters
                }

        # Fallback: no step crossed the 0.5 threshold (unusual formatting).
        # Use whichever step had the most combined letter-mass rather than
        # silently returning all-zero.
        if best_step_mass <= 0:
            print(f"WARNING: could not locate any answer-letter token in generation "
                  f"{raw_text!r} -- reward will be 0 for all options.")
        return raw_text, best_step_probs

    def medrag_answer(self, question, options=None, k=32, rrf_k=100, save_dir = None, snippets=None, snippets_ids=None, **kwargs):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        '''

        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''

        # retrieve relevant snippets
        if self.rag:
            if snippets is not None:
                retrieved_snippets = snippets[:k]
                scores = []
            elif snippets_ids is not None:
                if self.docExt is None:
                    self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
                retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                scores = []
            else:
                assert self.retrieval_system is not None
                retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)

            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
            if len(contexts) == 0:
                contexts = [""]
            if "openai" in self.llm_name.lower() or "arvan" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            elif "gemini" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            else:
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
        else:
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # generate answers
        answers = []
        if not self.rag:
            prompt_cot = self.templates["cot_prompt"].render(question=question, options=options)
            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": prompt_cot}
            ]
            ans = self.generate(messages, **kwargs)
            answers.append(re.sub("\s+", " ", ans))
        else:
            for context in contexts:
                prompt_medrag = self.templates["medrag_prompt"].render(context=context, question=question, options=options)
                messages=[
                        {"role": "system", "content": self.templates["medrag_system"]},
                        {"role": "user", "content": prompt_medrag}
                ]
                ans = self.generate(messages, **kwargs)
                answers.append(re.sub("\s+", " ", ans))
        
        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)
        
        return answers[0] if len(answers)==1 else answers, retrieved_snippets, scores

    def i_medrag_answer(self, question, options=None, k=32, rrf_k=100, save_path = None, n_rounds=4, n_queries=3, qa_cache_path=None, **kwargs):
        self.llm_call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_call_history = []

        start_time = time.perf_counter()
        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''
        QUESTION_PROMPT = f"Here is the question:\n{question}\n\n{options}"

        context = ""
        qa_cache = []
        if qa_cache_path is not None and os.path.exists(qa_cache_path):
            qa_cache = eval(open(qa_cache_path, 'r').read())[:n_rounds]
            if len(qa_cache) > 0:
                context = qa_cache[-1]
            n_rounds = n_rounds - len(qa_cache)
        last_context = None

        # Run in loop
        max_iterations = n_rounds + 3
        saved_messages = [{"role": "system", "content": self.templates["i_medrag_system"]}]

        for i in range(max_iterations):
            if i < n_rounds:
                if context == "":
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
                else:                
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
            elif context != last_context:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            elif len(messages) == 1:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            saved_messages.append(messages[-1])
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
            last_context = context
            last_content = self.generate(messages, **kwargs)
            response_message = {"role": "assistant", "content": last_content}
            saved_messages.append(response_message)
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)       
            if i >= n_rounds and ("## Answer" in last_content or "answer is" in last_content.lower()):
                messages.append(response_message)
                messages.append(
                    {
                        "role": "user",
                        "content": "Output the answer in JSON: {'answer': your_answer (A/B/C/D)}" if options else "Output the answer in JSON: {'answer': your_answer}",
                    }
                )
                saved_messages.append(messages[-1])
                answer_content = self.generate(messages, **kwargs)
                answer_message = {"role": "assistant", "content": answer_content}
                messages.append(answer_message)
                saved_messages.append(messages[-1])
                if save_path:
                    with open(save_path, 'w') as f:
                        json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
                latency_seconds = (
                    time.perf_counter()
                    - start_time
                )
                return (
                    messages[-1]["content"],
                    messages,
                    {
                            "llm_calls": self.llm_call_count,
                            "prompt_tokens": self.prompt_tokens,
                            "completion_tokens": self.completion_tokens,
                            "total_tokens": self.total_tokens,
                            "latency_seconds": latency_seconds,
                        }
                    )
            elif "## Queries" in last_content:
                messages = messages[:-1]
                if last_content.split("## Queries")[-1].strip() == "":
                    print("Empty queries. Continue with next iteration.")
                    continue
                try:
                    action_str = self.generate([
                        {
                            "role": "user",
                            "content": f"Parse the following passage and extract the queries as a list: {last_content}.\n\nPresent the queries as they are. DO NOT merge or break down queries. Output the list of queries in JSON format: {{\"output\": [\"query 1\", ..., \"query N\"]}}",
                        }
                    ], **kwargs)
                    parsed_json = robust_parse_json(action_str)
                    raw_queries = parsed_json.get("output", [])
                    action_list = [re.sub(r'^\d+\.\s*', '', str(s).strip()) for s in raw_queries]
                except Exception as E:
                    print("Error parsing action list. Continue with next iteration.")
                    error_class = E.__class__.__name__
                    error = f"{error_class}: {str(E)}"
                    print(error)
                    if save_path:
                        with open(save_path + ".error", 'a') as f:
                            f.write(f"{error}\n")                
                    continue
                for question in action_list:
                    if question.strip() == "":
                        continue
                    try:
                        rag_result = self.medrag_answer(question, k=k, rrf_k=rrf_k, **kwargs)[0]
                        context += f"\n\nQuery: {question}\nAnswer: {rag_result}"
                        context = context.strip()
                    except Exception as E:
                        error_class = E.__class__.__name__
                        error = f"{error_class}: {str(E)}"
                        print(error)
                        if save_path:
                            with open(save_path + ".error", 'a') as f:
                                f.write(f"{error}\n")
                qa_cache.append(context)
                if qa_cache_path:
                    with open(qa_cache_path, 'w') as f:
                        json.dump(qa_cache, f, indent=4)
            else:
                messages.append(response_message)
                print("No queries or answer. Continue with next iteration.")
                continue
        latency_seconds = (
            time.perf_counter()
            - start_time
        )
        return (
            messages[-1]["content"],
            messages,
            {
                "llm_calls": self.llm_call_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "latency_seconds": latency_seconds,
            }
        )

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)