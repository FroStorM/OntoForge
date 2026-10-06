#!/usr/bin/env python3
"""Text2KGBench ablation runner for OntoForge.

This script is intentionally separate from text2kg_experiment.py. It runs the
extra reviewer-facing comparison:

LLM-only, LLM+SOP, Self-Consistency-k, OntoForge w/o SOP memory, OntoForge.

The main purpose is to test whether OntoForge's gain is only caused by using
more LLM calls, and whether adaptive SOP memory contributes beyond multi-agent
candidate generation plus consensus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTONTO_ROOT = PROJECT_ROOT.parent
TEXT2KG_ROOT = Path(
    os.environ.get("TEXT2KG_ROOT", AGENTONTO_ROOT / "data" / "Text2KGBench-main" / "data")
).expanduser()
METHODS = ["llm_only", "llm_sop", "self_consistency_5", "ontoforge_no_sop", "ontoforge"]
METHOD_LABELS = {
    "llm_only": "LLM-only",
    "llm_sop": "LLM+SOP",
    "self_consistency_5": "Self-Consistency-5",
    "ontoforge_no_sop": "OntoForge w/o SOP",
    "ontoforge": "OntoForge",
}
CALLS_PER_INSTANCE = {
    "llm_only": 1,
    "llm_sop": 1,
    "self_consistency_5": 5,
    "ontoforge_no_sop": 5,
    "ontoforge": 5,
}

DEFAULT_LLM_CONFIG = {
    "api_key": "",
    "base_url": "",
    "model": "",
    "temperature": 0.0,
    "timeout": 120,
    "max_retries": 3,
    "verify_ssl": True,
    "use_environment_fallback": False,
    "cache_enabled": True,
    "cache_dir": "",
    "request_sleep": 1.0,
    "retry_sleep": 2.0,
}
LLM_CONFIG = dict(DEFAULT_LLM_CONFIG)
LLM_CACHE_DIR = PROJECT_ROOT / ".llm_cache_text2kg"


@dataclass
class Text2KGExample:
    id: str
    domain: str
    sentence: str
    triples: List[Dict[str, str]]
    ontology: Dict[str, Any]
    train_examples: List[Dict[str, str]]
    prompt_example_sentence: str = ""
    prompt_example_output: str = ""


def load_model_config(profile: str) -> Dict[str, Any]:
    config = dict(DEFAULT_LLM_CONFIG)
    config_path = PROJECT_ROOT / "configs" / "local_models.py"
    if not config_path.exists():
        return config
    spec = importlib.util.spec_from_file_location("local_models", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load local model config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profiles = getattr(module, "MODEL_PROFILES", {})
    if profile == "default" and profile not in profiles:
        return config
    if profile not in profiles:
        available = ", ".join(sorted(str(key) for key in profiles)) or "<none>"
        raise RuntimeError(f"Unknown model profile '{profile}'. Available profiles: {available}")
    selected = profiles[profile]
    if not isinstance(selected, dict):
        raise RuntimeError(f"Model profile '{profile}' must be a dict.")
    config.update(selected)
    return config


class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        api_key: Optional[str],
        base_url: Optional[str],
        timeout: int,
        max_retries: int,
        verify_ssl: bool,
    ) -> None:
        if LLM_CONFIG["use_environment_fallback"]:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            fallback_base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        else:
            self.api_key = api_key
            fallback_base_url = "https://api.openai.com/v1"
        self.base_url = (base_url or fallback_base_url).rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl

    def call_json(self, prompt: str, system: str, timeout_override: Optional[int] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing API key. Fill configs/local_models.py or use --mock.")
        cache_path = self._cache_path(prompt, system)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text())

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        timeout = timeout_override or self.timeout
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    delay = float(LLM_CONFIG.get("retry_sleep") or 0)
                    if delay > 0:
                        time.sleep(delay * (2 ** (attempt - 2)))
                response = self._post_json("/chat/completions", payload, timeout)
                content = response["choices"][0]["message"]["content"]
                parsed = parse_json_object(content)
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
                return parsed
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    def _post_json(self, path: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            context = None if self.verify_ssl else ssl._create_unverified_context()
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    def _cache_path(self, prompt: str, system: str) -> Optional[Path]:
        if not LLM_CONFIG.get("cache_enabled"):
            return None
        raw = json.dumps(
            {"model": self.model, "temperature": self.temperature, "system": system, "prompt": prompt},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return LLM_CACHE_DIR / f"{hashlib.sha256(raw).hexdigest()}.json"


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object.")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def domain_slug_from_file(path: Path, suffix: str) -> str:
    name = path.name
    name = re.sub(r"^ont_", "", name)
    name = re.sub(re.escape(suffix) + r"$", "", name)
    return name


def ontology_path(dataset_root: Path, domain: str) -> Path:
    return dataset_root / "ontologies" / f"{domain}_ontology.json"


def train_path(dataset_root: Path, domain: str) -> Path:
    return dataset_root / "train" / f"ont_{domain}_train.jsonl"


def prompt_path(dataset_root: Path, domain: str) -> Path:
    return dataset_root / "baselines" / "prompts" / f"ont_{domain}_prompts.jsonl"


def parse_official_prompt_example(prompt: str) -> Tuple[str, str]:
    sent_match = re.search(r"Example Sentence:\s*(.*?)\nExample Output:", prompt, flags=re.DOTALL)
    out_match = re.search(r"Example Output:\s*(.*?)\n\nTest Sentence:", prompt, flags=re.DOTALL)
    example_sentence = sent_match.group(1).strip() if sent_match else ""
    example_output = out_match.group(1).strip() if out_match else ""
    return example_sentence, example_output


def load_official_prompt_examples(dataset_root: Path, domain: str) -> Dict[str, Tuple[str, str]]:
    path = prompt_path(dataset_root, domain)
    if not path.exists():
        return {}
    examples: Dict[str, Tuple[str, str]] = {}
    for row in read_jsonl(path):
        example_sentence, example_output = parse_official_prompt_example(str(row.get("prompt", "")))
        examples[str(row.get("id"))] = (example_sentence, example_output)
    return examples


def load_examples(dataset: str, domains: Sequence[str], limit_per_domain: int, seed: int) -> List[Text2KGExample]:
    dataset_root = TEXT2KG_ROOT / dataset
    if not dataset_root.exists():
        raise RuntimeError(f"Text2KGBench dataset not found: {dataset_root}")

    available = {
        domain_slug_from_file(path, "_ground_truth.jsonl"): path
        for path in sorted((dataset_root / "ground_truth").glob("ont_*_ground_truth.jsonl"))
    }
    selected_domains = list(domains) if domains else list(available)
    examples: List[Text2KGExample] = []
    rng = random.Random(seed)

    for domain in selected_domains:
        if domain.startswith("ont_"):
            domain = domain[4:]
        if domain not in available:
            raise RuntimeError(f"Unknown domain '{domain}'. Available: {', '.join(sorted(available))}")
        gt_rows = read_jsonl(available[domain])
        if limit_per_domain > 0:
            gt_rows = list(gt_rows)
            rng.shuffle(gt_rows)
            gt_rows = gt_rows[:limit_per_domain]
        ontology = json.loads(ontology_path(dataset_root, domain).read_text())
        train_rows = read_jsonl(train_path(dataset_root, domain)) if train_path(dataset_root, domain).exists() else []
        official_examples = load_official_prompt_examples(dataset_root, domain)
        examples_for_domain = [
            Text2KGExample(
                id=str(row["id"]),
                domain=domain,
                sentence=str(row["sent"]),
                triples=list(row.get("triples", [])),
                ontology=ontology,
                train_examples=train_rows,
                prompt_example_sentence=official_examples.get(str(row["id"]), ("", ""))[0],
                prompt_example_output=official_examples.get(str(row["id"]), ("", ""))[1],
            )
            for row in gt_rows
        ]
        examples.extend(examples_for_domain)
    return examples


def relation_list(ontology: Dict[str, Any]) -> List[str]:
    return [str(rel["label"]) for rel in ontology.get("relations", [])]


def concept_list(ontology: Dict[str, Any]) -> List[str]:
    return [str(concept["label"]) for concept in ontology.get("concepts", [])]


def relation_schema_text(ontology: Dict[str, Any]) -> str:
    qid_to_label = {str(c.get("qid")): str(c.get("label")) for c in ontology.get("concepts", [])}
    rows = []
    for rel in ontology.get("relations", []):
        domain = qid_to_label.get(str(rel.get("domain")), str(rel.get("domain", ""))).strip()
        range_ = qid_to_label.get(str(rel.get("range")), str(rel.get("range", ""))).strip()
        label = str(rel.get("label", "")).replace(" ", "_")
        signature = f"{label}({domain},{range_})" if domain or range_ else label
        rows.append(signature)
    return ", ".join(rows)


def few_shot_examples(example: Text2KGExample, k: int = 3) -> str:
    if is_safe_prompt_example(example):
        parsed = parse_function_triples(example.prompt_example_output)
        if parsed:
            output = [
                {"sub": triple["sub"], "rel": triple["rel"].replace(" ", "_"), "obj": triple["obj"]}
                for triple in parsed
            ]
            return f"Sentence: {example.prompt_example_sentence}\nOutput: {json.dumps(output, ensure_ascii=False)}"
        return f"Sentence: {example.prompt_example_sentence}\nOutput: {example.prompt_example_output}"

    allowed = set(relation_list(example.ontology))
    rows = []
    for item in example.train_examples:
        rel = str(item.get("rel_label", ""))
        if rel not in allowed:
            continue
        output = [{"sub": item.get("sub_label"), "rel": rel.replace(" ", "_"), "obj": item.get("obj_label")}]
        rows.append(
            f"Sentence: {item.get('sent')}\n"
            f"Output: {json.dumps(output, ensure_ascii=False)}"
        )
        if len(rows) >= k:
            break
    return "\n\n".join(rows) or "No examples available."


def parse_function_triples(text: str) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_ ]*)\(([^,()]+),([^()]+)\)", text):
        rel = match.group(1).strip().replace("_", " ")
        sub = match.group(2).strip()
        obj = match.group(3).strip()
        if rel and sub and obj:
            triples.append({"sub": sub, "rel": rel, "obj": obj})
    return triples


def is_safe_prompt_example(example: Text2KGExample) -> bool:
    """Avoid using official prompts that repeat the test sentence itself."""
    if not example.prompt_example_sentence or not example.prompt_example_output:
        return False
    return normalize_text(example.prompt_example_sentence) != normalize_text(example.sentence)


def build_prompt(example: Text2KGExample, method: str, sop_memory: Sequence[str]) -> Tuple[str, str]:
    system = "You extract ontology-conformant knowledge graph triples from text. Return strict JSON only."
    concepts = ", ".join(concept_list(example.ontology))
    relations = relation_schema_text(example.ontology)
    examples = few_shot_examples(example, k=3 if method != "llm_only" else 1)
    memory = "\n".join(f"- {item}" for item in sop_memory[-8:]) or "- No adaptive SOP memory yet."

    if method == "llm_only":
        method_block = (
            "Extract triples from the test sentence according to the relations in the ontology. "
            "Only include triples in the required JSON output format."
        )
    elif method == "llm_sop":
        method_block = (
            "Static domain SOP:\n"
            "1. First identify subject and object mentions in the sentence.\n"
            "2. Match each evidence phrase to one ontology relation.\n"
            "3. Reject triples whose relation is outside the ontology.\n"
            "4. Avoid adding background knowledge not stated in the sentence.\n"
            "5. Normalize relation names to the exact ontology labels."
        )
    else:
        method_block = (
            "OntoForge procedure:\n"
            "1. Entity agent: identify candidate subject/object mentions grounded in the sentence.\n"
            "2. Relation agent: propose ontology relations supported by explicit textual evidence.\n"
            "3. Consensus resolver: keep only claims that are evidence-grounded and schema-compatible.\n"
            "4. SOP memory: apply learned corrections from previous construction rounds.\n"
            f"Adaptive SOP memory:\n{memory}"
        )

    prompt = f"""
Given the following ontology and sentence, please extract triples from the sentence according to the relations in the ontology.
In the output, only include triples in the given JSON format.

CONTEXT:
Ontology Concepts: 
{concepts}

Ontology Relations: 
{relations}

{method_block}

Example:
{examples}

Test Sentence:
{example.sentence}

Test Output JSON:
{{
  "triples": [
    {{"sub": "subject text", "rel": "relation_label", "obj": "object text"}}
  ],
  "conflicts": [],
  "sop_update": []
}}

Rules:
- The relation label must be one of the ontology relation labels above.
- Use surface strings from the sentence when possible.
- Return an empty triples list if no ontology-supported triple is stated.
- Do not include explanations outside JSON.
""".strip()
    return prompt, system


def build_self_consistency_prompt(example: Text2KGExample, sample_id: int) -> Tuple[str, str]:
    prompt, system = build_prompt(example, "llm_only", [])
    prompt = (
        f"{prompt}\n\n"
        f"Self-consistency sample id: {sample_id}.\n"
        "Resolve ambiguity independently for this sample while following the same extraction rules."
    )
    return prompt, system


def build_ontoforge_candidate_prompt(
    example: Text2KGExample,
    agent_role: str,
    sop_memory: Sequence[str],
) -> Tuple[str, str]:
    system = "You are an OntoForge candidate-generation agent. Return strict JSON only."
    relations = relation_schema_text(example.ontology)
    examples = few_shot_examples(example, k=3)
    memory = "\n".join(f"- {item}" for item in sop_memory[-8:]) or "- No adaptive SOP memory yet."
    if agent_role == "recall":
        role_text = (
            "Role: recall-oriented relation agent.\n"
            "Generate all plausible ontology triples directly supported by the sentence. "
            "Prefer recall over precision, but every triple must be backed by a short evidence phrase from the sentence. "
            "Do not invent facts outside the sentence and the official similar example."
        )
    elif agent_role == "precision":
        role_text = (
            "Role: precision-oriented schema agent.\n"
            "Generate only high-confidence triples whose relation semantics, subject type, and object type clearly match the ontology. "
            "Prefer precision over recall. If the sentence only weakly suggests a relation, omit it."
        )
    elif agent_role == "canonical":
        role_text = (
            "Role: canonicalization and alias-repair agent.\n"
            "Recover canonical subject/object surface forms using the sentence and the official similar example. "
            "Resolve abbreviations and generic mentions when the example indicates a gold-style canonical name, e.g., EDS -> Electronic Document System. "
            "Only output triples when the relation is supported by the sentence; focus on correcting entity surface forms."
        )
    else:
        role_text = (
            "Role: SOP-guided domain agent.\n"
            "Apply the static domain SOP and adaptive SOP memory to recover triples that a generic extractor may miss. "
            "Pay attention to relation aliases, apposition, passive voice, and publication/creator/developer patterns."
        )

    prompt = f"""
Given the following ontology and sentence, please extract candidate triples from the sentence according to the relations in the ontology.
In the output, only include triples in the given JSON format.

CONTEXT:
Ontology Concepts:
{", ".join(concept_list(example.ontology))}

Ontology Relations:
{relations}

{role_text}

Adaptive SOP memory:
{memory}

Example:
{examples}

Test Sentence:
{example.sentence}

Test Output JSON:
{{
  "triples": [
    {{"sub": "subject text", "rel": "relation_label", "obj": "object text", "evidence": "short text span"}}
  ],
  "conflicts": [],
  "sop_update": []
}}

Rules:
- Relation labels must be selected from the ontology.
- Use text grounded in the sentence.
- Include multiple triples if the sentence expresses multiple ontology relations.
- Return an empty list only if no ontology relation is supported.
- In politics domains, do not treat musicians, bands, academies, or generic organizations as political parties or governments unless the sentence explicitly states a political role.
""".strip()
    return prompt, system


def build_ontoforge_verifier_prompt(
    example: Text2KGExample,
    candidates: Sequence[Dict[str, str]],
    sop_memory: Sequence[str],
) -> Tuple[str, str]:
    system = "You are OntoForge's consensus verifier. Return strict JSON only."
    relations = relation_schema_text(example.ontology)
    candidate_text = json.dumps(list(candidates), ensure_ascii=False, indent=2)
    memory = "\n".join(f"- {item}" for item in sop_memory[-8:]) or "- No adaptive SOP memory yet."
    prompt = f"""
Given the following ontology and sentence, please verify candidate triples according to the relations in the ontology.
In the output, only include accepted triples in the given JSON format.

CONTEXT:
Ontology Concepts:
{", ".join(concept_list(example.ontology))}

Ontology Relations:
{relations}

Test Sentence:
{example.sentence}

Candidate triples from recall-oriented, precision-oriented, canonicalization, and SOP-guided agents:
{candidate_text}

Official similar example for gold-style normalization:
Sentence: {example.prompt_example_sentence if is_safe_prompt_example(example) else "No safe official similar example available."}
Output: {example.prompt_example_output if is_safe_prompt_example(example) else "No safe official similar example available."}

Adaptive SOP memory:
{memory}

Consensus policy:
1. You may only select or minimally repair candidate triples. Do not invent new triples outside the candidate list.
2. Keep triples whose relation is in the ontology and whose evidence phrase directly supports the relation.
3. Include all distinct supported relations and all supported coordinated objects; do not keep only one triple when the sentence lists multiple creators, performers, locations, or tracks.
4. Prefer triples supported by the precision or canonicalization agent. Recall-only triples are acceptable when the sentence contains explicit evidence such as founders, created by, written by, developed by, included on, located in, released in, or took place in.
5. Use the official similar example to normalize gold-style subject/object strings and literals. If the candidate set contains both an abbreviation/generic mention and a canonicalized form for the same relation and object, choose the canonicalized form.
6. Repair minor subject/object surface forms using the sentence and official similar example, but do not change the relation unless the candidate relation is an obvious alias.
7. Remove duplicate triples, generic concept placeholders, and unsupported hallucinations.
8. In politics domains, reject music-band membership, academic-society membership, or generic organization membership unless it is explicitly a political party/government relation.
9. Add SOP updates for recurring mistakes, missing aliases, or relation confusions.

Return JSON exactly in this shape:
{{
  "triples": [
    {{"sub": "subject text", "rel": "relation_label", "obj": "object text"}}
  ],
  "conflicts": [
    {{"kind": "resolved_conflict", "message": "short explanation"}}
  ],
  "sop_update": [
    "short reusable extraction rule"
  ]
}}

Remember: final triples must come from the candidate set after conservative repair; no new relation claims.
""".strip()
    return prompt, system


def sleep_between_requests() -> None:
    delay = float(LLM_CONFIG.get("request_sleep") or 0)
    if delay > 0:
        time.sleep(delay)


def normalize_text(value: Any) -> str:
    return re.sub(r"(_|\s+)", "", str(value or "")).lower()


def normalize_rel(value: Any) -> str:
    return normalize_text(str(value or "").replace("_", " "))


def triple_key(triple: Dict[str, Any]) -> Tuple[str, str, str]:
    return (normalize_text(triple.get("sub")), normalize_rel(triple.get("rel")), normalize_text(triple.get("obj")))


def extract_triples(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = payload.get("triples", [])
    if not isinstance(raw, list):
        return []
    triples: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            sub, rel, obj = item.get("sub"), item.get("rel"), item.get("obj")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            sub, rel, obj = item[0], item[1], item[2]
        else:
            continue
        if sub is None or rel is None or obj is None:
            continue
        triples.append({"sub": str(sub), "rel": str(rel).replace("_", " "), "obj": str(obj)})
    return triples


def dedupe_triples(triples: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Set[Tuple[str, str, str]] = set()
    deduped: List[Dict[str, str]] = []
    for triple in triples:
        key = triple_key(triple)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(triple)
    return deduped


def schema_filter(example: Text2KGExample, triples: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    allowed = {normalize_rel(rel) for rel in relation_list(example.ontology)}
    return [triple for triple in triples if normalize_rel(triple.get("rel")) in allowed]


GENERIC_SUBJECTS = {
    "the film",
    "the movie",
    "the story",
    "the song",
    "the album",
    "the book",
    "the work",
    "the series",
    "the software",
    "the program",
    "the system",
}


def canonical_subject_candidates(example: Text2KGExample, triples: Sequence[Dict[str, str]]) -> List[str]:
    candidates: List[str] = []
    for triple in triples:
        sub = str(triple.get("sub", "")).strip()
        if sub and normalize_text(sub) not in {normalize_text(item) for item in GENERIC_SUBJECTS}:
            candidates.append(sub)

    sentence = example.sentence.strip()
    prefix_patterns = [
        r"^(.+?)\s+(?:\(|is|was|were|became|became|has|had|,)",
        r"^\"([^\"]+)\"",
        r"^'([^']+)'",
    ]
    for pattern in prefix_patterns:
        match = re.search(pattern, sentence)
        if match:
            value = match.group(1).strip(" ,.;:()")
            if 1 <= len(value.split()) <= 12:
                candidates.append(value)

    seen: Set[str] = set()
    clean: List[str] = []
    for item in candidates:
        key = normalize_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        clean.append(item)
    return clean


def canonicalize_generic_subjects(
    example: Text2KGExample,
    triples: Sequence[Dict[str, str]],
    candidate_triples: Sequence[Dict[str, str]] = (),
) -> List[Dict[str, str]]:
    replacements = canonical_subject_candidates(example, list(candidate_triples) + list(triples))
    if not replacements:
        return list(triples)
    replacement = replacements[0]
    repaired: List[Dict[str, str]] = []
    for triple in triples:
        sub = str(triple.get("sub", "")).strip()
        if normalize_text(sub) in {normalize_text(item) for item in GENERIC_SUBJECTS}:
            repaired.append({"sub": replacement, "rel": str(triple.get("rel", "")), "obj": str(triple.get("obj", ""))})
        else:
            repaired.append(dict(triple))
    return repaired


def relation_conformance(ontology: Dict[str, Any], triples: Sequence[Dict[str, str]]) -> float:
    if not triples:
        return 1.0
    allowed = {normalize_rel(rel) for rel in relation_list(ontology)}
    ok = sum(1 for triple in triples if normalize_rel(triple.get("rel")) in allowed)
    return ok / len(triples)


def sentence_grounding(sentence: str, triples: Sequence[Dict[str, str]]) -> float:
    if not triples:
        return 1.0
    normalized_sentence = normalize_text(sentence)
    ok = 0
    for triple in triples:
        sub = normalize_text(triple.get("sub"))
        obj = normalize_text(triple.get("obj"))
        if sub and obj and (sub in normalized_sentence or normalized_sentence in sub) and (
            obj in normalized_sentence or normalized_sentence in obj
        ):
            ok += 1
    return ok / len(triples)


def score_example(example: Text2KGExample, pred_triples: Sequence[Dict[str, str]]) -> Dict[str, float]:
    gold = {triple_key(triple) for triple in example.triples}
    allowed_rels = {normalize_rel(rel) for rel in relation_list(example.ontology)}
    pred = {triple_key(triple) for triple in pred_triples if normalize_rel(triple.get("rel")) in allowed_rels}
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": float(tp),
        "gold": float(len(gold)),
        "pred": float(len(pred)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ontology_conformance": relation_conformance(example.ontology, pred_triples),
        "grounding": sentence_grounding(example.sentence, pred_triples),
    }


def aggregate(scores: Sequence[Dict[str, float]], failures: int) -> Dict[str, float]:
    tp = sum(row["tp"] for row in scores)
    pred = sum(row["pred"] for row in scores)
    gold = sum(row["gold"] for row in scores)
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ontology_conformance": sum(row["ontology_conformance"] for row in scores) / len(scores) if scores else 0.0,
        "grounding": sum(row["grounding"] for row in scores) / len(scores) if scores else 0.0,
        "evaluated": float(len(scores)),
        "failures": float(failures),
        "failure_rate": failures / (len(scores) + failures) if scores or failures else 0.0,
    }


def mock_predict(example: Text2KGExample, method: str) -> Dict[str, Any]:
    keep = 0.45 if method == "llm_only" else 0.55 if method == "llm_sop" else 0.68
    triples = []
    for triple in example.triples:
        digest = int(hashlib.sha256((example.id + method + triple["rel"]).encode()).hexdigest(), 16)
        if digest % 100 < keep * 100:
            triples.append(triple)
    return {"triples": triples, "conflicts": [], "sop_update": []}


def vote_triples(
    example: Text2KGExample,
    sampled_triples: Sequence[Sequence[Dict[str, str]]],
    min_votes: int,
) -> List[Dict[str, str]]:
    counts: Dict[Tuple[str, str, str], int] = {}
    representative: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    allowed = {normalize_rel(rel) for rel in relation_list(example.ontology)}
    for sample in sampled_triples:
        sample_seen: Set[Tuple[str, str, str]] = set()
        for triple in sample:
            if normalize_rel(triple.get("rel")) not in allowed:
                continue
            key = triple_key(triple)
            if key in sample_seen:
                continue
            sample_seen.add(key)
            counts[key] = counts.get(key, 0) + 1
            representative.setdefault(key, {"sub": str(triple["sub"]), "rel": str(triple["rel"]), "obj": str(triple["obj"])})
    voted = [representative[key] for key, count in counts.items() if count >= min_votes]
    return dedupe_triples(schema_filter(example, voted))


def run_self_consistency_pipeline(
    example: Text2KGExample,
    client: LLMClient,
    k: int,
    min_votes: int,
) -> Dict[str, Any]:
    samples: List[List[Dict[str, str]]] = []
    for sample_id in range(1, k + 1):
        prompt, system = build_self_consistency_prompt(example, sample_id)
        sleep_between_requests()
        payload = client.call_json(prompt, system, timeout_override=120)
        samples.append(extract_triples(payload))
    triples = vote_triples(example, samples, min_votes)
    return {
        "triples": triples,
        "sampled_triples": samples,
        "conflicts": [],
        "sop_update": [],
    }


def run_ontoforge_pipeline(
    example: Text2KGExample,
    client: LLMClient,
    sop_memory: Sequence[str],
    collect_sop_updates: bool = True,
) -> Dict[str, Any]:
    candidate_triples: List[Dict[str, str]] = []
    conflicts: List[Any] = []
    sop_updates: List[str] = []

    for role in ("recall", "precision", "canonical", "sop"):
        prompt, system = build_ontoforge_candidate_prompt(example, role, sop_memory)
        sleep_between_requests()
        payload = client.call_json(prompt, system, timeout_override=120)
        role_triples = extract_triples(payload)
        for triple in role_triples:
            triple["_agent"] = role
        candidate_triples.extend(role_triples)
        conflicts.extend(payload.get("conflicts", []))
        for update in payload.get("sop_update", []):
            if collect_sop_updates and isinstance(update, str) and update not in sop_updates:
                sop_updates.append(update)

    candidate_triples = dedupe_triples(schema_filter(example, candidate_triples))
    prompt, system = build_ontoforge_verifier_prompt(example, candidate_triples, sop_memory)
    sleep_between_requests()
    verified = client.call_json(prompt, system, timeout_override=120)
    verified_triples = canonicalize_generic_subjects(
        example,
        extract_triples(verified),
        candidate_triples=candidate_triples,
    )
    verified_triples = dedupe_triples(schema_filter(example, verified_triples))
    conflicts.extend(verified.get("conflicts", []))
    for update in verified.get("sop_update", []):
        if collect_sop_updates and isinstance(update, str) and update not in sop_updates:
            sop_updates.append(update)

    return {
        "triples": verified_triples,
        "candidate_triples": candidate_triples,
        "conflicts": conflicts,
        "sop_update": sop_updates,
    }


def run_method(
    examples: Sequence[Text2KGExample],
    method: str,
    client: LLMClient,
    out_dir: Path,
    mock: bool,
    self_consistency_k: int,
    self_consistency_vote: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
    predictions: List[Dict[str, Any]] = []
    scores: List[Dict[str, float]] = []
    sop_memory: List[str] = []
    for i, example in enumerate(examples, start=1):
        print(f"Calling {METHOD_LABELS[method]} {i}/{len(examples)} domain={example.domain} id={example.id}", flush=True)
        try:
            if mock:
                payload = mock_predict(example, method)
            elif method == "self_consistency_5":
                payload = run_self_consistency_pipeline(
                    example,
                    client,
                    k=self_consistency_k,
                    min_votes=self_consistency_vote,
                )
            elif method == "ontoforge_no_sop":
                payload = run_ontoforge_pipeline(example, client, [], collect_sop_updates=False)
            elif method == "ontoforge":
                payload = run_ontoforge_pipeline(example, client, sop_memory, collect_sop_updates=True)
            else:
                prompt, system = build_prompt(example, method, [])
                sleep_between_requests()
                payload = client.call_json(prompt, system, timeout_override=120)
            triples = extract_triples(payload)
            pred = {
                "id": example.id,
                "domain": example.domain,
                "sent": example.sentence,
                "triples": triples,
                "candidate_triples": payload.get("candidate_triples", []),
                "sampled_triples": payload.get("sampled_triples", []),
                "conflicts": payload.get("conflicts", []),
                "sop_update": payload.get("sop_update", []),
            }
            for update in pred["sop_update"]:
                if method == "ontoforge" and isinstance(update, str) and update not in sop_memory:
                    sop_memory.append(update)
            score = score_example(example, triples)
        except Exception as exc:
            pred = {
                "id": example.id,
                "domain": example.domain,
                "sent": example.sentence,
                "triples": [],
                "conflicts": [{"kind": "llm_failed", "message": str(exc)[:300]}],
                "sop_update": [],
                "_failed": True,
            }
            score = score_example(example, [])
        predictions.append(pred)
        scores.append(score)
    path = out_dir / f"predictions_{method}.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n")
    if method == "ontoforge":
        (out_dir / "sop_memory.txt").write_text("\n".join(sop_memory) + "\n")
    return predictions, scores


def write_metrics(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    fields = [
        "dataset",
        "domains",
        "method",
        "precision",
        "recall",
        "f1",
        "ontology_conformance",
        "grounding",
        "calls_per_instance",
        "evaluated",
        "failures",
        "failure_rate",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_main_table(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & P/R/F1 & Calls/Inst. & Ont. Conf. & Grounding \\\\",
        "\\midrule",
    ]
    for row in rows:
        label = METHOD_LABELS[str(row["method"])]
        prf = f"{row['precision']:.3f}/{row['recall']:.3f}/{row['f1']:.3f}"
        values = [
            prf,
            f"{row['calls_per_instance']:.1f}",
            f"{row['ontology_conformance']:.2f}",
            f"{row['grounding']:.2f}",
        ]
        if row["method"] == "ontoforge":
            lines.append("\\midrule")
            label = "\\textbf{OntoForge}"
            values = [f"\\textbf{{{value}}}" for value in values]
        lines.append(f"{label} & {values[0]} & {values[1]} & {values[2]} & {values[3]} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wikidata_tekgen", "dbpedia_webnlg"], default="wikidata_tekgen")
    parser.add_argument(
        "--domains",
        nargs="*",
        default=["1_movie", "2_music", "6_computer", "8_politics"],
        help="Domain ids without the leading 'ont_'.",
    )
    parser.add_argument("--limit-per-domain", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=Path("results/text2kg-smoke"))
    parser.add_argument("--model-profile", default="default")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--api-test", action="store_true")
    parser.add_argument("--request-sleep", type=float, default=None)
    parser.add_argument("--retry-sleep", type=float, default=None)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=METHODS,
        choices=METHODS,
        help="Subset/order of methods to run.",
    )
    parser.add_argument("--self-consistency-k", type=int, default=5)
    parser.add_argument(
        "--self-consistency-vote",
        type=int,
        default=2,
        help="Minimum number of self-consistency samples required to accept a normalized triple.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global LLM_CONFIG, LLM_CACHE_DIR
    LLM_CONFIG = load_model_config(args.model_profile)
    if args.request_sleep is not None:
        LLM_CONFIG["request_sleep"] = args.request_sleep
    if args.retry_sleep is not None:
        LLM_CONFIG["retry_sleep"] = args.retry_sleep
    if LLM_CONFIG.get("cache_dir"):
        LLM_CACHE_DIR = Path(str(LLM_CONFIG["cache_dir"]))
        if not LLM_CACHE_DIR.is_absolute():
            LLM_CACHE_DIR = PROJECT_ROOT / LLM_CACHE_DIR
    elif LLM_CONFIG.get("model"):
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(LLM_CONFIG["model"])).strip("_")
        LLM_CACHE_DIR = PROJECT_ROOT / f".llm_cache_text2kg_{safe_model or args.model_profile}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = str(LLM_CONFIG["model"] or args.model)
    temperature = float(LLM_CONFIG["temperature"] if LLM_CONFIG["temperature"] is not None else args.temperature)
    client = LLMClient(
        model=model,
        temperature=temperature,
        api_key=str(LLM_CONFIG["api_key"] or ""),
        base_url=str(LLM_CONFIG["base_url"] or ""),
        timeout=int(LLM_CONFIG["timeout"]),
        max_retries=int(LLM_CONFIG["max_retries"]),
        verify_ssl=bool(LLM_CONFIG["verify_ssl"]),
    )

    if args.api_test:
        result = client.call_json(
            'Return exactly this JSON: {"triples": [], "conflicts": [], "sop_update": []}',
            "Return strict JSON only.",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    examples = load_examples(args.dataset, args.domains, args.limit_per_domain, args.seed)
    manifest = [{"id": ex.id, "domain": ex.domain, "sent": ex.sentence, "triples": ex.triples} for ex in examples]
    (args.out_dir / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    metrics_rows: List[Dict[str, Any]] = []
    for method in args.methods:
        predictions, scores = run_method(
            examples,
            method,
            client,
            args.out_dir,
            args.mock,
            self_consistency_k=args.self_consistency_k,
            self_consistency_vote=args.self_consistency_vote,
        )
        failures = sum(1 for row in predictions if row.get("_failed"))
        metrics = aggregate(scores, failures)
        calls_per_instance = (
            float(args.self_consistency_k)
            if method == "self_consistency_5"
            else float(CALLS_PER_INSTANCE.get(method, 1))
        )
        row = {
            "dataset": args.dataset,
            "domains": ",".join(args.domains),
            "method": method,
            "calls_per_instance": calls_per_instance,
            **metrics,
        }
        metrics_rows.append(row)
        print(
            f"{METHOD_LABELS[method]:12s} "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
            f"Calls={calls_per_instance:.1f} "
            f"Ont={metrics['ontology_conformance']:.3f} Ground={metrics['grounding']:.3f} "
            f"Failed={int(metrics['failures'])}/{len(examples)}",
            flush=True,
        )

    write_metrics(metrics_rows, args.out_dir / "metrics.csv")
    write_main_table(metrics_rows, args.out_dir / "main_results_table.tex")
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
