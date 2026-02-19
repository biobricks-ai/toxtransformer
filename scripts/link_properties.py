#!/usr/bin/env python3
"""
Claude-based property linker for ToxTransformer tokens.

Uses Anthropic SDK tool_use to semantically match external benchmark endpoints
to ToxTransformer property tokens. Claude iteratively searches the token catalog,
examines token details, and selects matching tokens with construct validity reasoning.

Usage:
    python scripts/link_properties.py
    python scripts/link_properties.py --endpoints "admet-huggingface||admet-huggingface/admet/AMES"
    python scripts/link_properties.py --model claude-sonnet-4-20250514 --no-skip-existing
"""

import argparse
import json
import logging
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_DB_PATH = "cache/build_sqlite/cvae.sqlite"
DEFAULT_BENCHMARK_PATH = "publication/benchmark/data/external_binary_benchmark.parquet"
DEFAULT_OUTPUT_PATH = "cache/property_linkages.json"
DEFAULT_HOLDOUT_PATH = "publication/benchmark/results/table_per_property_bootstrap_holdout.csv"
MAX_TOOL_TURNS = 10


# ---------------------------------------------------------------------------
# Token Catalog — searchable index of all 6,647 ToxTransformer tokens
# ---------------------------------------------------------------------------

@dataclass
class TokenInfo:
    token_id: int
    title: Optional[str]
    bioassay_name: Optional[str]
    source: str
    categories: List[str]
    category_strengths: List[float]
    positive_count: int
    negative_count: int
    positive_rate: float
    metadata: Dict[str, Any]
    holdout_auc: Optional[float] = None  # internal holdout AUC (single-property)
    bootstrap_auc: Optional[float] = None  # bootstrap AUC (single-property)
    context_auc: Optional[float] = None  # holdout AUC with 20 context properties


class TokenCatalog:
    """Searchable catalog of all ToxTransformer property tokens."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, holdout_path: str = DEFAULT_HOLDOUT_PATH):
        self.db_path = db_path
        self.holdout_path = holdout_path
        self._tokens: Dict[int, TokenInfo] = {}
        self._categories: Dict[str, int] = {}
        self._load()

    def _load(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Load categories
        cursor = conn.execute("SELECT category_id, category FROM category")
        cat_map = {row["category_id"]: row["category"] for row in cursor}
        self._categories = {}

        # Load properties with categories
        cursor = conn.execute("""
            SELECT
                p.property_token,
                p.title,
                p.data,
                s.source,
                GROUP_CONCAT(c.category || '||' || pc.strength, ';;') as categories
            FROM property p
            LEFT JOIN source s ON p.source_id = s.source_id
            LEFT JOIN property_category pc ON p.property_id = pc.property_id
            LEFT JOIN category c ON pc.category_id = c.category_id
            GROUP BY p.property_token
        """)

        for row in cursor:
            token_id = int(row["property_token"])
            categories = []
            strengths = []
            if row["categories"]:
                for cat_str in row["categories"].split(";;"):
                    if "||" in cat_str:
                        cat, strength = cat_str.split("||", 1)
                        categories.append(cat)
                        try:
                            strengths.append(float(strength))
                        except ValueError:
                            strengths.append(0.0)

            metadata = {}
            if row["data"]:
                try:
                    metadata = json.loads(row["data"])
                except json.JSONDecodeError:
                    pass

            self._tokens[token_id] = TokenInfo(
                token_id=token_id,
                title=row["title"],
                bioassay_name=metadata.get("BioAssay Name"),
                source=row["source"] or "unknown",
                categories=categories,
                category_strengths=strengths,
                positive_count=0,
                negative_count=0,
                positive_rate=0.0,
                metadata=metadata,
            )

        # Load summary statistics
        cursor = conn.execute("""
            SELECT p.property_token, pss.positive_count, pss.negative_count
            FROM property_summary_statistics pss
            JOIN property p ON pss.property_id = p.property_id
        """)
        for row in cursor:
            token_id = int(row["property_token"])
            if token_id in self._tokens:
                pos = row["positive_count"] or 0
                neg = row["negative_count"] or 0
                total = pos + neg
                self._tokens[token_id].positive_count = pos
                self._tokens[token_id].negative_count = neg
                self._tokens[token_id].positive_rate = pos / total if total > 0 else 0.0

        conn.close()

        # Build category counts
        for t in self._tokens.values():
            for cat in t.categories:
                self._categories[cat] = self._categories.get(cat, 0) + 1

        # Load internal holdout/bootstrap AUC data
        n_perf = 0
        if Path(self.holdout_path).exists():
            perf_df = pd.read_csv(self.holdout_path)
            for _, row in perf_df.iterrows():
                tid = int(row["property_token"])
                if tid in self._tokens:
                    t = self._tokens[tid]
                    t.holdout_auc = float(row["holdout_nprops1"]) if pd.notna(row.get("holdout_nprops1")) else None
                    t.bootstrap_auc = float(row["bootstrap_nprops_1"]) if pd.notna(row.get("bootstrap_nprops_1")) else None
                    t.context_auc = float(row["holdout_nprops20"]) if pd.notna(row.get("holdout_nprops20")) else None
                    n_perf += 1
            log.info(f"Loaded internal AUC data for {n_perf} tokens")
        else:
            log.warning(f"No holdout AUC data at {self.holdout_path}")

        log.info(f"Loaded {len(self._tokens)} tokens, {len(self._categories)} categories")

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        source: Optional[str] = None,
        max_results: int = 30,
    ) -> List[Dict]:
        """Keyword search over token titles and BioAssay names."""
        query_lower = query.lower()
        results = []

        for t in self._tokens.values():
            # Filter by category/source
            if category and category.lower() not in [c.lower() for c in t.categories]:
                continue
            if source and source.lower() != t.source.lower():
                continue

            # Search in title and bioassay name
            text = ""
            if t.title:
                text += t.title.lower()
            if t.bioassay_name:
                text += " " + t.bioassay_name.lower()

            if query_lower in text:
                results.append(self._token_to_summary(t))

        # Sort by relevance (exact match in title first, then by positive_rate)
        results.sort(key=lambda x: (
            query_lower not in (x.get("title") or "").lower(),
            -x.get("positive_rate", 0),
        ))

        return results[:max_results]

    def get_details(self, token_ids: List[int]) -> List[Dict]:
        """Get full details for specific tokens."""
        results = []
        for tid in token_ids:
            t = self._tokens.get(tid)
            if t:
                results.append(self._token_to_detail(t))
        return results

    def list_categories(self) -> Dict[str, int]:
        """List all categories with counts."""
        return dict(sorted(self._categories.items(), key=lambda x: -x[1]))

    def search_by_performance(
        self,
        min_auc: float = 0.8,
        category: Optional[str] = None,
        source: Optional[str] = None,
        max_results: int = 30,
    ) -> List[Dict]:
        """Find tokens with high internal holdout AUC, optionally filtered."""
        results = []
        for t in self._tokens.values():
            if t.holdout_auc is None or t.holdout_auc < min_auc:
                continue
            if category and category.lower() not in [c.lower() for c in t.categories]:
                continue
            if source and source.lower() != t.source.lower():
                continue
            results.append(self._token_to_summary(t))

        results.sort(key=lambda x: -(x.get("holdout_auc") or 0))
        return results[:max_results]

    def _token_to_summary(self, t: TokenInfo) -> Dict:
        d = {
            "token_id": t.token_id,
            "title": t.title,
            "source": t.source,
            "categories": t.categories[:3],
            "positive_rate": round(t.positive_rate, 4),
            "training_samples": t.positive_count + t.negative_count,
        }
        if t.holdout_auc is not None:
            d["holdout_auc"] = round(t.holdout_auc, 4)
        return d

    def _token_to_detail(self, t: TokenInfo) -> Dict:
        d = {
            "token_id": t.token_id,
            "title": t.title,
            "bioassay_name": t.bioassay_name,
            "source": t.source,
            "categories": t.categories,
            "category_strengths": t.category_strengths,
            "positive_count": t.positive_count,
            "negative_count": t.negative_count,
            "positive_rate": round(t.positive_rate, 4),
            "total_samples": t.positive_count + t.negative_count,
        }
        if t.holdout_auc is not None:
            d["holdout_auc"] = round(t.holdout_auc, 4)
        if t.bootstrap_auc is not None:
            d["bootstrap_auc"] = round(t.bootstrap_auc, 4)
        if t.context_auc is not None:
            d["context_auc_20props"] = round(t.context_auc, 4)
        # Quality tier for quick assessment
        if t.holdout_auc is not None:
            if t.holdout_auc >= 0.9:
                d["model_confidence"] = "excellent"
            elif t.holdout_auc >= 0.8:
                d["model_confidence"] = "good"
            elif t.holdout_auc >= 0.7:
                d["model_confidence"] = "moderate"
            else:
                d["model_confidence"] = "weak"
        else:
            d["model_confidence"] = "not_evaluated"
        return d


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_tokens",
        "description": (
            "Search ToxTransformer token catalog by keyword. "
            "Searches over token titles and BioAssay names. "
            "Returns up to 30 matching tokens with summary info including "
            "holdout_auc (internal cross-validation AUC) and training_samples. "
            "Use multiple searches with different keywords to find relevant tokens."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword to search for (e.g., 'ames', 'herg', 'cyp3a4', 'hepat', 'zebrafish')",
                },
                "category": {
                    "type": "string",
                    "description": "Optional: filter by category name (e.g., 'cardiotoxicity', 'genotoxicity')",
                },
                "source": {
                    "type": "string",
                    "description": "Optional: filter by data source (pubchem, toxcast, tox21, chembl, bindingdb, ice, etc.)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_token_details",
        "description": (
            "Get full metadata for specific token IDs. "
            "Returns title, BioAssay name, source, all categories with strengths, "
            "positive/negative counts, positive rate, holdout_auc (internal cross-validation "
            "performance), bootstrap_auc, context_auc_20props, and model_confidence tier "
            "(excellent/good/moderate/weak/not_evaluated)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "token_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of token IDs to look up",
                },
            },
            "required": ["token_ids"],
        },
    },
    {
        "name": "search_high_confidence_tokens",
        "description": (
            "Find tokens where ToxTransformer has high internal performance (holdout AUC). "
            "Only returns tokens with evaluated holdout AUC above the specified threshold. "
            "Use this to find tokens the model is genuinely good at predicting. "
            "Tokens with holdout_auc >= 0.9 are 'excellent', >= 0.8 are 'good'. "
            "Can filter by category or source."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_auc": {
                    "type": "number",
                    "description": "Minimum holdout AUC threshold (default: 0.8). Range: 0.5-1.0",
                },
                "category": {
                    "type": "string",
                    "description": "Optional: filter by category (e.g., 'hepatotoxicity', 'cardiotoxicity')",
                },
                "source": {
                    "type": "string",
                    "description": "Optional: filter by data source",
                },
            },
        },
    },
    {
        "name": "list_categories",
        "description": (
            "List all 37 toxicity categories with the number of tokens in each. "
            "Useful for understanding what kinds of assays are available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "submit_linkage",
        "description": (
            "Submit the final token selection for this endpoint. "
            "Call this once you have identified the best matching tokens."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "token_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Selected token IDs that match this endpoint",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low", "none"],
                    "description": (
                        "Confidence in the match quality. "
                        "'high' = direct assay match AND model has good internal AUC (>=0.8). "
                        "'medium' = same mechanism/target but different assay format, or direct match with weaker AUC. "
                        "'low' = related biology but uncertain construct validity, or low internal AUC. "
                        "'none' = no suitable tokens found."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why these tokens were selected, including internal AUC considerations",
                },
                "construct_notes": {
                    "type": "string",
                    "description": "Notes on construct validity: differences between the external endpoint and the selected tokens (e.g., in-vitro vs clinical, different species, different assay format)",
                },
                "expected_quality": {
                    "type": "string",
                    "enum": ["strong", "moderate", "uncertain"],
                    "description": (
                        "Expected prediction quality based on both construct match AND internal model performance. "
                        "'strong' = good construct match + high internal AUC (>=0.85). "
                        "'moderate' = decent match + reasonable AUC, or excellent match + moderate AUC. "
                        "'uncertain' = weak construct or low AUC or no AUC data."
                    ),
                },
            },
            "required": ["token_ids", "confidence", "reasoning", "construct_notes", "expected_quality"],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert toxicologist and cheminformatics scientist. Your task is to link \
external toxicity benchmark endpoints to ToxTransformer property tokens.

ToxTransformer is a decoder-only transformer that predicts 6,647 binary toxicity \
properties from molecular structure (SELFIES). Each property token corresponds to a \
specific bioassay (mostly from PubChem, ToxCast, ChEMBL, BindingDB, ICE, Tox21, SIDER). \
You need to find tokens that measure the SAME or very similar biological endpoint as \
the external benchmark.

## Internal model performance data

Many tokens include **holdout_auc** — the model's cross-validated AUC on held-out \
molecules for that specific token. This tells you how well ToxTransformer has actually \
learned to predict that property:
- **holdout_auc >= 0.90** ("excellent"): model is highly confident, predictions are reliable
- **holdout_auc 0.80-0.89** ("good"): model is reasonably good at this property
- **holdout_auc 0.70-0.79** ("moderate"): model has learned something but predictions are noisy
- **holdout_auc < 0.70** ("weak"): model struggles with this property — external predictions will likely be poor
- **not_evaluated**: no holdout data (older tokens); treat as unknown quality

## First-principles reasoning

For zero-shot transfer to work, TWO conditions must hold:
1. **Construct validity**: the token must measure the same (or very similar) biology as the external endpoint
2. **Model competence**: ToxTransformer must actually be good at predicting that token internally

A perfect construct match on a token the model can't predict (low AUC) will fail. \
Conversely, a token the model predicts well but that measures different biology will also fail. \
BOTH matter.

**When choosing between candidate tokens:**
- Prefer tokens with holdout_auc >= 0.80 over those with lower or unknown AUC
- A token with holdout_auc 0.92 and slightly broader construct is often better than \
  one with holdout_auc 0.65 and perfect construct match
- If all matching tokens have weak AUC (<0.70), consider confidence=low regardless of construct match
- Anti-predictive tokens (ones that measure the OPPOSITE of what you need) are DANGEROUS — \
  including them will flip the signal and produce AUC < 0.5. Exclude these entirely.

## Tools available
- `search_tokens`: keyword search over token titles/BioAssay names (includes holdout_auc)
- `get_token_details`: get full metadata + internal AUC + model confidence tier
- `search_high_confidence_tokens`: find tokens with high internal AUC, filtered by category/source
- `list_categories`: see all 37 categories with counts
- `submit_linkage`: submit your final token selection

## Strategy
1. Start by understanding what the external endpoint measures
2. Search with multiple relevant keywords (target name, mechanism, assay type)
3. Use `search_high_confidence_tokens` to find well-predicted tokens in relevant categories
4. Examine promising tokens in detail — check BOTH construct match AND holdout_auc
5. Consider construct validity carefully:
   - Prefer DIRECT assay matches (same target, same assay type)
   - Be cautious with mechanistic proxies (e.g., hERG for cardiotoxicity is OK, but general cytotoxicity for organ toxicity is NOT)
   - Note differences: in-vitro vs in-vivo, different species, different assay format
   - Check positive rates — very different rates suggest different constructs
   - Watch for SUBSTRATE vs INHIBITOR confusion (e.g., CYP substrate tokens should NOT be used for inhibition endpoints)
6. Set expected_quality based on combined construct + AUC assessment
7. Submit your selection with confidence level and reasoning

## Confidence levels
- **high**: Direct assay match AND model has good internal AUC (>=0.8)
- **medium**: Same mechanism/target but different assay format, OR direct match with moderate AUC (0.7-0.8)
- **low**: Related biology but uncertain construct validity, OR low internal AUC (<0.7)
- **none**: No suitable tokens found — better to return empty than force a bad match

## Expected quality levels (for submit_linkage)
- **strong**: Good construct match + high internal AUC (>=0.85). Expect good zero-shot transfer.
- **moderate**: Decent match + reasonable AUC, or excellent match + moderate AUC. Transfer may work.
- **uncertain**: Weak construct match, low AUC, or no AUC data. Zero-shot transfer unreliable.

## Critical warnings
- It's better to return NO tokens (confidence=none) than to force a bad match
- Averaging unrelated tokens will produce noise, not signal
- Including anti-predictive tokens (e.g., cytotoxicity tokens for organ toxicity) can produce AUC < 0.5
- If the endpoint measures a clinical/in-vivo outcome (e.g., "liver toxicity from clinical trials"), \
in-vitro HepG2 viability tokens are a WEAK proxy — use confidence=low
- CYP substrate tokens are NOT the same as CYP inhibitor tokens — never mix these
- SIDER organ-level tokens (hepatobiliary, cardiac, etc.) capture clinical adverse events which are \
broader than any single mechanism — they may be the best match for clinical toxicity endpoints but \
check their internal AUC
- Always call submit_linkage when done, even if with empty token_ids and confidence=none
"""


# ---------------------------------------------------------------------------
# Tool execution helper (shared by all backends)
# ---------------------------------------------------------------------------

def execute_tool(catalog: TokenCatalog, tool_name: str, tool_input: Dict, turn: int) -> tuple:
    """Execute a tool call and return (result_json, submitted_dict_or_none).

    Returns (result_text, submitted_dict_or_None).
    """
    # Defensive: convert MapComposite or protobuf-like objects to plain dicts
    if not isinstance(tool_input, dict):
        try:
            tool_input = dict(tool_input)
        except Exception:
            tool_input = {}

    try:
        if tool_name == "search_tokens":
            result = catalog.search(
                query=tool_input["query"],
                category=tool_input.get("category"),
                source=tool_input.get("source"),
            )
            log.info(f"  Turn {turn}: search_tokens('{tool_input['query']}') -> {len(result)} results")
            return json.dumps(result, indent=2), None

        elif tool_name == "get_token_details":
            token_ids = tool_input["token_ids"]
            if not isinstance(token_ids, list):
                token_ids = [int(token_ids)]
            else:
                token_ids = [int(t) for t in token_ids]
            result = catalog.get_details(token_ids)
            log.info(f"  Turn {turn}: get_token_details({token_ids[:5]}...) -> {len(result)} results")
            return json.dumps(result, indent=2), None

        elif tool_name == "search_high_confidence_tokens":
            result = catalog.search_by_performance(
                min_auc=float(tool_input.get("min_auc", 0.8)),
                category=tool_input.get("category"),
                source=tool_input.get("source"),
            )
            log.info(f"  Turn {turn}: search_high_confidence_tokens(min_auc={tool_input.get('min_auc', 0.8)}, category={tool_input.get('category')}) -> {len(result)} results")
            return json.dumps(result, indent=2), None

        elif tool_name == "list_categories":
            result = catalog.list_categories()
            log.info(f"  Turn {turn}: list_categories() -> {len(result)} categories")
            return json.dumps(result, indent=2), None

        elif tool_name == "submit_linkage":
            token_ids = tool_input.get("token_ids", [])
            if not isinstance(token_ids, list):
                token_ids = [int(token_ids)]
            else:
                token_ids = [int(t) for t in token_ids]
            confidence = tool_input.get("confidence", "none")
            reasoning = tool_input.get("reasoning", "")
            construct_notes = tool_input.get("construct_notes", "")
            expected_quality = tool_input.get("expected_quality", "uncertain")
            log.info(
                f"  Turn {turn}: submit_linkage("
                f"tokens={token_ids}, "
                f"confidence={confidence}, "
                f"expected_quality={expected_quality})"
            )
            submitted = {
                "tokens": token_ids,
                "reasoning": reasoning,
                "confidence": confidence,
                "construct_notes": construct_notes,
                "expected_quality": expected_quality,
            }
            return "Linkage submitted successfully.", submitted

        else:
            return f"Unknown tool: {tool_name}", None

    except Exception as e:
        log.warning(f"  Turn {turn}: Tool {tool_name} error: {e} (input: {tool_input})")
        return f"Error executing {tool_name}: {e}. Please try again with valid arguments.", None


def _build_user_message(endpoint_key: str, endpoint_info: Dict) -> str:
    source = endpoint_info["source"]
    prop = endpoint_info["property"]
    n_samples = endpoint_info["n_samples"]
    n_pos = endpoint_info["n_pos"]
    pos_rate = n_pos / n_samples if n_samples > 0 else 0
    return (
        f"Link this external benchmark endpoint to ToxTransformer tokens:\n\n"
        f"**Endpoint key**: `{endpoint_key}`\n"
        f"**Source dataset**: {source}\n"
        f"**Property path**: {prop}\n"
        f"**Samples**: {n_samples} total, {n_pos} positive ({pos_rate:.1%} positive rate)\n\n"
        f"Search for matching tokens and submit your selection."
    )


def _make_result(endpoint_key: str, endpoint_info: Dict, submitted: Optional[Dict] = None, reason: str = "") -> Dict:
    base = {
        "source": endpoint_info["source"],
        "property": endpoint_info["property"],
        "name": endpoint_key,
    }
    if submitted:
        base.update(submitted)
    else:
        base.update({
            "tokens": [],
            "reasoning": reason or "No linkage submitted",
            "confidence": "none",
            "construct_notes": "",
        })
    return base


# ---------------------------------------------------------------------------
# Anthropic/Vertex backend
# ---------------------------------------------------------------------------

def run_linkage_conversation(
    client,
    model: str,
    catalog: TokenCatalog,
    endpoint_key: str,
    endpoint_info: Dict,
) -> Dict:
    """Run a tool-use conversation to link one endpoint to tokens (Anthropic/Vertex)."""
    user_message = _build_user_message(endpoint_key, endpoint_info)

    messages = [{"role": "user", "content": user_message}]

    for turn in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_uses = [b for b in assistant_content if b.type == "tool_use"]
        if not tool_uses:
            log.warning(f"  Turn {turn+1}: No tool calls, forcing none result")
            return _make_result(endpoint_key, endpoint_info, reason="LLM did not submit a linkage")

        tool_results = []
        submitted = None

        for tool_use in tool_uses:
            result_text, sub = execute_tool(catalog, tool_use.name, tool_use.input, turn + 1)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result_text,
            })
            if sub:
                submitted = sub

        messages.append({"role": "user", "content": tool_results})

        if submitted:
            return _make_result(endpoint_key, endpoint_info, submitted)

    log.warning(f"  Exhausted {MAX_TOOL_TURNS} turns without submission")
    return _make_result(endpoint_key, endpoint_info, reason=f"Exhausted {MAX_TOOL_TURNS} tool-use turns")


# ---------------------------------------------------------------------------
# Google AI (Gemini) backend
# ---------------------------------------------------------------------------

def _tools_to_gemini_declarations() -> list:
    """Convert our Anthropic-format tool definitions to Gemini function declarations."""
    declarations = []
    for tool in TOOLS:
        # Convert Anthropic input_schema to Gemini-compatible schema
        schema = tool["input_schema"].copy()
        # Gemini doesn't support top-level "required" in the same way for function params
        # but the google-genai SDK handles it
        declarations.append({
            "name": tool["name"],
            "description": tool["description"],
            "parameters": schema,
        })
    return declarations


def run_linkage_conversation_gemini(
    client,
    model: str,
    catalog: TokenCatalog,
    endpoint_key: str,
    endpoint_info: Dict,
) -> Dict:
    """Run a tool-use conversation using Google AI (Gemini)."""
    from google.genai import types

    user_message = _build_user_message(endpoint_key, endpoint_info)

    gemini_tools = [types.Tool(function_declarations=_tools_to_gemini_declarations())]

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_message)]),
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=gemini_tools,
        max_output_tokens=4096,
    )

    for turn in range(MAX_TOOL_TURNS):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # Append assistant response
        contents.append(response.candidates[0].content)

        # Check for function calls
        function_calls = []
        for part in response.candidates[0].content.parts:
            if part.function_call:
                function_calls.append(part)

        if not function_calls:
            # Check if there was a text response with submit info (sometimes Gemini does this)
            log.warning(f"  Turn {turn+1}: No function calls, forcing none result")
            return _make_result(endpoint_key, endpoint_info, reason="LLM did not submit a linkage")

        # Execute function calls and build response
        function_response_parts = []
        submitted = None

        for fc_part in function_calls:
            fc = fc_part.function_call
            tool_input = dict(fc.args) if fc.args else {}
            result_text, sub = execute_tool(catalog, fc.name, tool_input, turn + 1)
            function_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result_text},
                )
            )
            if sub:
                submitted = sub

        contents.append(types.Content(role="user", parts=function_response_parts))

        if submitted:
            return _make_result(endpoint_key, endpoint_info, submitted)

    log.warning(f"  Exhausted {MAX_TOOL_TURNS} turns without submission")
    return _make_result(endpoint_key, endpoint_info, reason=f"Exhausted {MAX_TOOL_TURNS} tool-use turns")


# ---------------------------------------------------------------------------
# Load endpoints
# ---------------------------------------------------------------------------

def load_endpoints(benchmark_path: str, filter_keys: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Load viable endpoints from benchmark parquet."""
    df = pd.read_parquet(benchmark_path)

    endpoints_agg = df.groupby(["source", "property"]).agg(
        n_samples=("value", "count"),
        n_pos=("value", "sum"),
    ).reset_index()
    endpoints_agg["n_neg"] = endpoints_agg["n_samples"] - endpoints_agg["n_pos"]
    endpoints_agg["n_pos"] = endpoints_agg["n_pos"].astype(int)
    endpoints_agg["n_neg"] = endpoints_agg["n_neg"].astype(int)

    # Filter viable: >=20 samples, >=5 per class
    viable = endpoints_agg[
        (endpoints_agg["n_samples"] >= 20)
        & (endpoints_agg["n_pos"] >= 5)
        & (endpoints_agg["n_neg"] >= 5)
    ]

    result = {}
    for _, row in viable.iterrows():
        key = f"{row['source']}||{row['property']}"
        result[key] = {
            "source": row["source"],
            "property": row["property"],
            "n_samples": int(row["n_samples"]),
            "n_pos": int(row["n_pos"]),
            "n_neg": int(row["n_neg"]),
        }

    if filter_keys:
        result = {k: v for k, v in result.items() if k in filter_keys}

    log.info(f"Loaded {len(result)} viable endpoints")
    return result


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------

def atomic_save(data: Dict, path: str):
    """Save JSON atomically via temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Link external endpoints to ToxTransformer tokens using Claude")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to cvae.sqlite")
    parser.add_argument("--holdout", default=DEFAULT_HOLDOUT_PATH, help="Path to holdout AUC CSV")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_PATH, help="Path to benchmark parquet")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSON path")
    parser.add_argument("--endpoints", type=str, help="Comma-separated endpoint keys to process (default: all)")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip already-linked endpoints (default: true)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-process all endpoints")
    parser.add_argument("--vertex", action="store_true", help="Use Vertex AI instead of Anthropic API")
    parser.add_argument("--vertex-project", default="toxindex", help="GCP project for Vertex AI")
    parser.add_argument("--vertex-region", default="us-east5", help="GCP region for Vertex AI")
    parser.add_argument("--google-ai", action="store_true", help="Use Google AI (Gemini) instead of Anthropic/Vertex")
    parser.add_argument("--google-ai-key", type=str, help="Google AI API key (or set GOOGLE_AI_API_KEY env var)")
    args = parser.parse_args()

    if args.no_skip_existing:
        args.skip_existing = False

    # Load catalog
    log.info("Loading token catalog...")
    catalog = TokenCatalog(args.db, holdout_path=args.holdout)

    # Load endpoints
    filter_keys = None
    if args.endpoints:
        filter_keys = [k.strip() for k in args.endpoints.split(",")]
    endpoints = load_endpoints(args.benchmark, filter_keys)

    if not endpoints:
        log.error("No endpoints to process")
        return

    # Load existing linkages
    output_path = Path(args.output)
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        log.info(f"Loaded {len(existing)} existing linkages from {output_path}")

    # Init client based on backend
    use_gemini = args.google_ai
    if use_gemini:
        from google import genai
        api_key = args.google_ai_key or os.environ.get("GOOGLE_AI_API_KEY")
        if not api_key:
            log.error("Google AI API key required. Pass --google-ai-key or set GOOGLE_AI_API_KEY")
            return
        client = genai.Client(api_key=api_key)
        if not args.model or args.model == DEFAULT_MODEL:
            args.model = "gemini-2.5-flash"
        log.info(f"Using Google AI (Gemini) with model={args.model}")
    elif args.vertex:
        import anthropic
        client = anthropic.AnthropicVertex(
            project_id=args.vertex_project,
            region=args.vertex_region,
        )
        log.info(f"Using Vertex AI (project={args.vertex_project}, region={args.vertex_region})")
    else:
        import anthropic
        client = anthropic.Anthropic()

    # Process each endpoint
    to_process = []
    for key, info in sorted(endpoints.items()):
        if args.skip_existing and key in existing:
            log.info(f"Skipping (already linked): {key}")
            continue
        to_process.append((key, info))

    log.info(f"Processing {len(to_process)} endpoints with model={args.model}")

    run_fn = run_linkage_conversation_gemini if use_gemini else run_linkage_conversation

    for i, (key, info) in enumerate(to_process):
        log.info(f"\n[{i+1}/{len(to_process)}] Linking: {key}")
        log.info(f"  Samples: {info['n_samples']} (pos={info['n_pos']}, neg={info['n_neg']})")

        try:
            result = run_fn(client, args.model, catalog, key, info)
            existing[key] = result
            atomic_save(existing, str(output_path))
            log.info(
                f"  Result: confidence={result['confidence']}, "
                f"tokens={len(result['tokens'])}, "
                f"reasoning={result['reasoning'][:100]}..."
            )
        except Exception as e:
            log.error(f"  Error: {e}", exc_info=True)
            existing[key] = _make_result(key, info, reason=f"Error: {e}")
            atomic_save(existing, str(output_path))

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    conf_counts = {}
    for v in existing.values():
        c = v.get("confidence", "unknown")
        conf_counts[c] = conf_counts.get(c, 0) + 1
    for c, n in sorted(conf_counts.items()):
        log.info(f"  {c}: {n}")
    log.info(f"  Total: {len(existing)}")
    log.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
