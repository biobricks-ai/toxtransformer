"""
Semantic Feature Selector

Uses LLM (Gemini) to select ToxTransformer features that are semantically
relevant to a target property, reducing overfitting through informed feature selection.
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)


@dataclass
class PropertyMetadata:
    """Metadata for a ToxTransformer property."""
    property_token: int
    title: Optional[str]
    categories: List[str]
    category_strengths: List[str]
    source: str
    metadata: Dict[str, Any]


@dataclass
class SemanticSelectionResult:
    """Result of semantic feature selection."""
    selected_tokens: List[int]
    selected_categories: List[str]
    reasoning: str
    n_selected: int
    n_total: int


class ToxTransformerMetadataLoader:
    """Loads and provides access to ToxTransformer property metadata."""

    def __init__(self, db_path: str = "brick/cvae.sqlite"):
        self.db_path = db_path
        self._properties: Optional[Dict[int, PropertyMetadata]] = None
        self._categories: Optional[List[str]] = None
        self._token_to_categories: Optional[Dict[int, List[str]]] = None

    def _load_metadata(self):
        """Load all property metadata from database."""
        if self._properties is not None:
            return

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Load all categories
        cursor = conn.execute("SELECT category_id, category FROM category")
        category_map = {row["category_id"]: row["category"] for row in cursor}
        self._categories = list(category_map.values())

        # Load properties with their categories
        cursor = conn.execute("""
            SELECT
                p.property_token,
                p.title,
                p.data,
                s.source,
                GROUP_CONCAT(c.category || '::' || pc.strength) as categories
            FROM property p
            LEFT JOIN source s ON p.source_id = s.source_id
            LEFT JOIN property_category pc ON p.property_id = pc.property_id
            LEFT JOIN category c ON pc.category_id = c.category_id
            GROUP BY p.property_token
        """)

        self._properties = {}
        self._token_to_categories = {}

        for row in cursor:
            token = int(row["property_token"])

            # Parse categories
            categories = []
            strengths = []
            if row["categories"]:
                for cat_str in row["categories"].split(","):
                    if "::" in cat_str:
                        cat, strength = cat_str.split("::")
                        categories.append(cat)
                        strengths.append(strength)

            # Parse metadata JSON
            metadata = {}
            if row["data"]:
                try:
                    metadata = json.loads(row["data"])
                except json.JSONDecodeError:
                    pass

            prop = PropertyMetadata(
                property_token=token,
                title=row["title"],
                categories=categories,
                category_strengths=strengths,
                source=row["source"] or "unknown",
                metadata=metadata,
            )

            self._properties[token] = prop
            self._token_to_categories[token] = categories

        conn.close()
        logging.info(f"Loaded metadata for {len(self._properties)} properties")

    def get_all_categories(self) -> List[str]:
        """Get list of all toxicity categories."""
        self._load_metadata()
        return self._categories

    def get_property(self, token: int) -> Optional[PropertyMetadata]:
        """Get metadata for a specific property token."""
        self._load_metadata()
        return self._properties.get(token)

    def get_all_properties(self) -> Dict[int, PropertyMetadata]:
        """Get all property metadata."""
        self._load_metadata()
        return self._properties

    def get_tokens_by_category(self, category: str) -> List[int]:
        """Get all property tokens in a category."""
        self._load_metadata()
        tokens = []
        for token, cats in self._token_to_categories.items():
            if category.lower() in [c.lower() for c in cats]:
                tokens.append(token)
        return tokens

    def get_category_summary(self) -> Dict[str, int]:
        """Get count of properties per category."""
        self._load_metadata()
        summary = {}
        for cats in self._token_to_categories.values():
            for cat in cats:
                summary[cat] = summary.get(cat, 0) + 1
        return dict(sorted(summary.items(), key=lambda x: -x[1]))

    def format_for_llm(self, max_examples: int = 5) -> str:
        """Format metadata summary for LLM consumption."""
        self._load_metadata()

        lines = []
        lines.append("## ToxTransformer Property Categories\n")

        category_summary = self.get_category_summary()
        for cat, count in category_summary.items():
            lines.append(f"- **{cat}**: {count} properties")

            # Add example titles
            tokens = self.get_tokens_by_category(cat)[:max_examples]
            for token in tokens:
                prop = self._properties.get(token)
                if prop and prop.title:
                    lines.append(f"  - Token {token}: {prop.title[:80]}...")

        return "\n".join(lines)


class SemanticFeatureSelector:
    """
    Selects ToxTransformer features using LLM-based semantic matching.

    Uses Gemini to identify which property categories and specific tokens
    are most relevant to a target toxicity endpoint.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        db_path: str = "brick/cvae.sqlite",
        api_key: Optional[str] = None,
        use_vertex_ai: bool = False,
        vertex_project: Optional[str] = None,
        vertex_location: str = "us-central1",
    ):
        """
        Initialize the semantic selector.

        Args:
            model: Gemini model to use.
            db_path: Path to ToxTransformer database.
            api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
            use_vertex_ai: Use Vertex AI instead of Google AI API.
            vertex_project: GCP project ID for Vertex AI.
            vertex_location: GCP region for Vertex AI.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.use_vertex_ai = use_vertex_ai
        self.vertex_project = vertex_project or os.environ.get("GOOGLE_CLOUD_PROJECT", "insilica-internal")
        self.vertex_location = vertex_location
        self.metadata_loader = ToxTransformerMetadataLoader(db_path)

        if not self.api_key and not self.use_vertex_ai:
            # Try to auto-detect if we should use Vertex AI
            try:
                import google.auth
                credentials, project = google.auth.default()
                if project:
                    logging.info(f"Auto-detected GCP project: {project}. Using Vertex AI.")
                    self.use_vertex_ai = True
                    self.vertex_project = project
            except Exception:
                logging.warning(
                    "No Google API key found and Vertex AI not available. "
                    "Set GOOGLE_API_KEY or authenticate with GCP."
                )

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with a prompt."""
        if self.use_vertex_ai:
            return self._call_vertex_ai(prompt)

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for semantic selection. "
                "Install with: pip install google-generativeai"
            )

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        response = model.generate_content(prompt)
        return response.text

    def _call_vertex_ai(self, prompt: str) -> str:
        """Call Vertex AI Gemini API with a prompt."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform is required for Vertex AI. "
                "Install with: pip install google-cloud-aiplatform"
            )

        vertexai.init(project=self.vertex_project, location=self.vertex_location)

        # Map model names to Vertex AI format
        model_name = self.model
        if not model_name.startswith("gemini-"):
            model_name = f"gemini-{model_name}"

        logging.info(f"Using Vertex AI with project={self.vertex_project}, model={model_name}")
        model = GenerativeModel(model_name)

        response = model.generate_content(prompt)
        return response.text

    def select_features(
        self,
        target_property: str,
        target_description: Optional[str] = None,
        max_categories: int = 10,
        max_tokens_per_category: int = 100,
        include_related: bool = True,
    ) -> SemanticSelectionResult:
        """
        Select relevant features for a target property using LLM.

        Args:
            target_property: Name of the target property (e.g., "liver toxicity").
            target_description: Optional detailed description of what we're predicting.
            max_categories: Maximum number of categories to select.
            max_tokens_per_category: Maximum tokens to include per category.
            include_related: Include tangentially related categories.

        Returns:
            SemanticSelectionResult with selected tokens and reasoning.
        """
        # Get available categories
        category_summary = self.metadata_loader.get_category_summary()

        # Build prompt
        prompt = self._build_selection_prompt(
            target_property=target_property,
            target_description=target_description,
            category_summary=category_summary,
            max_categories=max_categories,
            include_related=include_related,
        )

        # Call LLM
        logging.info(f"Calling {self.model} for semantic feature selection...")
        response = self._call_gemini(prompt)

        # Parse response
        selected_categories, reasoning = self._parse_response(response, category_summary)

        # Get tokens for selected categories
        selected_tokens = []
        for category in selected_categories:
            tokens = self.metadata_loader.get_tokens_by_category(category)
            selected_tokens.extend(tokens[:max_tokens_per_category])

        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for t in selected_tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

        all_properties = self.metadata_loader.get_all_properties()

        result = SemanticSelectionResult(
            selected_tokens=unique_tokens,
            selected_categories=selected_categories,
            reasoning=reasoning,
            n_selected=len(unique_tokens),
            n_total=len(all_properties),
        )

        logging.info(
            f"Selected {result.n_selected}/{result.n_total} features "
            f"from {len(selected_categories)} categories"
        )

        return result

    def _build_selection_prompt(
        self,
        target_property: str,
        target_description: Optional[str],
        category_summary: Dict[str, int],
        max_categories: int,
        include_related: bool,
    ) -> str:
        """Build the prompt for LLM feature selection."""
        categories_text = "\n".join(
            f"- {cat}: {count} properties" for cat, count in category_summary.items()
        )

        related_text = ""
        if include_related:
            related_text = """
Also consider categories that might be indirectly related. For example:
- Liver toxicity might benefit from: hepatotoxicity (direct), metabolic toxicity (liver metabolizes drugs),
  kinetics/ADME (hepatic clearance), cardiotoxicity (shared mechanisms), etc.
- Cardiac toxicity might benefit from: cardiotoxicity (direct), kinetics (drug distribution),
  ion channel effects, hERG inhibition, etc.
"""

        description_text = ""
        if target_description:
            description_text = f"\nAdditional context about the target:\n{target_description}\n"

        prompt = f"""You are an expert toxicologist helping to select relevant features for a machine learning model.

## Task
We are training a model to predict: **{target_property}**
{description_text}
We have access to ToxTransformer, which predicts 6,647 binary toxicity properties across various categories.
We want to select the most relevant property CATEGORIES to use as features.

## Available Categories
{categories_text}

## Instructions
Select up to {max_categories} categories that are most relevant for predicting {target_property}.

Consider:
1. Direct relevance: Categories that directly measure the target endpoint
2. Mechanistic relevance: Categories that measure related biological mechanisms
3. Predictive value: Categories that empirically tend to correlate with the target
{related_text}

## Response Format
Respond with a JSON object containing:
1. "selected_categories": List of category names (exactly as shown above)
2. "reasoning": Brief explanation of why each category was selected

Example:
```json
{{
  "selected_categories": ["hepatotoxicity", "metabolic toxicity", "kinetics (pharmacokinetics, toxicokinetics, adme, cmax, auc, etc)"],
  "reasoning": "Hepatotoxicity is directly relevant. Metabolic toxicity captures liver metabolism effects. Kinetics includes hepatic clearance markers."
}}
```

Provide your selection:"""

        return prompt

    def _parse_response(
        self, response: str, category_summary: Dict[str, int]
    ) -> Tuple[List[str], str]:
        """Parse LLM response to extract selected categories."""
        # Try to extract JSON from response
        import re

        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logging.warning("Could not parse LLM response as JSON")
                return list(category_summary.keys())[:5], response

        try:
            data = json.loads(json_str)
            selected = data.get("selected_categories", [])
            reasoning = data.get("reasoning", "")

            # Validate categories exist
            valid_categories = []
            available = {c.lower(): c for c in category_summary.keys()}
            for cat in selected:
                if cat.lower() in available:
                    valid_categories.append(available[cat.lower()])
                else:
                    logging.warning(f"Unknown category: {cat}")

            return valid_categories, reasoning

        except json.JSONDecodeError as e:
            logging.warning(f"JSON parse error: {e}")
            return list(category_summary.keys())[:5], response

    def select_by_category_names(
        self, categories: List[str], max_tokens_per_category: int = 100
    ) -> List[int]:
        """
        Select tokens by category names directly (no LLM call).

        Useful when you already know which categories are relevant.
        """
        selected_tokens = []
        for category in categories:
            tokens = self.metadata_loader.get_tokens_by_category(category)
            selected_tokens.extend(tokens[:max_tokens_per_category])

        # Remove duplicates
        seen = set()
        unique_tokens = []
        for t in selected_tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

        return unique_tokens


def get_semantic_feature_indices(
    target_property: str,
    target_description: Optional[str] = None,
    property_tokens: Optional[List[int]] = None,
    db_path: str = "brick/cvae.sqlite",
    model: str = "gemini-2.0-flash",
    max_categories: int = 10,
    cache_path: Optional[str] = None,
    use_vertex_ai: bool = False,
    vertex_project: Optional[str] = None,
    vertex_location: str = "us-central1",
) -> Tuple[np.ndarray, SemanticSelectionResult]:
    """
    Get feature indices for semantic selection.

    This is the main entry point for integrating semantic selection
    with the adapter training pipeline.

    Args:
        target_property: Name of the target property.
        target_description: Optional description.
        property_tokens: Ordered list of property tokens (from feature extractor).
        db_path: Path to ToxTransformer database.
        model: Gemini model to use.
        max_categories: Maximum categories to select.
        cache_path: Optional path to cache selection results.
        use_vertex_ai: Use Vertex AI instead of Google AI API.
        vertex_project: GCP project ID for Vertex AI.
        vertex_location: GCP region for Vertex AI.

    Returns:
        Tuple of (feature_indices array, SemanticSelectionResult).
    """
    # Check cache
    if cache_path:
        cache_file = Path(cache_path) / f"{target_property.replace(' ', '_')}_selection.json"
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
            result = SemanticSelectionResult(**cached)
            logging.info(f"Loaded cached selection: {len(result.selected_tokens)} tokens")

            if property_tokens:
                token_to_idx = {t: i for i, t in enumerate(property_tokens)}
                indices = [token_to_idx[t] for t in result.selected_tokens if t in token_to_idx]
                return np.array(indices), result
            else:
                return np.array(result.selected_tokens), result

    # Perform selection
    selector = SemanticFeatureSelector(
        model=model,
        db_path=db_path,
        use_vertex_ai=use_vertex_ai,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )
    result = selector.select_features(
        target_property=target_property,
        target_description=target_description,
        max_categories=max_categories,
    )

    # Cache result
    if cache_path:
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{target_property.replace(' ', '_')}_selection.json"
        with open(cache_file, "w") as f:
            json.dump(
                {
                    "selected_tokens": result.selected_tokens,
                    "selected_categories": result.selected_categories,
                    "reasoning": result.reasoning,
                    "n_selected": result.n_selected,
                    "n_total": result.n_total,
                },
                f,
                indent=2,
            )

    # Convert tokens to indices
    if property_tokens:
        token_to_idx = {t: i for i, t in enumerate(property_tokens)}
        indices = [token_to_idx[t] for t in result.selected_tokens if t in token_to_idx]
        return np.array(indices), result

    return np.array(result.selected_tokens), result
