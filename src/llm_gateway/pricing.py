"""Model prices and request cost.

Prices are USD per million tokens, with cached input tokens priced separately (prompt caching
discounts). They come from config here; day 14 loads the versioned `model_prices` table from
Postgres into the same object. Unknown models (e.g. local Ollama models) cost 0.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cached_input_per_mtok: float | None = None  # defaults to the input price


class Pricing:
    def __init__(self, prices: dict[str, Price] | None = None):
        self.prices = dict(prices or {})

    @classmethod
    def from_config(cls, cfg: dict | None) -> "Pricing":
        return cls({model: Price(**{k: float(v) for k, v in p.items()}) for model, p in (cfg or {}).items()})

    def cost(self, model: str, usage: dict | None) -> float:
        price = self.prices.get(model)
        if price is None or not usage:
            return 0.0
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        cached_price = (
            price.cached_input_per_mtok if price.cached_input_per_mtok is not None else price.input_per_mtok
        )
        return (
            (prompt - cached) * price.input_per_mtok
            + cached * cached_price
            + completion * price.output_per_mtok
        ) / 1e6
