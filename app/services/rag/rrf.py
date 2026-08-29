from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RankedItem:
    item_id: str
    rank: int
    payload: dict


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RankedItem]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[tuple[str, float, dict]]:
    if k < 1:
        raise ValueError("k must be positive")

    scores: defaultdict[str, float] = defaultdict(float)
    payloads: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for item in ranked_list:
            if item.rank < 1:
                raise ValueError("ranks must start at 1")
            scores[item.item_id] += 1 / (k + item.rank)
            payloads.setdefault(item.item_id, item.payload)

    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if limit is not None:
        fused = fused[:limit]
    return [(item_id, score, payloads[item_id]) for item_id, score in fused]
