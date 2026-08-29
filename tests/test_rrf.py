import pytest

from app.services.rag.rrf import RankedItem, reciprocal_rank_fusion


def test_rrf_rewards_items_present_in_both_lists() -> None:
    dense = [RankedItem("shared", 1, {"name": "shared"}), RankedItem("dense", 2, {})]
    sparse = [RankedItem("shared", 2, {"name": "shared"}), RankedItem("sparse", 1, {})]

    result = reciprocal_rank_fusion([dense, sparse], k=60)

    assert result[0][0] == "shared"
    assert result[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_is_deterministic_for_ties() -> None:
    items = [RankedItem("b", 1, {}), RankedItem("a", 1, {})]
    result = reciprocal_rank_fusion([items], k=60)
    assert [item[0] for item in result] == ["a", "b"]


def test_rrf_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="ranks"):
        reciprocal_rank_fusion([[RankedItem("bad", 0, {})]])
