"""Integration tests for search recall against the real dataset.

These tests use the real database and real embedding service (no LLM) to verify
that each search tool returns the correct images for various query types.
Requires: image_search.db populated with the 18-image dataset.
"""

from __future__ import annotations

import pytest

from image_search_app.tools.search_tools import (
    search_by_caption,
    search_by_location,
    search_by_person,
    search_by_time,
)
from image_search_app.agent.langgraph_flow import assemble_response
from image_search_app.vector.chroma_store import ChromaStore
from image_search_app.vector.embeddings import EmbeddingService

# --- Image ID constants (from the real dataset) ---

TOWER = "1bd19b78-0e6f-44fd-96e1-af29e8f98b6b"          # tower.jpg — Eiffel Tower, Paris
SYD = "a7c9382c-9dda-438b-977f-7c48955538d3"             # syd5.jpg — Sydney Opera House
PYRAMID = "841a7cf4-3715-4d6f-a55b-236354205cdb"         # b6.jpg — Sphinx & Pyramid, Giza
BUND = "8437ae23-ed7a-448f-9e73-d65b78d8c798"            # bund.jpg — Shanghai skyline night
TIANANMEN = "78ea9454-3e4e-4c0e-9c7c-b35d91acd990"       # tiananmen — Tiananmen Gate
BEACH_PEOPLE = "b9efd1d0-42fc-42ce-9e76-c56df4b3e2cc"    # beach_people.jpg — 4 people running
BEACH_GIRL = "73a7f1be-3ba8-4108-99fc-05a04aae01f5"      # beach_girl.jpg — Jane, Parga
BEACH_GIRL2 = "781a1512-e09e-452f-b7e6-48954e776786"     # beach_girl2.jpg — Amy, Ornos
COLIN = "5ae9475d-663f-4fb4-a48d-a2c091682f5d"           # Colin_Powell_0043.jpg
MICHAEL = "18e071af-166e-4cd1-bdb6-4c28d2d0cc9a"         # Michael_Powell_0001.jpg
IMG85 = "fa06bf50-f89d-40c8-8290-a005b189f6d3"           # IMG_0085.jpeg — Hong & Xia Kaixuan
FOREIGN = "f29026c7-a19a-4b8d-bc2f-440cad7283c7"         # FOREIGN*.jpg — Chen Li, LA
COASTAL = "09c90a42-9f0b-4016-9874-70dfb7caee55"         # IMG_0183.jpeg — coastal view
FISH = "f320a502-d08d-4aa2-bf07-84c73b99799e"            # IMG_0197.jpeg — fish dish
NORIO = "371a6b3b-8b3d-4b29-9647-51fa29309e4b"           # Norio_Ohga_0001.jpg
NORM = "6f610d57-01d0-405a-8a9d-3c7856ebb62f"            # Norm_Coleman_0002.jpg
OLESYA = "9cbc42b5-4669-4b2d-ba2f-5ea6cdf49ea1"          # Olesya_Bonabarenko_0002.jpg
OXANA = "fbb1a744-02b1-4183-af96-01ef7298910d"           # Oxana_Fedorova_0002.jpg


def _ids(results: list[dict]) -> set[str]:
    """Extract image IDs from tool results, filtering out hint dicts."""
    return {r["image_id"] for r in results if "image_id" in r}


# --- Fixtures ---

@pytest.fixture(scope="module")
def store():
    return ChromaStore()


@pytest.fixture(scope="module")
def embeddings():
    svc = EmbeddingService()
    # EmbeddingService lazy-loads on first use; just return it
    return svc


# =============================================================================
# 1. Person search recall
# =============================================================================


class TestPersonSearch:
    def test_full_name(self):
        results = search_by_person("Colin Powell")
        ids = _ids(results)
        assert COLIN in ids

    def test_partial_last_name_matches_multiple(self):
        """'Powell' should match both Colin Powell and Michael Powell."""
        results = search_by_person("Powell")
        ids = _ids(results)
        assert COLIN in ids
        assert MICHAEL in ids

    def test_partial_first_name(self):
        """'Colin' should match Colin Powell but NOT Norm Coleman."""
        results = search_by_person("Colin")
        ids = _ids(results)
        assert COLIN in ids
        assert NORM not in ids

    def test_single_common_name(self):
        results = search_by_person("tom")
        ids = _ids(results)
        assert BEACH_PEOPLE in ids

    def test_partial_name_multi_match(self):
        """'liu' matches both 'liu gang' (bund) and 'liu jingjing' (tower)."""
        results = search_by_person("liu")
        ids = _ids(results)
        assert BUND in ids
        assert TOWER in ids

    def test_exact_single_person(self):
        results = search_by_person("jane")
        ids = _ids(results)
        assert BEACH_GIRL in ids
        assert len(ids) == 1

    def test_case_insensitive(self):
        """Search is case-insensitive."""
        results = search_by_person("COLIN POWELL")
        ids = _ids(results)
        assert COLIN in ids

    def test_nonexistent_person(self):
        results = search_by_person("nonexistent_person_xyz")
        assert _ids(results) == set()

    def test_empty_name(self):
        results = search_by_person("")
        assert _ids(results) == set()

    def test_multi_word_name(self):
        """Multi-word names like 'xia kaixuan' should work."""
        results = search_by_person("xia kaixuan")
        ids = _ids(results)
        assert IMG85 in ids


# =============================================================================
# 2. Caption / scene search recall
# =============================================================================


class TestCaptionSearch:
    def test_beach_scene(self, store, embeddings):
        """'beach' should find all beach images."""
        results = search_by_caption("beach", store=store, embeddings=embeddings)
        ids = _ids(results)
        assert BEACH_PEOPLE in ids, f"beach_people missing from: {ids}"
        # At least one of the beach girl images should match
        assert ids & {BEACH_GIRL, BEACH_GIRL2}, f"No beach girl images in: {ids}"

    def test_eiffel_tower_landmark(self, store, embeddings):
        results = search_by_caption("Eiffel Tower", store=store, embeddings=embeddings)
        ids = _ids(results)
        assert TOWER in ids, f"tower.jpg missing from Eiffel Tower search: {ids}"

    def test_pyramid_landmark(self, store, embeddings):
        results = search_by_caption("pyramid", store=store, embeddings=embeddings)
        ids = _ids(results)
        assert PYRAMID in ids, f"b6.jpg missing from pyramid search: {ids}"

    def test_opera_house_landmark(self, store, embeddings):
        results = search_by_caption("Opera House", store=store, embeddings=embeddings)
        ids = _ids(results)
        assert SYD in ids, f"syd5.jpg missing from Opera House search: {ids}"

    def test_suit_and_tie(self, store, embeddings):
        """Formal attire query should find suited people."""
        results = search_by_caption("suit and tie", store=store, embeddings=embeddings)
        ids = _ids(results)
        # At least one of the suited people should be found
        assert ids & {COLIN, MICHAEL, NORM, NORIO}, (
            f"No suited people found for 'suit and tie': {ids}"
        )

    def test_food_query(self, store, embeddings):
        results = search_by_caption("fish dish on a plate", store=store, embeddings=embeddings)
        ids = _ids(results)
        assert FISH in ids, f"fish dish image missing: {ids}"

    def test_coastal_scenery(self, store, embeddings):
        results = search_by_caption("coastal view with cliffs and boat", store=store, embeddings=embeddings)
        ids = _ids(results)
        assert COASTAL in ids, f"coastal image missing: {ids}"

    def test_night_skyline(self, store, embeddings):
        results = search_by_caption("city skyline at night", store=store, embeddings=embeddings)
        ids = _ids(results)
        assert BUND in ids, f"bund.jpg missing from skyline search: {ids}"

    def test_sunglasses_hat(self, store, embeddings):
        """Accessory-based query."""
        results = search_by_caption("straw hat and sunglasses", store=store, embeddings=embeddings)
        ids = _ids(results)
        assert ids & {BEACH_GIRL, BEACH_GIRL2}, (
            f"Beach girl images with hat/sunglasses missing: {ids}"
        )

    def test_people_running(self, store, embeddings):
        results = search_by_caption("people running", store=store, embeddings=embeddings)
        ids = _ids(results)
        assert BEACH_PEOPLE in ids, f"beach_people missing from running search: {ids}"


# =============================================================================
# 3. Location search recall
# =============================================================================


class TestLocationSearch:
    def test_france(self):
        ids = _ids(search_by_location("France"))
        assert TOWER in ids

    def test_paris(self):
        ids = _ids(search_by_location("Paris"))
        assert TOWER in ids

    def test_australia(self):
        ids = _ids(search_by_location("Australia"))
        assert SYD in ids

    def test_sydney(self):
        ids = _ids(search_by_location("Sydney"))
        assert SYD in ids

    def test_egypt(self):
        ids = _ids(search_by_location("Egypt"))
        assert PYRAMID in ids

    def test_giza(self):
        ids = _ids(search_by_location("Giza"))
        assert PYRAMID in ids

    def test_shanghai(self):
        """Shanghai matches bund.jpg (Pudong) and IMG_0085 (Shanghai city)."""
        ids = _ids(search_by_location("Shanghai"))
        assert BUND in ids
        assert IMG85 in ids

    def test_greece(self):
        ids = _ids(search_by_location("Greece"))
        assert BEACH_GIRL in ids
        assert BEACH_GIRL2 in ids

    def test_florida(self):
        ids = _ids(search_by_location("Florida"))
        assert BEACH_PEOPLE in ids

    def test_miami(self):
        ids = _ids(search_by_location("Miami"))
        assert BEACH_PEOPLE in ids

    def test_california(self):
        ids = _ids(search_by_location("California"))
        assert FOREIGN in ids

    def test_china_multiple(self):
        """China should match multiple images."""
        ids = _ids(search_by_location("China"))
        assert TIANANMEN in ids
        assert BUND in ids
        assert COASTAL in ids
        assert FISH in ids
        assert IMG85 in ids

    def test_new_york(self):
        """'New York' matches both city and state for Michael Powell."""
        ids = _ids(search_by_location("New York"))
        assert MICHAEL in ids

    def test_united_states(self):
        """'United States' should match all US images."""
        ids = _ids(search_by_location("United States"))
        assert COLIN in ids
        assert MICHAEL in ids
        assert BEACH_PEOPLE in ids
        assert FOREIGN in ids

    def test_scene_word_returns_hint(self):
        """Scene descriptions like 'library' should return a hint, not geo results."""
        results = search_by_location("library")
        image_results = [r for r in results if "image_id" in r]
        hints = [r for r in results if "hint" in r]
        assert len(image_results) == 0
        assert len(hints) == 1
        assert "search_by_caption" in hints[0]["hint"]

    def test_empty_location(self):
        results = search_by_location("")
        assert results == []


# =============================================================================
# 4. Time search recall
# =============================================================================


class TestTimeSearch:
    def test_year_2022(self):
        """2022 has 4 images: tower, syd5, b6, Michael_Powell."""
        ids = _ids(search_by_time("2022"))
        assert TOWER in ids
        assert SYD in ids
        assert PYRAMID in ids
        assert MICHAEL in ids
        assert len(ids) == 4

    def test_year_2020(self):
        """2020 has 3 images: beach_people, beach_girl2, bund."""
        ids = _ids(search_by_time("2020"))
        assert BEACH_PEOPLE in ids
        assert BEACH_GIRL2 in ids
        assert BUND in ids
        assert len(ids) == 3

    def test_year_2025(self):
        """2025 has 2 images: IMG_0183, IMG_0197."""
        ids = _ids(search_by_time("2025"))
        assert COASTAL in ids
        assert FISH in ids
        assert len(ids) == 2

    def test_year_2016(self):
        """2016 has 2 images: Norio_Ohga, Oxana_Fedorova."""
        ids = _ids(search_by_time("2016"))
        assert NORIO in ids
        assert OXANA in ids
        assert len(ids) == 2

    def test_year_2024(self):
        """2024 has 2 images: Colin_Powell, IMG_0085."""
        ids = _ids(search_by_time("2024"))
        assert COLIN in ids
        assert IMG85 in ids
        assert len(ids) == 2

    def test_nonexistent_year(self):
        """A year with no images should return empty."""
        ids = _ids(search_by_time("2010"))
        assert len(ids) == 0


# =============================================================================
# 5. Assembly tests (multi-tool queries)
# =============================================================================


class TestAssembly:
    def test_person_plus_location_solid(self):
        """liu jingjing + France → tower.jpg should be solid."""
        tool_results = {
            "search_by_person": search_by_person("liu jingjing"),
            "search_by_location": search_by_location("France"),
        }
        response = assemble_response(tool_results)
        solid_ids = {str(r.image_id) for r in response.solid_results}
        assert TOWER in solid_ids

    def test_person_plus_caption(self, store, embeddings):
        """tom + beach → beach_people is solid; other beaches are soft."""
        tool_results = {
            "search_by_person": search_by_person("tom"),
            "search_by_caption": search_by_caption(
                "beach", store=store, embeddings=embeddings,
            ),
        }
        response = assemble_response(tool_results)
        solid_ids = {str(r.image_id) for r in response.solid_results}
        soft_ids = {str(r.image_id) for r in response.soft_results}

        assert BEACH_PEOPLE in solid_ids, f"beach_people should be solid: {solid_ids}"
        # Other beach images should be soft (matched caption but not person)
        beach_in_soft = soft_ids & {BEACH_GIRL, BEACH_GIRL2}
        assert len(beach_in_soft) > 0, f"Other beach images should be soft: {soft_ids}"

    def test_location_plus_caption(self, store, embeddings):
        """Greece + beach → Greek beach images should be solid."""
        tool_results = {
            "search_by_location": search_by_location("Greece"),
            "search_by_caption": search_by_caption(
                "beach", store=store, embeddings=embeddings,
            ),
        }
        response = assemble_response(tool_results)
        solid_ids = {str(r.image_id) for r in response.solid_results}
        # At least one Greek beach image should be solid
        assert solid_ids & {BEACH_GIRL, BEACH_GIRL2}, (
            f"Greek beach images should be solid: {solid_ids}"
        )

    def test_person_plus_time(self):
        """Colin Powell + 2024 → Colin should be solid."""
        tool_results = {
            "search_by_person": search_by_person("Colin Powell"),
            "search_by_time": search_by_time("2024"),
        }
        response = assemble_response(tool_results)
        solid_ids = {str(r.image_id) for r in response.solid_results}
        assert COLIN in solid_ids

    def test_three_tools_narrow(self, store, embeddings):
        """lisa + Egypt + pyramid → b6.jpg should be the only solid result."""
        tool_results = {
            "search_by_person": search_by_person("lisa"),
            "search_by_location": search_by_location("Egypt"),
            "search_by_caption": search_by_caption(
                "pyramid", store=store, embeddings=embeddings,
            ),
        }
        response = assemble_response(tool_results)
        solid_ids = {str(r.image_id) for r in response.solid_results}
        assert PYRAMID in solid_ids
        assert len(solid_ids) == 1

    def test_no_overlap_all_soft(self):
        """When tools have no overlapping results, all go to soft."""
        tool_results = {
            "search_by_person": search_by_person("jane"),       # beach_girl (Greece)
            "search_by_location": search_by_location("Egypt"),  # b6 (Giza)
        }
        response = assemble_response(tool_results)
        solid_ids = {str(r.image_id) for r in response.solid_results}
        soft_ids = {str(r.image_id) for r in response.soft_results}
        assert len(solid_ids) == 0
        assert BEACH_GIRL in soft_ids
        assert PYRAMID in soft_ids

    def test_solid_results_have_explanations(self, store, embeddings):
        """Solid results should have 'Matched all criteria' explanation."""
        tool_results = {
            "search_by_person": search_by_person("tian mei"),
            "search_by_location": search_by_location("Sydney"),
        }
        response = assemble_response(tool_results)
        assert len(response.solid_results) > 0
        reason = response.solid_results[0].explanation.reason
        assert "Matched all criteria" in reason
