"""Unit tests for MarkdownFilter normalization layer (Step 1)."""

import pytest
from src.rag.chunking import MarkdownFilter


@pytest.fixture
def filter_layer() -> MarkdownFilter:
    return MarkdownFilter()


class TestMarkdownFilter:
    def test_preserves_headings_and_hierarchy(self, filter_layer: MarkdownFilter):
        text = "# Health Benefits\n\n## Eligibility\n\n### Dependents"
        cleaned = filter_layer.normalize(text)
        assert "# Health Benefits" in cleaned
        assert "## Eligibility" in cleaned
        assert "### Dependents" in cleaned

    def test_preserves_bullet_and_numbered_lists(self, filter_layer: MarkdownFilter):
        text = (
            "- Medical care\n"
            "- Dental care\n"
            "- Vision care\n\n"
            "1. Goodwill purpose\n"
            "2. Customary\n"
            "3. Not cash"
        )
        cleaned = filter_layer.normalize(text)
        assert "- Medical care" in cleaned
        assert "- Dental care" in cleaned
        assert "1. Goodwill purpose" in cleaned
        assert "3. Not cash" in cleaned

    def test_preserves_urls_emails_dates_and_currency(self, filter_layer: MarkdownFilter):
        text = (
            "Contact [People Team](https://sourcegraph.slack.com) at "
            "people-ops@sourcegraph.com by 2024-01-01 for $500 stipend (100% coverage)."
        )
        cleaned = filter_layer.normalize(text)
        assert "[People Team](https://sourcegraph.slack.com)" in cleaned
        assert "people-ops@sourcegraph.com" in cleaned
        assert "2024-01-01" in cleaned
        assert "$500" in cleaned
        assert "100%" in cleaned

    def test_normalizes_excessive_blank_lines(self, filter_layer: MarkdownFilter):
        text = "Heading\n\n\n\n\nParagraph with lots of blanks.\n\n\n\nEnd."
        cleaned = filter_layer.normalize(text)
        assert "\n\n\n" not in cleaned
        assert "Heading\n\nParagraph with lots of blanks.\n\nEnd." == cleaned

    def test_strips_html_spans_while_preserving_text(self, filter_layer: MarkdownFilter):
        text = "Benefits effective from <span style=\"text-decoration:underline;\">2023-11-20</span> to 2024-01-01."
        cleaned = filter_layer.normalize(text)
        assert "Benefits effective from 2023-11-20 to 2024-01-01." == cleaned

    def test_strips_unprintable_control_characters(self, filter_layer: MarkdownFilter):
        text = "Policy\x00 rules\x07 for\x1f teammates."
        cleaned = filter_layer.normalize(text)
        assert cleaned == "Policy rules for teammates."

    def test_preserves_faq_structure_and_policy_terminology(self, filter_layer: MarkdownFilter):
        text = (
            "**What:** Open Enrollment is your once-a-year opportunity.\n\n"
            "**When:** Took place from 2023-11-20 to 2023-12-01.\n\n"
            "**How:** Make changes in your Rippling account."
        )
        cleaned = filter_layer.normalize(text)
        assert "**What:**" in cleaned
        assert "**When:**" in cleaned
        assert "**How:**" in cleaned
        assert "Rippling" in cleaned
        assert "Open Enrollment" in cleaned
