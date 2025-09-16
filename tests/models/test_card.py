"""
Unit tests for the Card model.

Tests cover:
- Card creation from Scryfall data
- Pydantic validation
- Field handling (required vs optional)
- Edge cases and error conditions
"""
import pytest
from pydantic import ValidationError
from src.mtg_search_agent.models.card import Card


class TestCardModel:
    """Test suite for the Card model."""

    def test_card_from_scryfall_complete_data(self, sample_card_data):
        """Test creating a Card from complete Scryfall data."""
        card = Card.from_scryfall(sample_card_data)

        assert card.id == sample_card_data["id"]
        assert card.name == sample_card_data["name"]
        assert card.mana_cost == sample_card_data["mana_cost"]
        assert card.cmc == sample_card_data["cmc"]
        assert card.type_line == sample_card_data["type_line"]
        assert card.oracle_text == sample_card_data["oracle_text"]
        assert card.colors == sample_card_data["colors"]
        assert card.color_identity == sample_card_data["color_identity"]
        assert card.keywords == sample_card_data["keywords"]
        assert card.legalities == sample_card_data["legalities"]
        assert card.set == sample_card_data["set"]
        assert card.set_name == sample_card_data["set_name"]
        assert card.rarity == sample_card_data["rarity"]
        assert card.collector_number == sample_card_data["collector_number"]
        assert card.artist == sample_card_data["artist"]
        assert card.scryfall_uri == sample_card_data["scryfall_uri"]
        assert card.image_uris == sample_card_data["image_uris"]
        assert card.prices == sample_card_data["prices"]

    def test_card_from_scryfall_minimal_data(self):
        """Test creating a Card with only required fields."""
        minimal_data = {
            "id": "test-id-123",
            "name": "Test Card",
            "type_line": "Instant",
            "set": "tst",
            "set_name": "Test Set",
            "rarity": "common",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/card/tst/1/test-card"
        }

        card = Card.from_scryfall(minimal_data)

        # Required fields
        assert card.id == "test-id-123"
        assert card.name == "Test Card"
        assert card.type_line == "Instant"
        assert card.set == "tst"
        assert card.set_name == "Test Set"
        assert card.rarity == "common"
        assert card.collector_number == "1"
        assert card.scryfall_uri == "https://scryfall.com/card/tst/1/test-card"

        # Optional fields should have default values
        assert card.mana_cost is None
        assert card.cmc is None
        assert card.oracle_text is None
        assert card.power is None
        assert card.toughness is None
        assert card.colors == []
        assert card.color_identity == []
        assert card.keywords == []
        assert card.legalities == {}
        assert card.artist is None
        assert card.image_uris is None
        assert card.prices is None

    def test_card_with_creature_data(self, sample_creature_data):
        """Test creating a creature Card with power/toughness."""
        card = Card.from_scryfall(sample_creature_data)

        assert card.power == "4"
        assert card.toughness == "4"
        assert "Creature" in card.type_line
        assert "Flying" in card.keywords
        assert "Vigilance" in card.keywords

    def test_card_direct_instantiation(self, sample_card_data):
        """Test creating Card directly (not from Scryfall)."""
        card = Card(
            id=sample_card_data["id"],
            name=sample_card_data["name"],
            type_line=sample_card_data["type_line"],
            set=sample_card_data["set"],
            set_name=sample_card_data["set_name"],
            rarity=sample_card_data["rarity"],
            collector_number=sample_card_data["collector_number"],
            scryfall_uri=sample_card_data["scryfall_uri"]
        )

        assert card.name == sample_card_data["name"]
        assert card.type_line == sample_card_data["type_line"]

    def test_card_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        incomplete_data = {
            "name": "Test Card",
            "type_line": "Instant"
            # Missing id, set, set_name, rarity, collector_number, scryfall_uri
        }

        with pytest.raises(ValidationError) as exc_info:
            Card(**incomplete_data)

        error = exc_info.value
        assert "id" in str(error)

    def test_card_invalid_data_types(self):
        """Test validation with incorrect data types."""
        with pytest.raises(ValidationError):
            Card(
                id=123,  # Should be string
                name="Test Card",
                type_line="Instant",
                set="tst",
                set_name="Test Set",
                rarity="common",
                collector_number="1",
                scryfall_uri="https://scryfall.com/test"
            )

    def test_card_empty_lists_and_dicts(self):
        """Test Card handles empty lists and dictionaries properly."""
        data = {
            "id": "test-id",
            "name": "Test Card",
            "type_line": "Instant",
            "colors": [],
            "color_identity": [],
            "keywords": [],
            "legalities": {},
            "set": "tst",
            "set_name": "Test Set",
            "rarity": "common",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/test"
        }

        card = Card.from_scryfall(data)
        assert card.colors == []
        assert card.color_identity == []
        assert card.keywords == []
        assert card.legalities == {}

    def test_card_none_values_in_optional_fields(self):
        """Test Card handles None values in optional fields."""
        data = {
            "id": "test-id",
            "name": "Test Card",
            "type_line": "Instant",
            "mana_cost": None,
            "cmc": None,
            "oracle_text": None,
            "power": None,
            "toughness": None,
            "artist": None,
            "image_uris": None,
            "prices": None,
            "set": "tst",
            "set_name": "Test Set",
            "rarity": "common",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/test"
        }

        card = Card.from_scryfall(data)
        assert card.mana_cost is None
        assert card.cmc is None
        assert card.oracle_text is None
        assert card.power is None
        assert card.toughness is None
        assert card.artist is None
        assert card.image_uris is None
        assert card.prices is None

    def test_card_with_complex_mana_cost(self):
        """Test Card with complex mana cost."""
        data = {
            "id": "complex-mana-id",
            "name": "Complex Mana Card",
            "mana_cost": "{2}{W}{U}{B}{R}{G}",
            "cmc": 7.0,
            "type_line": "Legendary Creature — Elder Dragon",
            "colors": ["W", "U", "B", "R", "G"],
            "color_identity": ["W", "U", "B", "R", "G"],
            "set": "cmn",
            "set_name": "Complex Mana",
            "rarity": "mythic",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/complex"
        }

        card = Card.from_scryfall(data)
        assert card.mana_cost == "{2}{W}{U}{B}{R}{G}"
        assert card.cmc == 7.0
        assert len(card.colors) == 5
        assert len(card.color_identity) == 5

    def test_card_with_special_characters(self):
        """Test Card with special characters in text fields."""
        data = {
            "id": "special-char-id",
            "name": "Æther Spëllbomb",
            "oracle_text": "{1}, Sacrifice Æther Spëllbomb: Return target creature to its owner's hand.",
            "type_line": "Artifact",
            "artist": "Artist with ñames",
            "set": "spc",
            "set_name": "Special Characters",
            "rarity": "common",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/special"
        }

        card = Card.from_scryfall(data)
        assert card.name == "Æther Spëllbomb"
        assert "Æther" in card.oracle_text
        assert card.artist == "Artist with ñames"

    def test_card_legalities_structure(self):
        """Test Card legalities field structure."""
        data = {
            "id": "legal-id",
            "name": "Legal Test Card",
            "type_line": "Instant",
            "legalities": {
                "standard": "legal",
                "pioneer": "legal",
                "modern": "legal",
                "legacy": "legal",
                "vintage": "legal",
                "commander": "legal",
                "brawl": "not_legal",
                "historic": "banned"
            },
            "set": "leg",
            "set_name": "Legalities",
            "rarity": "common",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/legal"
        }

        card = Card.from_scryfall(data)
        assert card.legalities["standard"] == "legal"
        assert card.legalities["historic"] == "banned"
        assert card.legalities["brawl"] == "not_legal"

    def test_card_prices_structure(self):
        """Test Card prices field structure with various currencies."""
        data = {
            "id": "price-id",
            "name": "Expensive Card",
            "type_line": "Artifact",
            "prices": {
                "usd": "199.99",
                "usd_foil": "299.99",
                "usd_etched": None,
                "eur": "179.50",
                "eur_foil": "249.75",
                "tix": "25.50"
            },
            "set": "exp",
            "set_name": "Expensive",
            "rarity": "mythic",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/expensive"
        }

        card = Card.from_scryfall(data)
        assert card.prices["usd"] == "199.99"
        assert card.prices["usd_foil"] == "299.99"
        assert card.prices["usd_etched"] is None
        assert card.prices["eur"] == "179.50"
        assert card.prices["tix"] == "25.50"

    def test_card_image_uris_structure(self):
        """Test Card image_uris field structure."""
        data = {
            "id": "image-id",
            "name": "Image Test Card",
            "type_line": "Sorcery",
            "image_uris": {
                "small": "https://cards.scryfall.io/small/front/1/2/123.jpg",
                "normal": "https://cards.scryfall.io/normal/front/1/2/123.jpg",
                "large": "https://cards.scryfall.io/large/front/1/2/123.jpg",
                "png": "https://cards.scryfall.io/png/front/1/2/123.png",
                "art_crop": "https://cards.scryfall.io/art_crop/front/1/2/123.jpg",
                "border_crop": "https://cards.scryfall.io/border_crop/front/1/2/123.jpg"
            },
            "set": "img",
            "set_name": "Images",
            "rarity": "common",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/images"
        }

        card = Card.from_scryfall(data)
        assert "small" in card.image_uris
        assert "normal" in card.image_uris
        assert "large" in card.image_uris
        assert card.image_uris["small"].endswith(".jpg")

    def test_card_zero_cmc(self):
        """Test Card with zero converted mana cost."""
        data = {
            "id": "zero-cmc-id",
            "name": "Free Spell",
            "mana_cost": "",
            "cmc": 0.0,
            "type_line": "Instant",
            "set": "free",
            "set_name": "Free Spells",
            "rarity": "common",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/free"
        }

        card = Card.from_scryfall(data)
        assert card.mana_cost == ""
        assert card.cmc == 0.0

    def test_card_fractional_cmc(self):
        """Test Card with fractional converted mana cost (for special cards)."""
        data = {
            "id": "fractional-id",
            "name": "Half Mana Card",
            "cmc": 0.5,  # Some split cards have fractional CMC
            "type_line": "Instant",
            "set": "frac",
            "set_name": "Fractional",
            "rarity": "common",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/fractional"
        }

        card = Card.from_scryfall(data)
        assert card.cmc == 0.5

    def test_card_equality_and_hash(self, sample_card_data):
        """Test Card equality comparison and hashing."""
        card1 = Card.from_scryfall(sample_card_data)
        card2 = Card.from_scryfall(sample_card_data)

        # Should be equal based on same data
        assert card1 == card2

        # Should be hashable (for use in sets/dicts)
        card_set = {card1, card2}
        assert len(card_set) == 1  # Should deduplicate

    def test_card_string_representation(self, sample_card_data):
        """Test Card string representation."""
        card = Card.from_scryfall(sample_card_data)
        card_str = str(card)

        # Should contain the card name
        assert card.name in card_str