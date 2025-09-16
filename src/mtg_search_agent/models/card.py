from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class Card(BaseModel):
    """Represents a Magic: The Gathering card from Scryfall API"""
    id: str
    name: str
    mana_cost: Optional[str] = None
    cmc: Optional[float] = None
    type_line: str
    oracle_text: Optional[str] = None
    power: Optional[str] = None
    toughness: Optional[str] = None
    colors: List[str] = []
    color_identity: List[str] = []
    keywords: List[str] = []
    legalities: Dict[str, str] = {}
    set: str
    set_name: str
    rarity: str
    collector_number: str
    artist: Optional[str] = None
    scryfall_uri: str
    image_uris: Optional[Dict[str, str]] = None
    prices: Optional[Dict[str, Optional[str]]] = None

    def __hash__(self) -> int:
        """Make Card hashable using its unique id"""
        return hash(self.id)

    def __eq__(self, other) -> bool:
        """Override equality to use id for comparison"""
        if not isinstance(other, Card):
            return False
        return self.id == other.id
    
    @classmethod
    def from_scryfall(cls, data: Dict[str, Any]) -> "Card":
        """Create Card from Scryfall API response"""
        return cls(
            id=data["id"],
            name=data["name"],
            mana_cost=data.get("mana_cost"),
            cmc=data.get("cmc"),
            type_line=data["type_line"],
            oracle_text=data.get("oracle_text"),
            power=data.get("power"),
            toughness=data.get("toughness"),
            colors=data.get("colors", []),
            color_identity=data.get("color_identity", []),
            keywords=data.get("keywords", []),
            legalities=data.get("legalities", {}),
            set=data["set"],
            set_name=data["set_name"],
            rarity=data["rarity"],
            collector_number=data["collector_number"],
            artist=data.get("artist"),
            scryfall_uri=data["scryfall_uri"],
            image_uris=data.get("image_uris"),
            prices=data.get("prices")
        )