import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class LegalDocument:
    act: str
    year: int
    section: str
    chapter: str
    content: str
    source: str
    doc_type: str  # "statute" or "user_doc"
    doc_hash: str = field(init=False)
    stable_id: str = field(init=False)

    def __post_init__(self):
        # Calculate SHA256 Hash for deduplication and unique identification
        hasher = hashlib.sha256()
        hasher.update(self.content.encode('utf-8'))
        self.doc_hash = hasher.hexdigest()
        
        # Format stable ID
        act_prefix = "".join([word[0] for word in self.act.split()]).upper()
        clean_section = self.section.replace(" ", "")
        
        # FIX: If section is "General" or blank, append the hash slice to guarantee uniqueness!
        if not clean_section or clean_section.lower() == "general":
            self.stable_id = f"{act_prefix}-GEN-{self.doc_hash[:8]}"
        else:
            self.stable_id = f"{act_prefix}-{clean_section}"

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "act": self.act,
            "year": self.year,
            "section": self.section,
            "chapter": self.chapter,
            "source": self.source,
            "doc_type": self.doc_type,
            "doc_hash": self.doc_hash,
            "stable_id": self.stable_id,
            "language": "en"
        }