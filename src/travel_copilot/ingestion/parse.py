import bz2
import xml.etree.ElementTree as ET
import mwparserfromhell
import re
from pathlib import Path
from typing import Optional, Generator
from .. import config

def parse_articles(dump_path: str, limit: Optional[int] = None) -> Generator[dict, None, None]:
    """
    Generator that yields cleaned Wikivoyage articles from a .bz2 XML dump.
    """
    # Namespaces to skip (typical MediaWiki namespaces)
    SKIP_PREFIXES = (
        "Category:", "Template:", "Wikivoyage:", "User:", "File:", 
        "MediaWiki:", "Help:", "Talk:", "Portal:", "Module:", "Special:"
    )

    count = 0
    
    # Use bz2.open to read the compressed file directly
    with bz2.open(dump_path, "rb") as f:
        # iterparse allows streaming the XML without loading it all into memory.
        # We listen for "start" to capture the root and "end" to process full tags.
        context = ET.iterparse(f, events=("start", "end"))
        
        # We need to handle the XML namespace. MediaWiki usually uses a URI.
        # We can detect it from the first element.
        namespace = ""
        root = None
        
        for event, elem in context:
            # The first "start" event will be the root element (<mediawiki>)
            if root is None and event == "start":
                root = elem
                continue

            # We only process full articles on the "end" event
            if event == "end":
                # Extract namespace from the tag name (e.g., {http://...}page)
                if not namespace and "}" in elem.tag:
                    namespace = elem.tag.split("}")[0] + "}"
                
                tag_name = elem.tag.replace(namespace, "")
                
                if tag_name == "page":
                    title_elem = elem.find(f"{namespace}title")
                    revision_elem = elem.find(f"{namespace}revision")
                    
                    if title_elem is not None and revision_elem is not None:
                        title = title_elem.text or ""
                        
                        # Skip non-article pages based on prefix
                        if any(title.startswith(prefix) for prefix in SKIP_PREFIXES):
                            elem.clear()
                            # Clear root to release the shell
                            root.clear()
                            continue
                            
                        text_elem = revision_elem.find(f"{namespace}text")
                        if text_elem is not None:
                            raw_text = text_elem.text or ""
                            
                            # Skip redirects
                            if raw_text.strip().lower().startswith("#redirect"):
                                elem.clear()
                                root.clear()
                                continue
                            
                            # Clean wikitext to plain text
                            wikicode = mwparserfromhell.parse(raw_text)
                            clean_text = wikicode.strip_code().strip()
                            
                            # Remove residual image directives (thumb|, 200px|, etc.)
                            clean_text = re.sub(
                                r'(thumb|left|right|center|frameless|\d+px)\|', 
                                '', 
                                clean_text, 
                                flags=re.IGNORECASE
                            )
                            
                            # Skip stubs / very short articles
                            if len(clean_text) > 200:
                                yield {
                                    "title": title,
                                    "text": clean_text
                                }
                                count += 1
                    
                    # Crucial for memory management: 
                    # 1. Clear the page element itself
                    elem.clear()
                    # 2. Clear the root to release the accumulation of empty page shells
                    root.clear()
                    
                    if limit and count >= limit:
                        break

if __name__ == "__main__":
    # Test the parser
    
    # Construct the path to the dump file
    dump_file = config.DATA_PATH / "enwikivoyage-latest-pages-articles.xml.bz2"
    
    if not dump_file.exists():
        print(f"Error: Dump file not found at {dump_file}")
    else:
        print(f"Starting test parse of: {dump_file.name}")
        print("-" * 30)
        
        for i, article in enumerate(parse_articles(str(dump_file), limit=3)):
            print(f"Article {i+1}: {article['title']}")
            print(f"Snippet: {article['text'][:300]}...")
            print("-" * 30)
