# chunk.py
# ---------------------------------------------------------------------------
# PURPOSE: Take a parsed Wikivoyage article (clean text + title) and split it
# into smaller, retrieval-ready "chunks". Each chunk is a self-contained piece
# of text small enough to embed precisely, with metadata (title, section)
# riding along so we can cite sources and trace where answers came from.
#
# STRATEGY: "Section-aware with a recursive fallback".
#   1. Split the article along Wikivoyage's own section headings
#      (Understand, See, Eat, ...). These are natural topic boundaries.
#   2. Within each section, if it's too long, hand it to LangChain's
#      RecursiveCharacterTextSplitter, which cleanly splits by size with
#      overlap, preferring natural boundaries (paragraph > line > sentence > word).
#   3. If an article has NO recognizable headings, fall back to treating the
#      whole article as one block and letting the splitter handle it.
# ---------------------------------------------------------------------------

import re
from typing import List, Dict, Generator, Any

# LangChain's splitter handles the fiddly "split by size with clean overlap"
# mechanics for us. We keep the section logic ourselves (it's tailored to our
# data); we delegate the within-section character splitting to this library.
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parse import parse_articles
from .. import config


# The standard set of Wikivoyage section headings. After markup is stripped,
# these appear as standalone lines in the article text. We use this list to
# detect where one section ends and the next begins.
WIKIVOYAGE_SECTIONS = [
    "Understand", "Get in", "Get around", "See", "Do", "Buy",
    "Eat", "Drink", "Sleep", "Connect", "Stay safe",
    "Stay healthy", "Go next", "Talk", "Cope", "Learn", "Work"
]


def chunk_article(
    article: Dict[str, str],
    max_chars: int = 1000,
    overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Split ONE article into retrieval-ready chunks using a section-aware strategy.

    Args:
        article:   a dict like {"title": ..., "text": ...} from parse_articles.
        max_chars: target maximum size of each chunk, in characters.
        overlap:   how many characters of the previous chunk to repeat at the
                   start of the next, so context spanning a boundary isn't lost.

    Returns:
        A list of chunk dicts, each: {"text", "title", "section", "chunk_id"}.
    """
    # Pull out the article's title and text. .get(...) with a default avoids
    # a crash if a key is somehow missing.
    title = article.get("title", "Unknown")
    text = article.get("text", "")

    # --- Configure the within-section splitter ------------------------------
    # We build the splitter here (using this call's max_chars/overlap) so those
    # values stay configurable per call. The `separators` list tells it the
    # PREFERRED places to cut, in order: try paragraph breaks first, then single
    # line breaks, then sentence ends, then spaces, and only as a last resort
    # ("") cut mid-word. This is what avoids ugly mid-word splits.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # --- Step 1: Detect sections -------------------------------------------
    # Build a regex that matches a line that is EXACTLY one of the section
    # names. ^...$ means "the whole line is this and nothing else", so a line
    # that merely mentions "See" inside a sentence won't be mistaken for a heading.
    section_pattern = r'^(' + '|'.join(WIKIVOYAGE_SECTIONS) + r')$'

    # Break the article into individual lines so we can scan for headings.
    lines = text.splitlines()

    # We'll accumulate a list of (section_name, section_text) pairs.
    sections = []

    # Text that appears BEFORE the first heading is the article's intro.
    current_section_name = "Introduction"
    current_section_content = []  # lines collected for the current section

    # Track whether we ever saw a real heading — drives the fallback below.
    found_any_section = False

    for line in lines:
        clean_line = line.strip()

        # Does this line, on its own, match a section heading?
        # re.IGNORECASE so "see" / "See" / "SEE" all match.
        if re.match(section_pattern, clean_line, re.IGNORECASE):
            # We hit a NEW heading. First, save the section we were building
            # (only if it actually has content).
            if current_section_content:
                sections.append(
                    (current_section_name, "\n".join(current_section_content).strip())
                )

            # Start a fresh section under this heading.
            current_section_name = clean_line
            current_section_content = []
            found_any_section = True
        else:
            # Not a heading — it's body text, so add it to the current section.
            current_section_content.append(line)

    # After the loop, don't forget the LAST section still being collected.
    if current_section_content:
        sections.append(
            (current_section_name, "\n".join(current_section_content).strip())
        )

    # --- Step 2: Fallback when there are no headings -----------------------
    # Some articles (short stubs, oddly formatted ones) have no recognizable
    # sections. In that case, treat the entire article as one "Introduction"
    # block; the splitter below will still chunk it sensibly by size.
    if not found_any_section:
        sections = [("Introduction", text.strip())]

    # --- Step 3: Turn sections into chunks ---------------------------------
    chunks = []
    chunk_index = 0  # running counter, used to give each chunk a unique id

    for section_name, section_text in sections:
        # Skip empty sections (e.g. a heading with nothing under it).
        if not section_text:
            continue

        # Hand the section text to the splitter. If the section is already
        # under max_chars, this returns it as a single piece; if it's longer,
        # it returns multiple overlapping pieces.
        text_pieces = splitter.split_text(section_text)

        for piece in text_pieces:
            # Guard against blank/whitespace-only pieces.
            if not piece.strip():
                continue

            # Build the chunk record. The metadata is the important part:
            #  - title/section let us cite the source and understand context
            #  - chunk_id is a unique, human-readable handle ("Aachen::0")
            #    that we'll also use as the record's id in the vector DB.
            chunks.append({
                "text": piece,
                "title": title,
                "section": section_name,
                "chunk_id": f"{title}::{chunk_index}"
            })
            chunk_index += 1

    return chunks


def chunk_articles(
    articles: Generator[Dict[str, str], None, None],
    **kwargs
) -> Generator[Dict[str, Any], None, None]:
    """
    Take a STREAM of articles (from parse_articles) and yield chunks from all
    of them, one at a time. This stays a generator so we never hold every chunk
    in memory at once — same streaming principle as the parser.

    **kwargs lets callers pass max_chars/overlap straight through to chunk_article.
    """
    for article in articles:
        for chunk in chunk_article(article, **kwargs):
            yield chunk


if __name__ == "__main__":
    # Quick test: parse 3 articles, chunk them, and inspect the result.
    # Run with: poetry run python -m src.travel_copilot.ingestion.chunk
    dump_file = config.DATA_PATH / "enwikivoyage-latest-pages-articles.xml.bz2"

    print(f"Starting chunking test from: {dump_file.name}")

    # 1. Parse a few articles (generator, limit=3 for fast feedback).
    articles = parse_articles(str(dump_file), limit=3)

    # 2. Chunk them. We wrap in list() here only so we can count and slice for
    #    the printout; in the real pipeline we'd keep it as a lazy generator.
    all_chunks = list(chunk_articles(articles, max_chars=1000, overlap=150))

    print("-" * 30)
    print(f"Total articles parsed: 3")
    print(f"Total chunks created: {len(all_chunks)}")
    print("-" * 30)

    # 3. Inspect the first 5 chunks — check section labels look right and the
    #    text reads as coherent passages.
    for i, chunk in enumerate(all_chunks[:5]):
        print(f"Chunk {i+1} | Title: {chunk['title']} | Section: {chunk['section']}")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Snippet: {chunk['text'][:150]}...")
        print("-" * 30)
