"""
Utility functions for text processing, table extraction, etc.
"""

import re
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    """
    if not text:
        return []
    # Hard cap to avoid MemoryError on huge sections
    text = text[:chunk_size * 20]
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text) and len(chunks) < 20:
        end = min(start + chunk_size, len(text))
        if end < len(text):
            last_punct = max(
                text.rfind('.', max(start, end-100), end),
                text.rfind('?', max(start, end-100), end),
                text.rfind('!', max(start, end-100), end),
                text.rfind('\n', max(start, end-100), end)
            )
            if last_punct > start:
                end = last_punct + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def extract_tables_from_html(html_content: str) -> List[Dict[str, Any]]:
    """
    Extract tables from HTML content
    
    Returns:
        List of dicts with keys: 'header', 'rows'
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = []
    
    for table in soup.find_all('table'):
        header = []
        rows = []
        
        # Extract headers
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                header = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        
        # If no thead, try first row as header
        if not header:
            first_row = table.find('tr')
            if first_row:
                header = [td.get_text(strip=True) for td in first_row.find_all(['th', 'td'])]
                # Remove first row from data rows
                rows_start = 1
            else:
                rows_start = 0
        else:
            rows_start = 0
        
        # Extract data rows
        tbody = table.find('tbody')
        if tbody:
            row_elements = tbody.find_all('tr')
        else:
            row_elements = table.find_all('tr')[rows_start:]
        
        for row in row_elements:
            cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            if cells:
                rows.append(cells)
        
        if rows:
            tables.append({
                'header': header,
                'rows': rows
            })
    
    return tables


def extract_sections_from_10k(html_content: str, toc_data: List[Dict]) -> List[Dict]:
    """
    Extract 10-K sections from HTML.

    Uses DOM-order indexing throughout — no plain-text-to-DOM mapping needed.
    Each node gets a sequential DOM index; Item headings are located directly
    in the DOM; tables are assigned to sections by DOM index range.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    default_sections = [
        ("1",  "Business"),
        ("1A", "Risk Factors"),
        ("1B", "Unresolved Staff Comments"),
        ("1C", "Cybersecurity"),
        ("2",  "Properties"),
        ("3",  "Legal Proceedings"),
        ("4",  "Mine Safety Disclosures"),
        ("5",  "Market for Registrant"),
        ("6",  "Selected Financial Data"),
        ("7",  "Management"),
        ("7A", "Quantitative and Qualitative"),
        ("8",  "Financial Statements"),
        ("9",  "Changes in and Disagreements"),
        ("9A", "Controls and Procedures"),
        ("9B", "Other Information"),
        ("9C", "Disclosure Regarding Foreign Jurisdictions"),
        ("10", "Directors"),
        ("11", "Executive Compensation"),
        ("12", "Security Ownership"),
        ("13", "Certain Relationships"),
        ("14", "Principal Accountant"),
        ("15", "Exhibits"),
    ]
    known_items = {n for n, _ in default_sections}
    item_title = {n: h for n, h in default_sections}

    # --- 1. Assign sequential DOM index to every node ---
    all_nodes = list(soup.descendants)
    node_to_idx: Dict[int, int] = {id(n): i for i, n in enumerate(all_nodes)}
    total_nodes = len(all_nodes)

    # --- 2. Find Item heading positions directly in DOM ---
    item_pattern = re.compile(r'(?<!\w)Item\s*(\d+[A-Z]?)[\.\s]', re.IGNORECASE)

    # item_num -> list of (dom_idx, text_node)
    item_occurrences: Dict[str, List[tuple]] = {}
    for node in all_nodes:
        if not isinstance(node, str):
            continue
        text = node.replace("\xa0", " ").replace("\u2003", " ").replace("\u2002", " ")
        for m in item_pattern.finditer(text):
            item_num = m.group(1).upper()
            if item_num not in known_items:
                continue
            dom_idx = node_to_idx[id(node)]
            item_occurrences.setdefault(item_num, []).append((dom_idx, node))

    if not item_occurrences:
        return []

    # --- 3. For each item, pick the BODY occurrence (not TOC) ---
    # Key insight: in a 10-K, the TOC lists all Items in sequence with very
    # short gaps between them. The body has each Item followed by long content.
    # We use a two-pass approach:
    #   Pass 1: compute content length after each occurrence (capped for speed)
    #   Pass 2: pick the occurrence with MAX content (= body, not TOC)
    # Then verify section order makes sense (body Items appear in order 1..15)

    all_heading_positions = sorted(
        dom_idx for positions in item_occurrences.values() for dom_idx, _ in positions
    )

    def _content_length_after(dom_idx: int) -> int:
        next_headings = [p for p in all_heading_positions if p > dom_idx]
        end = min(next_headings[0] if next_headings else total_nodes, dom_idx + 8000)
        total = 0
        for node in all_nodes[dom_idx:end]:
            if isinstance(node, str):
                total += len(node.strip())
        return total

    # Pick occurrence with MAXIMUM content after it per item
    best_occurrence: Dict[str, int] = {}
    for item_num, occurrences in item_occurrences.items():
        best_idx = max(occurrences, key=lambda x: _content_length_after(x[0]))[0]
        best_occurrence[item_num] = best_idx

    # --- 4. Build section DOM ranges from best occurrences only ---
    # Sort by DOM position — this gives the correct document order
    sorted_sections = sorted(best_occurrence.items(), key=lambda x: x[1])
    section_dom_ranges = []  # (item_num, dom_start, dom_end)
    for i, (item_num, dom_start) in enumerate(sorted_sections):
        if i + 1 < len(sorted_sections):
            dom_end = sorted_sections[i + 1][1]
        else:
            dom_end = total_nodes
        section_dom_ranges.append((item_num, dom_start, dom_end))

    # --- 5. Extract plain text per section directly from DOM nodes ---
    section_text: Dict[str, str] = {}
    for item_num, dom_start, dom_end in section_dom_ranges:
        parts = []
        for node in all_nodes[dom_start:dom_end]:
            if isinstance(node, str):
                t = node.replace("\xa0", " ").replace("\u2003", " ").replace("\u2002", " ").strip()
                if t:
                    parts.append(t)
        section_text[item_num] = " ".join(parts)

    # --- 6. Parse all tables with their DOM index ---
    def _parse_table(table_tag) -> Dict:
        header = []
        rows = []
        thead = table_tag.find('thead')
        if thead:
            hr = thead.find('tr')
            if hr:
                header = [th.get_text(strip=True) for th in hr.find_all(['th', 'td'])]
        rows_start = 0
        if not header:
            first_row = table_tag.find('tr')
            if first_row:
                header = [td.get_text(strip=True) for td in first_row.find_all(['th', 'td'])]
                rows_start = 1
        tbody = table_tag.find('tbody')
        row_elements = tbody.find_all('tr') if tbody else table_tag.find_all('tr')[rows_start:]
        for row in row_elements:
            cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            if cells:
                rows.append(cells)
        return {'header': header, 'rows': rows}

    all_tables_with_dom = []
    for table_tag in soup.find_all('table'):
        tbl_dom = node_to_idx.get(id(table_tag), -1)
        tbl = _parse_table(table_tag)
        if tbl['rows']:
            all_tables_with_dom.append((tbl_dom, tbl))

    # --- 7. Assign tables to sections by DOM range ---
    section_tables: Dict[str, List] = {item_num: [] for item_num, _, _ in section_dom_ranges}
    for tbl_dom, tbl_dict in all_tables_with_dom:
        assigned = False
        for item_num, dom_start, dom_end in section_dom_ranges:
            if dom_start <= tbl_dom < dom_end:
                section_tables[item_num].append(tbl_dict)
                assigned = True
                break
        if not assigned:
            # Fallback: nearest section before this table
            best = None
            for item_num, dom_start, dom_end in section_dom_ranges:
                if dom_start <= tbl_dom:
                    best = item_num
            if best:
                section_tables[best].append(tbl_dict)

    # --- 8. Assemble final section list ---
    sections = []
    for item_num, dom_start, dom_end in section_dom_ranges:
        text = section_text.get(item_num, "")
        if len(text) < 80:
            continue
        tbls = section_tables.get(item_num, [])
        sections.append({
            "item_num":  item_num,
            "title":     f"Item {item_num}. {item_title.get(item_num, item_num)}",
            "text":      text,
            "html":      "",
            "tables":    tbls,
            "start_pos": dom_start,
            "end_pos":   dom_end,
            "level":     1,
        })

    sections.sort(key=lambda s: s["start_pos"])
    return sections

    default_sections = [
        ("1",  "Business"),
        ("1A", "Risk Factors"),
        ("1B", "Unresolved Staff Comments"),
        ("1C", "Cybersecurity"),
        ("2",  "Properties"),
        ("3",  "Legal Proceedings"),
        ("4",  "Mine Safety Disclosures"),
        ("5",  "Market for Registrant"),
        ("6",  "Selected Financial Data"),
        ("7",  "Management"),
        ("7A", "Quantitative and Qualitative"),
        ("8",  "Financial Statements"),
        ("9",  "Changes in and Disagreements"),
        ("9A", "Controls and Procedures"),
        ("9B", "Other Information"),
        ("9C", "Disclosure Regarding Foreign Jurisdictions"),
        ("10", "Directors"),
        ("11", "Executive Compensation"),
        ("12", "Security Ownership"),
        ("13", "Certain Relationships"),
        ("14", "Principal Accountant"),
        ("15", "Exhibits"),
    ]
    known_items = {n for n, _ in default_sections}

    any_item = re.compile(r'(?<!\w)Item\s*(\d+[A-Z]?)[\.\s]', re.IGNORECASE)

    # Collect ALL positions for each item_num in plain text
    all_positions: Dict[str, List[int]] = {}
    for m in any_item.finditer(plain):
        item_num = m.group(1).upper()
        if item_num not in known_items:
            continue
        all_positions.setdefault(item_num, []).append(m.start())

    if not all_positions:
        return []

    all_starts = sorted(
        pos for positions in all_positions.values() for pos in positions)

    def _text_after(start: int) -> str:
        next_starts = [s for s in all_starts if s > start]
        end = next_starts[0] if next_starts else len(plain)
        return plain[start:end]

    # --- 2. Assign a global DOM index to every node for position tracking ---
    # Walk all tags in document order and record each table's index
    all_tags = list(soup.descendants)
    tag_index: Dict[int, int] = {}  # id(tag) -> dom_order
    dom_order = 0
    for node in all_tags:
        tag_index[id(node)] = dom_order
        dom_order += 1

    # --- 3. Locate Item headings in DOM order ---
    # For each Item heading found in plain text, find the corresponding DOM node
    # Strategy: walk all text nodes, build a mapping from plain-text char offset
    # to DOM order by accumulating text lengths
    # Build char_offset -> dom_order map via text node walk
    char_to_dom: List[int] = []  # index = plain char pos, value = dom_order
    plain_rebuilt = []
    for node in soup.descendants:
        if isinstance(node, str):
            node_dom = tag_index.get(id(node), 0)
            normalized = node.replace("\xa0", " ").replace("\u2003", " ").replace("\u2002", " ")
            normalized = re.sub(r'[ \t]+', ' ', normalized)
            plain_rebuilt.append(normalized)
            char_to_dom.extend([node_dom] * len(normalized))

    # char_to_dom may differ slightly from plain due to separator=" " in get_text
    # Use a simpler approach: for each Item match position in plain,
    # find the nearest dom_order by scanning char_to_dom
    def _plain_pos_to_dom(pos: int) -> int:
        if pos < len(char_to_dom):
            return char_to_dom[pos]
        return dom_order  # end of document

    # --- 4. Extract ALL tables with their DOM order ---
    def _parse_table(table_tag) -> Dict:
        header = []
        rows = []
        thead = table_tag.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                header = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        rows_start = 0
        if not header:
            first_row = table_tag.find('tr')
            if first_row:
                header = [td.get_text(strip=True) for td in first_row.find_all(['th', 'td'])]
                rows_start = 1
        tbody = table_tag.find('tbody')
        row_elements = tbody.find_all('tr') if tbody else table_tag.find_all('tr')[rows_start:]
        for row in row_elements:
            cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            if cells:
                rows.append(cells)
        return {'header': header, 'rows': rows}

    all_tables_with_dom = []  # (dom_order, table_dict)
    for table_tag in soup.find_all('table'):
        tbl_dom = tag_index.get(id(table_tag), -1)
        tbl = _parse_table(table_tag)
        if tbl['rows']:
            all_tables_with_dom.append((tbl_dom, tbl))

    # --- 5. Build sections with DOM boundaries ---
    plain_len = len(plain)
    section_ranges = []  # (item_num, best_pos, end_pos, text, dom_start, dom_end)
    for item_num, positions in all_positions.items():
        best_pos = max(positions, key=lambda p: len(_text_after(p)))
        text = re.sub(r'\s+', ' ', _text_after(best_pos)).strip()
        if len(text) < 80:
            continue
        next_starts = [s for s in all_starts if s > best_pos]
        end_pos = next_starts[0] if next_starts else plain_len
        dom_start = _plain_pos_to_dom(best_pos)
        dom_end = _plain_pos_to_dom(end_pos)
        section_ranges.append((item_num, best_pos, end_pos, text, dom_start, dom_end))

    # --- 6. Assign tables to sections by DOM order ---
    section_tables: Dict[str, List] = {r[0]: [] for r in section_ranges}
    # Sort sections by dom_start for efficient assignment
    section_ranges_sorted = sorted(section_ranges, key=lambda r: r[4])
    for tbl_dom, tbl_dict in all_tables_with_dom:
        # Find the section whose dom range contains this table
        assigned = False
        for item_num, _, _, _, dom_start, dom_end in section_ranges_sorted:
            if dom_start <= tbl_dom < dom_end:
                section_tables[item_num].append(tbl_dict)
                assigned = True
                break
        if not assigned:
            # Fallback: assign to the section with the closest dom_start before this table
            best = None
            for item_num, _, _, _, dom_start, dom_end in section_ranges_sorted:
                if dom_start <= tbl_dom:
                    best = item_num
            if best:
                section_tables[best].append(tbl_dict)

    # --- 7. Assemble final section list ---
    sections = []
    for item_num, best_pos, end_pos, text, dom_start, dom_end in section_ranges:
        title_hint = next((h for n, h in default_sections if n == item_num), item_num)
        tbls = section_tables.get(item_num, [])
        # Serialize assigned tables back to minimal HTML for compatibility
        html_parts = []
        for t in tbls:
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                for r in t['rows']
            )
            html_parts.append(f"<table>{rows_html}</table>")
        sections.append({
            "item_num":  item_num,
            "title":     f"Item {item_num}. {title_hint}",
            "text":      text,
            "html":      "".join(html_parts),
            "tables":    tbls,   # direct list, avoids re-parsing HTML
            "start_pos": best_pos,
            "end_pos":   end_pos,
            "level":     1,
        })

    sections.sort(key=lambda s: s["start_pos"])
    return sections


def normalize_number(value: str) -> float:
    """
    Normalize financial number strings to float
    Examples: "$1.2M" -> 1200000, "15%" -> 0.15, "(2.3)" -> -2.3
    """
    if not value:
        return 0.0
    
    # Remove commas and currency symbols
    cleaned = re.sub(r'[^\d\-\.\(\)\%BbMmKk]', '', value)
    
    # Handle parentheses for negative numbers
    if '(' in cleaned and ')' in cleaned:
        cleaned = '-' + cleaned.replace('(', '').replace(')', '')
    
    # Handle multipliers
    multiplier = 1
    if 'B' in cleaned or 'b' in cleaned:
        multiplier = 1_000_000_000
        cleaned = re.sub(r'[Bb]', '', cleaned)
    elif 'M' in cleaned or 'm' in cleaned:
        multiplier = 1_000_000
        cleaned = re.sub(r'[Mm]', '', cleaned)
    elif 'K' in cleaned or 'k' in cleaned:
        multiplier = 1_000
        cleaned = re.sub(r'[Kk]', '', cleaned)
    
    # Handle percentages
    if '%' in cleaned:
        multiplier /= 100
        cleaned = cleaned.replace('%', '')
    
    try:
        num = float(cleaned)
        return num * multiplier
    except ValueError:
        return 0.0


def extract_numerical_entities(text: str) -> List[Tuple[str, float]]:
    """
    Extract numerical entities from text with their normalized values
    """
    patterns = [
        (r'\$?\d+(?:,\d{3})*(?:\.\d+)?\s*[BbMmKk]?%?', 'currency'),
        (r'\d+(?:\.\d+)?%', 'percentage'),
        (r'\d+(?:,\d{3})*(?:\.\d+)?', 'number')
    ]
    
    results = []
    for pattern, _ in patterns:
        for match in re.finditer(pattern, text):
            value_str = match.group(0)
            normalized = normalize_number(value_str)
            results.append((value_str, normalized))
    
    return results


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Simple ROUGE-L implementation for validation
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    # LCS length
    dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i, p in enumerate(pred_tokens, 1):
        for j, r in enumerate(ref_tokens, 1):
            if p == r:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs = dp[len(pred_tokens)][len(ref_tokens)]
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_company_metadata(csv_path: str) -> Dict[str, Dict]:
    """
    Load company metadata from CSV (ticker, CIK, industry, sector)
    """
    import csv
    metadata = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get('ticker', '').upper()
            if ticker:
                metadata[ticker] = {
                    'cik': row.get('cik', ''),
                    'industry': row.get('industry', ''),
                    'sector': row.get('sector', ''),
                    'fiscal_year_end': row.get('fiscal_year_end', '')
                }
    return metadata