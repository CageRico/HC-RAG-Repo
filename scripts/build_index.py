"""
Build the three-level hierarchical index from financial documents
"""

import os
import sys
import json
import yaml
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchical_index import (
    HierarchicalIndex, DocumentNode, SectionNode,
    TextChunkNode, TableCellNode, TOCParser
)
from src.utils import chunk_text, extract_tables_from_html, extract_sections_from_10k

try:
    from src.encoders import (
        TextEncoder,
        TableEncoder,
        RetrievalEncoder,
        load_alignment_checkpoint,
    )
    _ENCODERS_AVAILABLE = True
except Exception as e:
    print(f"DEBUG encoders import error: {e}")
    _ENCODERS_AVAILABLE = False

# Sector mapping (same as build_multidoc2025.py)
SP500_COMPANIES = {
    "Information Technology": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","AMD","INTC"],
    "Financials":             ["JPM","BAC","WFC","GS","MS","BLK","AXP","C"],
    "Healthcare":             ["JNJ","UNH","LLY","PFE","ABBV","MRK","TMO","ABT"],
    "Consumer Discretionary": ["AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TGT"],
    "Consumer Staples":       ["WMT","PG","KO","PEP","COST","PM","MO","CL"],
    "Industrials":            ["GE","HON","UPS","CAT","DE","LMT","RTX","BA"],
    "Communication Services": ["META","GOOGL","NFLX","DIS","CMCSA","T","VZ","CHTR"],
    "Energy":                 ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO"],
    "Materials":              ["LIN","APD","ECL","DD","NEM","FCX","ALB","IFF"],
    "Real Estate":            ["AMT","PLD","CCI","EQIX","PSA","DLR","O","WELL"],
    "Utilities":              ["NEE","DUK","SO","D","AEP","EXC","XEL","ED"],
}
TICKER_TO_SECTOR = {t: s for s, tickers in SP500_COMPANIES.items() for t in tickers}


class IndexBuilder:
    """Build hierarchical index from financial documents (10-K filings)"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.index = HierarchicalIndex(self.config["index"])
        self.chunk_size = self.config["index"]["chunk_size"]
        self.chunk_overlap = self.config["index"]["chunk_overlap"]
        
        # Initialize encoders for embedding (optional, can be done later)
        self.encoder = None
        self._init_encoder()
        
    def _init_encoder(self):
        if not _ENCODERS_AVAILABLE:
            print("Warning: encoders not available (transformers import error), skipping embedding generation")
            self.encoder = None
            return
        text_model = self.config["models"].get("text_encoder", "")
        if not text_model:
            print("No text_encoder configured, skipping embedding generation")
            self.encoder = None
            return
        try:
            local_files_only = self.config["models"].get("local_files_only", True)
            text_encoder = TextEncoder(
                model_name=self.config["models"]["text_encoder"],
                embedding_dim=self.config["alignment"]["embedding_dim"],
                local_files_only=local_files_only,
            )
            table_encoder = TableEncoder(
                model_name=self.config["models"]["table_encoder"],
                embedding_dim=self.config["alignment"]["embedding_dim"],
                local_files_only=local_files_only,
            )
            align_ckpt = os.path.join(
                self.config["paths"]["checkpoint_dir"],
                "align_checkpoint_best.pt",
            )
            if load_alignment_checkpoint(text_encoder, table_encoder, align_ckpt):
                print(f"Loaded alignment checkpoint from {align_ckpt}")
            else:
                print("Alignment checkpoint not found; using base encoder projections.")
            self.encoder = RetrievalEncoder(text_encoder, table_encoder)
            print("Encoder initialized for embedding generation")
        except Exception as e:
            print(f"Warning: Could not initialize encoder: {e}")
            self.encoder = None
    
    def add_document(self, doc_id: str, company_name: str, fiscal_year: str,
                     industry: str, content_html: str, toc_data: List[Dict] = None):
        """
        Add a complete financial document to the index
        
        Args:
            doc_id: unique document identifier
            company_name: company name
            fiscal_year: fiscal year
            industry: industry classification
            content_html: HTML content of the 10-K filing
            toc_data: optional pre-parsed TOC
        """
        # L1: Document node
        doc_node = DocumentNode(
            doc_id=doc_id,
            company_name=company_name,
            fiscal_year=fiscal_year,
            industry=industry,
            report_type="10-K"
        )
        self.index.add_document(doc_node)

        # Parse TOC if not provided
        if toc_data is None:
            toc_data = TOCParser.parse_html_toc(content_html)

        # Extract sections based on TOC
        sections = extract_sections_from_10k(content_html, toc_data)

        total_chunks = 0
        total_tables = 0

        for section_info in sections:
            section_id = f"{doc_id}_{section_info['item_num']}"
            section_node = SectionNode(
                section_id=section_id,
                title=section_info['title'],
                level=section_info.get('level', 1),
                start_pos=section_info.get('start_pos', 0),
                end_pos=section_info.get('end_pos', 0)
            )
            self.index.add_section(section_node, doc_id)
            
            # Process section content: split into text chunks (cap at 20 per section)
            text_content = section_info.get('text', '')[:self.chunk_size * 20]
            chunks = chunk_text(text_content, self.chunk_size, self.chunk_overlap)[:20]

            # Build chunk nodes first, then batch-encode
            chunk_nodes = []
            chunk_texts = []
            for i, chunk_str in enumerate(chunks):
                chunk_id = f"{section_id}_chunk_{i}"
                chunk_node = TextChunkNode(
                    chunk_id=chunk_id,
                    text=chunk_str,
                    start_char=i * (self.chunk_size - self.chunk_overlap),
                    end_char=i * (self.chunk_size - self.chunk_overlap) + len(chunk_str)
                )
                chunk_nodes.append(chunk_node)
                chunk_texts.append(chunk_str)

            if self.encoder and chunk_texts:
                embeddings = self.encoder.encode_text_chunks_batch(chunk_texts)
                for node, emb in zip(chunk_nodes, embeddings):
                    node.embedding = emb

            for chunk_node in chunk_nodes:
                self.index.add_text_chunk(chunk_node, section_id)
            total_chunks += len(chunk_nodes)

            # Process tables in this section
            # Prefer pre-parsed tables list; fall back to HTML parsing
            tables = section_info.get('tables') or extract_tables_from_html(section_info.get('html', ''))
            total_tables += len(tables)
            for table_idx, table in enumerate(tables):
                headers = table.get('header', [])
                rows = table.get('rows', [])
                cell_nodes = []
                cell_texts = []
                for row_idx, row in enumerate(rows):
                    for col_idx, cell_value in enumerate(row):
                        col_header = headers[col_idx] if col_idx < len(headers) else f"Col{col_idx}"
                        row_header = row[0] if row and len(row) > 0 else f"Row{row_idx}"
                        cell_id = f"{section_id}_table_{table_idx}_r{row_idx}_c{col_idx}"
                        cell_node = TableCellNode(
                            cell_id=cell_id,
                            row_header=str(row_header),
                            col_header=str(col_header),
                            value=str(cell_value),
                            table_id=f"table_{table_idx}",
                            row_idx=row_idx,
                            col_idx=col_idx
                        )
                        cell_nodes.append(cell_node)
                        cell_texts.append(f"{row_header} {col_header} {cell_value}")

                if self.encoder and cell_texts:
                    embeddings = self.encoder.encode_table_cells_batch(cell_texts)
                    for node, emb in zip(cell_nodes, embeddings):
                        node.embedding = emb

                for cell_node in cell_nodes:
                    self.index.add_table_cell(cell_node, section_id)
        
        print(f"Added document {doc_id}: {total_chunks} text chunks, "
              f"{total_tables} tables processed")
    
    def add_cross_doc_relations(self, industry_group: str, doc_ids: List[str]):
        """Add same-industry cross-document edges"""
        for i, doc_id in enumerate(doc_ids):
            for j, other_id in enumerate(doc_ids):
                if i != j:
                    self.index.add_cross_doc_edge(doc_id, other_id, relation="same_industry")
    
    def save(self, output_path: str):
        """Save index to disk"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.index.save(output_path)
        print(f"Index saved to {output_path}")
    
    def load(self, input_path: str):
        """Load existing index"""
        self.index.load(input_path)
        print(f"Index loaded from {input_path}")


def build_from_directory(data_dir: str, config_path: str = "config.yaml"):
    """Build index from a directory of 10-K filings, one file at a time."""
    import gc
    import time
    builder = IndexBuilder(config_path)

    html_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.html'))
    all_doc_ids = []
    doc_times = []
    total_start = time.perf_counter()

    for html_file in tqdm(html_files, desc="Building index"):
        parts   = html_file.replace('.html', '').split('_')
        company = parts[0] if parts else "Unknown"
        year    = parts[1] if len(parts) > 1 else "2024"
        doc_id  = f"{company}_{year}"
        sector  = TICKER_TO_SECTOR.get(company.upper(), "Unknown")

        try:
            with open(os.path.join(data_dir, html_file), 'r',
                      encoding='utf-8', errors='replace') as f:
                content = f.read(8 * 1024 * 1024)

            doc_start = time.perf_counter()
            builder.add_document(
                doc_id=doc_id,
                company_name=company,
                fiscal_year=year,
                industry=sector,
                content_html=content,
            )
            doc_elapsed = time.perf_counter() - doc_start
            doc_times.append(doc_elapsed)
            all_doc_ids.append(doc_id)
        except Exception as e:
            import traceback
            print(f"  [WARN] Skipping {html_file}: {e}")
            traceback.print_exc()
        finally:
            del content
            gc.collect()

    # Cross-doc edges within same sector
    from collections import defaultdict
    sector_docs = defaultdict(list)
    for doc_id in all_doc_ids:
        ticker = doc_id.split('_')[0].upper()
        s = TICKER_TO_SECTOR.get(ticker, "Unknown")
        sector_docs[s].append(doc_id)
    for sector, doc_ids in sector_docs.items():
        builder.add_cross_doc_relations(sector, doc_ids)

    os.makedirs("./indexes", exist_ok=True)
    builder.save("./indexes/hierarchical_index.pkl")
    all_doc_ids = [f.replace('.html', '') for f in html_files]
    builder.add_cross_doc_relations("all", all_doc_ids)

    output_path = "./indexes/hierarchical_index.pkl"
    builder.save(output_path)

    # Report timing (Table 3: index construction time)
    total_elapsed = time.perf_counter() - total_start
    n_docs = len(doc_times)
    if n_docs > 0:
        avg_min = (sum(doc_times) / n_docs) / 60
        print(f"\nIndex construction timing:")
        print(f"  Documents processed : {n_docs}")
        print(f"  Total time          : {total_elapsed/60:.1f} min")
        print(f"  Avg time per doc    : {avg_min:.2f} min/doc")
        # Save timing to outputs for Table 3
        os.makedirs("./outputs", exist_ok=True)
        import json
        with open("./outputs/index_timing.json", "w") as f:
            json.dump({
                "n_docs": n_docs,
                "total_minutes": round(total_elapsed / 60, 2),
                "avg_min_per_doc": round(avg_min, 2),
            }, f, indent=2)
        print(f"  Timing saved to ./outputs/index_timing.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HC-RAG hierarchical index")
    parser.add_argument("--data_dir", type=str, default="./data/raw",
                        help="Directory containing HTML filings")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config file path")
    parser.add_argument("--output", type=str, default="./indexes/hierarchical_index.pkl",
                        help="Output index path")
    args = parser.parse_args()
    
    builder = IndexBuilder(args.config)
    
    # Example: add a single document manually
    # builder.add_document(...)
    
    # Or build from directory
    if os.path.isdir(args.data_dir):
        build_from_directory(args.data_dir, args.config)
    else:
        print(f"Data directory {args.data_dir} not found. Please prepare HTML filings.")
