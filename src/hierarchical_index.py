"""
Three-Level Hierarchical Heterogeneous Graph Index
Document Nodes (L1) -> Section Nodes (L2) -> Semantic Unit Nodes (L3)
"""

import json
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict


class NodeType(Enum):
    DOCUMENT = "document"
    SECTION = "section"
    TEXT_CHUNK = "text_chunk"
    TABLE_CELL = "table_cell"


class EdgeType(Enum):
    HIERARCHICAL = "hierarchical"  # parent-child relation
    SEQUENTIAL = "sequential"      # next-prev section
    CROSS_DOC = "cross_doc"        # same industry, same period, etc.
    CROSS_REF = "cross_ref"        # internal references


@dataclass
class BaseNode:
    node_id: str
    node_type: NodeType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class DocumentNode(BaseNode):
    """L1: Document-level node"""
    def __init__(self, doc_id: str, company_name: str, fiscal_year: str, 
                 industry: str, report_type: str = "10-K"):
        super().__init__(
            node_id=doc_id,
            node_type=NodeType.DOCUMENT,
            content=None,
            metadata={
                "company_name": company_name,
                "fiscal_year": fiscal_year,
                "industry": industry,
                "report_type": report_type
            }
        )


@dataclass
class SectionNode(BaseNode):
    """L2: Section-level node"""
    def __init__(self, section_id: str, title: str, level: int, 
                 start_pos: int, end_pos: int):
        super().__init__(
            node_id=section_id,
            node_type=NodeType.SECTION,
            content=None,
            metadata={
                "title": title,
                "level": level,
                "start_pos": start_pos,
                "end_pos": end_pos
            }
        )


@dataclass
class TextChunkNode(BaseNode):
    """L3: Text chunk node"""
    def __init__(self, chunk_id: str, text: str, start_char: int, end_char: int):
        super().__init__(
            node_id=chunk_id,
            node_type=NodeType.TEXT_CHUNK,
            content=text,
            metadata={"start_char": start_char, "end_char": end_char}
        )


@dataclass
class TableCellNode(BaseNode):
    """L3: Table cell node"""
    def __init__(self, cell_id: str, row_header: str, col_header: str, 
                 value: str, table_id: str, row_idx: int, col_idx: int):
        # Store structured cell information
        cell_content = {
            "row_header": row_header,
            "col_header": col_header,
            "value": value,
            "table_id": table_id,
            "position": (row_idx, col_idx)
        }
        super().__init__(
            node_id=cell_id,
            node_type=NodeType.TABLE_CELL,
            content=cell_content,
            metadata={
                "row_header": row_header,
                "col_header": col_header,
                "value": value,
                "table_id": table_id
            }
        )


class HierarchicalIndex:
    """
    Three-level hierarchical heterogeneous graph index
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: Dict[str, List[Tuple[str, EdgeType]]] = defaultdict(list)
        self.reverse_edges: Dict[str, List[Tuple[str, EdgeType]]] = defaultdict(list)
        
        # Level-specific mappings
        self.doc_nodes: Dict[str, DocumentNode] = {}
        self.section_nodes: Dict[str, SectionNode] = {}
        self.text_chunk_nodes: Dict[str, TextChunkNode] = {}
        self.table_cell_nodes: Dict[str, TableCellNode] = {}
        
        # Document to sections mapping
        self.doc_to_sections: Dict[str, List[str]] = defaultdict(list)
        # Section to chunks mapping
        self.section_to_chunks: Dict[str, List[str]] = defaultdict(list)
        # Document cross-doc relations
        self.cross_doc_edges: Dict[str, List[str]] = defaultdict(list)
        # Sequential section relations within a document
        self.section_next: Dict[str, str] = {}
        self.section_prev: Dict[str, str] = {}
        
    def add_document(self, doc_node: DocumentNode) -> str:
        """Add L1 Document Node"""
        self.nodes[doc_node.node_id] = doc_node
        self.doc_nodes[doc_node.node_id] = doc_node
        return doc_node.node_id
    
    def add_section(self, section_node: SectionNode, parent_doc_id: str) -> str:
        """Add L2 Section Node with hierarchical edge to parent document"""
        self.nodes[section_node.node_id] = section_node
        self.section_nodes[section_node.node_id] = section_node
        
        # Add hierarchical edge
        self._add_edge(parent_doc_id, section_node.node_id, EdgeType.HIERARCHICAL)
        self.doc_to_sections[parent_doc_id].append(section_node.node_id)
        
        return section_node.node_id
    
    def add_text_chunk(self, chunk_node: TextChunkNode, parent_section_id: str) -> str:
        """Add L3 Text Chunk Node"""
        self.nodes[chunk_node.node_id] = chunk_node
        self.text_chunk_nodes[chunk_node.node_id] = chunk_node
        
        self._add_edge(parent_section_id, chunk_node.node_id, EdgeType.HIERARCHICAL)
        self.section_to_chunks[parent_section_id].append(chunk_node.node_id)
        
        return chunk_node.node_id
    
    def add_table_cell(self, cell_node: TableCellNode, parent_section_id: str) -> str:
        """Add L3 Table Cell Node"""
        self.nodes[cell_node.node_id] = cell_node
        self.table_cell_nodes[cell_node.node_id] = cell_node
        
        self._add_edge(parent_section_id, cell_node.node_id, EdgeType.HIERARCHICAL)
        self.section_to_chunks[parent_section_id].append(cell_node.node_id)
        
        return cell_node.node_id
    
    def add_cross_doc_edge(self, doc_id_1: str, doc_id_2: str, relation: str = "same_industry"):
        """Add cross-document relation edge (L1 level)"""
        self._add_edge(doc_id_1, doc_id_2, EdgeType.CROSS_DOC, metadata={"relation": relation})
        self.cross_doc_edges[doc_id_1].append(doc_id_2)
    
    def add_sequential_edge(self, prev_section_id: str, next_section_id: str):
        """Add sequential edge between sections"""
        if prev_section_id == next_section_id:
            return
        if self.section_next.get(prev_section_id) == next_section_id:
            return
        self._add_edge(prev_section_id, next_section_id, EdgeType.SEQUENTIAL)
        self.section_next[prev_section_id] = next_section_id
        self.section_prev[next_section_id] = prev_section_id
    
    def _add_edge(self, from_id: str, to_id: str, edge_type: EdgeType, metadata: Dict = None):
        """Add edge between nodes"""
        self.edges[from_id].append((to_id, edge_type))
        self.reverse_edges[to_id].append((from_id, edge_type))
    
    def get_children(self, node_id: str, node_type: Optional[NodeType] = None) -> List[BaseNode]:
        """Get child nodes of a given node"""
        children = []
        for child_id, edge_type in self.edges.get(node_id, []):
            if edge_type == EdgeType.HIERARCHICAL:
                child_node = self.nodes.get(child_id)
                if child_node and (node_type is None or child_node.node_type == node_type):
                    children.append(child_node)
        return children
    
    def get_parent(self, node_id: str) -> Optional[BaseNode]:
        """Get parent node"""
        for parent_id, edge_type in self.reverse_edges.get(node_id, []):
            if edge_type == EdgeType.HIERARCHICAL:
                return self.nodes.get(parent_id)
        return None
    
    def get_document_sections(self, doc_id: str) -> List[SectionNode]:
        """Get all sections of a document"""
        sections = []
        for section_id in self.doc_to_sections.get(doc_id, []):
            sections.append(self.section_nodes[section_id])
        return sorted(sections, key=lambda x: x.metadata.get("start_pos", 0))
    
    def get_section_chunks(self, section_id: str) -> List[BaseNode]:
        """Get all chunks (text + table cells) of a section"""
        chunks = []
        for chunk_id in self.section_to_chunks.get(section_id, []):
            chunk = self.nodes.get(chunk_id)
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def get_related_documents(self, doc_id: str, relation: str = None) -> List[DocumentNode]:
        """Get cross-document related documents"""
        related = []
        for rel_doc_id in self.cross_doc_edges.get(doc_id, []):
            related.append(self.doc_nodes[rel_doc_id])
        return related

    def get_next_section(self, section_id: str) -> Optional[SectionNode]:
        """Get the next sequential section in the same document."""
        next_id = self.section_next.get(section_id)
        if not next_id:
            return None
        return self.section_nodes.get(next_id)

    def get_previous_section(self, section_id: str) -> Optional[SectionNode]:
        """Get the previous sequential section in the same document."""
        prev_id = self.section_prev.get(section_id)
        if not prev_id:
            return None
        return self.section_nodes.get(prev_id)

    def get_sequential_neighbors(self, section_id: str, hops: int = 1) -> List[SectionNode]:
        """Get previous/next section neighbors up to the given hop distance."""
        if hops <= 0:
            return []

        neighbors: List[SectionNode] = []
        seen = {section_id}
        prev_id = section_id
        next_id = section_id

        for _ in range(hops):
            prev_id = self.section_prev.get(prev_id)
            next_id = self.section_next.get(next_id)

            candidates = [cand for cand in (prev_id, next_id) if cand and cand not in seen]
            for cand in candidates:
                node = self.section_nodes.get(cand)
                if node:
                    neighbors.append(node)
                    seen.add(cand)

        return neighbors

    def _rebuild_sequential_maps(self):
        """Rebuild next/prev section maps from stored sequential edges."""
        self.section_next = {}
        self.section_prev = {}
        for from_id, edge_list in self.edges.items():
            for to_id, edge_type in edge_list:
                if edge_type == EdgeType.SEQUENTIAL:
                    self.section_next[from_id] = to_id
                    self.section_prev[to_id] = from_id

    def _reconstruct_sequential_edges_from_sections(self):
        """Reconstruct sequential edges from per-document section order."""
        for doc_id, section_ids in self.doc_to_sections.items():
            ordered_ids = sorted(
                section_ids,
                key=lambda sec_id: self.section_nodes[sec_id].metadata.get("start_pos", 0),
            )
            for prev_section_id, next_section_id in zip(ordered_ids, ordered_ids[1:]):
                self.add_sequential_edge(prev_section_id, next_section_id)
    
    def filter_by_metadata(self, node_type: NodeType, **kwargs) -> List[BaseNode]:
        """Filter nodes by metadata attributes"""
        results = []
        nodes_map = {
            NodeType.DOCUMENT: self.doc_nodes,
            NodeType.SECTION: self.section_nodes,
            NodeType.TEXT_CHUNK: self.text_chunk_nodes,
            NodeType.TABLE_CELL: self.table_cell_nodes
        }
        
        for node_id, node in nodes_map.get(node_type, {}).items():
            match = True
            for key, value in kwargs.items():
                if node.metadata.get(key) != value:
                    match = False
                    break
            if match:
                results.append(node)
        return results
    
    def save(self, path: str):
        """Save index to disk"""
        data = {
            "nodes": self.nodes,
            "edges": dict(self.edges),
            "reverse_edges": dict(self.reverse_edges),
            "doc_nodes": self.doc_nodes,
            "section_nodes": self.section_nodes,
            "text_chunk_nodes": self.text_chunk_nodes,
            "table_cell_nodes": self.table_cell_nodes,
            "doc_to_sections": dict(self.doc_to_sections),
            "section_to_chunks": dict(self.section_to_chunks),
            "cross_doc_edges": dict(self.cross_doc_edges),
            "section_next": self.section_next,
            "section_prev": self.section_prev,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load index from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.nodes = data["nodes"]
        self.edges = defaultdict(list, data["edges"])
        self.reverse_edges = defaultdict(list, data["reverse_edges"])
        self.doc_nodes = data["doc_nodes"]
        self.section_nodes = data["section_nodes"]
        self.text_chunk_nodes = data["text_chunk_nodes"]
        self.table_cell_nodes = data["table_cell_nodes"]
        self.doc_to_sections = defaultdict(list, data["doc_to_sections"])
        self.section_to_chunks = defaultdict(list, data["section_to_chunks"])
        self.cross_doc_edges = defaultdict(list, data["cross_doc_edges"])
        self.section_next = data.get("section_next", {})
        self.section_prev = data.get("section_prev", {})
        if not self.section_next and not self.section_prev:
            self._rebuild_sequential_maps()
        if not self.section_next and not self.section_prev:
            self._reconstruct_sequential_edges_from_sections()


class TOCParser:
    """Parse Table of Contents from financial reports"""
    
    @staticmethod
    def parse_html_toc(html_content: str) -> List[Dict[str, Any]]:
        """Parse TOC from HTML/XML format (e.g., SEC EDGAR filings)"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        sections = []
        # Find section headers (typical 10-K sections)
        section_patterns = [
            "Item 1", "Item 1A", "Item 1B", "Item 2", "Item 3",
            "Item 4", "Item 5", "Item 6", "Item 7", "Item 7A",
            "Item 8", "Item 9", "Item 9A", "Item 9B", "Item 10",
            "Item 11", "Item 12", "Item 13", "Item 14", "Item 15"
        ]
        
        for pattern in section_patterns:
            elements = soup.find_all(string=lambda text: text and pattern in text)
            for elem in elements:
                sections.append({
                    "title": elem.strip(),
                    "level": 1 if "Item" in elem else 2,
                    "position": elem.parent.sourceline if hasattr(elem.parent, 'sourceline') else 0
                })
        
        return sections
    
    @staticmethod
    def extract_section_content(document: str, section_title: str) -> str:
        """Extract content for a given section using regex"""
        import re
        pattern = rf"{section_title}.*?(?=(?:Item \d+|$))"
        match = re.search(pattern, document, re.DOTALL | re.IGNORECASE)
        return match.group(0) if match else ""
