"""DOCX emitter package for producing styled book and marketing DOCX documents."""

from src.emitter.docx_emitter import (
    add_html_content_to_cell,
    add_paragraph_from_html_node,
    apply_formatting,
    assemble_docx,
    assemble_marketing_docx,
    delete_paragraph,
    html_to_docx,
    markdown_to_docx,
    process_mixed_content,
    process_node_content,
    set_page_numbering,
)

__all__ = [
    "add_html_content_to_cell",
    "add_paragraph_from_html_node",
    "apply_formatting",
    "assemble_docx",
    "assemble_marketing_docx",
    "delete_paragraph",
    "html_to_docx",
    "markdown_to_docx",
    "process_mixed_content",
    "process_node_content",
    "set_page_numbering",
]
