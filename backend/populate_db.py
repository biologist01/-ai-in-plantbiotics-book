"""
Script to populate the vector database with textbook content from local markdown files
"""
from app.vector_db import VectorDB
import os
import sys
from pathlib import Path
import re
from typing import List, Dict, Optional

def _infer_language(path: Path) -> str:
    p = str(path).lower()
    return "urdu" if "urdu" in p else "english"


def _infer_background(path: Path) -> str:
    p = str(path).lower()
    if "hardware" in p:
        return "hardware"
    if "software" in p:
        return "software"
    return ""


def _generate_questions(doc_title: str, section_heading: str, source: str) -> List[str]:
    """Generate a practical (finite) set of common user questions for a chunk."""
    topic = (section_heading or doc_title or "").strip()
    questions = set()

    if topic:
        questions.update(
            {
                f"Explain {topic}.",
                f"Summarize {topic}.",
                f"What is {topic}?",
                f"Why is {topic} important in plant biotechnology?",
                f"How is {topic} used in AI for plant biotechnology?",
                f"What are key takeaways of {topic}?",
            }
        )

    # Module/chapter hints based on path
    m = re.search(r"module[-_/ ](\d+)", source.lower())
    if m:
        n = m.group(1)
        questions.update(
            {
                f"Explain module {n}.",
                f"What is covered in module {n}?",
                f"Explain chapter {n}.",
                f"What is covered in chapter {n}?",
            }
        )

    # Doc-level catchalls
    if doc_title:
        questions.update(
            {
                f"What is covered in {doc_title}?",
                f"Explain the main ideas in {doc_title}.",
            }
        )

    # Keep it bounded
    ordered = []
    for q in sorted(questions):
        if len(ordered) >= 8:
            break
        ordered.append(q)
    return ordered


def read_markdown_files(docs_dir: Path, project_root: Path, include_generated_questions: bool = True) -> List[Dict[str, str]]:
    """Read all markdown files from a docs directory and return chunked documents."""
    documents: List[Dict[str, str]] = []
    docs_path = Path(docs_dir)
    
    for md_file in docs_path.rglob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract title from first # heading or use filename
            title = md_file.stem
            lines = content.split('\n')
            for line in lines:
                if line.startswith('# '):
                    title = line.replace('# ', '').strip()
                    break

            # Improved chunking: split by ## headings to keep sections together
            sections = []
            current_section = []
            current_heading = title

            for line in lines:
                if line.startswith('## '):
                    # Save previous section if it has content
                    if current_section:
                        section_text = '\n'.join(current_section).strip()
                        if len(section_text) > 50:
                            sections.append({
                                "heading": current_heading,
                                "content": section_text
                            })
                    # Start new section
                    current_heading = line.replace('## ', '').strip()
                    current_section = [line]
                else:
                    current_section.append(line)

            # Don't forget the last section
            if current_section:
                section_text = '\n'.join(current_section).strip()
                if len(section_text) > 50:
                    sections.append({
                        "heading": current_heading,
                        "content": section_text
                    })

            # Create documents from sections
            for i, section in enumerate(sections):
                language = _infer_language(md_file)
                background = _infer_background(md_file)
                source_rel = str(md_file.relative_to(project_root))

                documents.append({
                    "title": f"{title} - {section['heading']}",
                    "content": section['content'],
                    "source": source_rel,
                    "language": language,
                    "background": background,
                    "doc_type": "content",
                })

                if include_generated_questions:
                    for q in _generate_questions(title, section['heading'], source_rel):
                        # Store as Q+context for better matching on question-style queries
                        documents.append({
                            "title": f"FAQ - {q}",
                            "content": f"Q: {q}\n\nRelevant book content:\n{section['content']}",
                            "source": source_rel,
                            "language": language,
                            "background": background,
                            "doc_type": "faq",
                        })
        except Exception as e:
            print(f"Error reading {md_file}: {e}")
    
    return documents

def main():
    project_root = Path(__file__).parent.parent

    # Index all generated docs variants (English/Urdu + hardware/software versions)
    docs_dirs = [
        project_root / "website" / "docs",
        project_root / "website" / "docs-software",
        project_root / "website" / "docs-hardware",
        project_root / "website" / "docs-urdu",
        project_root / "website" / "docs-urdu-software",
        project_root / "website" / "docs-urdu-hardware",
    ]

    existing_dirs = [d for d in docs_dirs if d.exists()]
    if not existing_dirs:
        print("ERROR: No docs directories found under website/")
        sys.exit(1)

    documents: List[Dict[str, str]] = []
    for docs_dir in existing_dirs:
        print(f"Reading markdown files from: {docs_dir}")
        documents.extend(read_markdown_files(docs_dir, project_root=project_root, include_generated_questions=True))

    if not documents:
        print("ERROR: No content found")
        sys.exit(1)

    print(f"Found {len(documents)} chunks (content + generated FAQ) from markdown files")

    print("\nSetting up vector database...")
    vector_db = VectorDB()
    vector_db.create_collection()

    print("\nAdding documents to vector database...")
    vector_db.add_documents(documents, batch_size=96)

    print("\nIndexing complete!")
    print(f"   - Chunks indexed: {len(documents)}")
    print(f"   - Collection: {vector_db.collection_name}")

if __name__ == "__main__":
    main()
