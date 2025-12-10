"""
Script to populate the vector database with all textbook content from local markdown files
Includes English and Urdu documentation from all directories
"""
from app.vector_db import VectorDB
import os
import sys
from pathlib import Path

def read_markdown_files(docs_dirs):
    """Read all markdown files from multiple docs directories"""
    documents = []

    for docs_dir in docs_dirs:
        docs_path = Path(docs_dir)

        if not docs_path.exists():
            print(f"Warning: Docs directory not found: {docs_path}")
            continue

        print(f"Reading markdown files from: {docs_path}")

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
                    # Determine language from directory path
                    language = "english"
                    if "urdu" in str(md_file).lower():
                        language = "urdu"

                    documents.append({
                        "title": f"{title} - {section['heading']}",
                        "content": section['content'],
                        "source": str(md_file.relative_to(docs_path.parent.parent)),
                        "language": language
                    })

            except Exception as e:
                print(f"Error reading {md_file}: {e}")

    return documents

def main():
    # Paths to all docs directories
    project_root = Path(__file__).parent.parent
    docs_dirs = [
        project_root / "website" / "docs",                    # English base docs
        project_root / "website" / "docs-hardware",          # English hardware docs
        project_root / "website" / "docs-software",          # English software docs
        project_root / "website" / "docs-urdu",              # Urdu base docs
        project_root / "website" / "docs-urdu-hardware",     # Urdu hardware docs
        project_root / "website" / "docs-urdu-software"      # Urdu software docs
    ]

    print("Reading markdown files from all documentation directories...")
    documents = read_markdown_files(docs_dirs)

    if not documents:
        print("ERROR: No content found in any directory")
        sys.exit(1)

    # Count documents by language
    english_docs = [d for d in documents if d.get('language') == 'english']
    urdu_docs = [d for d in documents if d.get('language') == 'urdu']

    print(f"\nFound {len(documents)} total text chunks:")
    print(f"   - English chunks: {len(english_docs)}")
    print(f"   - Urdu chunks: {len(urdu_docs)}")

    print("\nSetting up vector database...")
    vector_db = VectorDB()

    # Clear existing collection if it exists
    try:
        vector_db.client.delete_collection(vector_db.collection_name)
        print(f"Cleared existing collection: {vector_db.collection_name}")
    except:
        pass

    vector_db.create_collection()

    print("\nAdding documents to vector database...")
    vector_db.add_documents(documents)

    print("\nIndexing complete!")
    print(f"   - Total chunks indexed: {len(documents)}")
    print(f"   - Collection: {vector_db.collection_name}")
    print(f"   - Languages: English ({len(english_docs)}), Urdu ({len(urdu_docs)})")

if __name__ == "__main__":
    main()