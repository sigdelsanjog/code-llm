"""
Download MedQuAD dataset from GitHub.
MedQuAD: Medical Question Answering Dataset
Source: https://github.com/abachaa/MedQuAD

Usage:
    python download_medquad.py
"""

import os
import sys
import subprocess
from pathlib import Path


def download_medquad():
    """
    Clone the MedQuAD repository from GitHub.
    
    The dataset contains medical QA pairs from various sources:
    - NIH (National Institutes of Health)
    - CDC, FDA, GARD, etc.
    
    Total: ~47,000 medical question-answer pairs
    """
    
    # Get the script's directory
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / "dataset"
    medquad_dir = dataset_dir / "MedQuAD"
    
    # Create dataset directory if it doesn't exist
    dataset_dir.mkdir(exist_ok=True)
    
    print(f"Dataset directory: {dataset_dir.absolute()}")
    
    # Check if MedQuAD already exists
    if medquad_dir.exists():
        print(f"\n⚠️  MedQuAD already exists at: {medquad_dir.absolute()}")
        response = input("Do you want to remove and re-download? (y/n): ").strip().lower()
        
        if response == 'y':
            print("Removing existing MedQuAD directory...")
            subprocess.run(["rm", "-rf", str(medquad_dir)], check=True)
        else:
            print("Keeping existing dataset. Exiting.")
            return
    
    # Clone the repository
    print("\n📥 Downloading MedQuAD dataset...")
    print("This may take a few minutes depending on your connection.\n")
    
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/abachaa/MedQuAD.git", str(medquad_dir)],
            check=True,
            cwd=dataset_dir
        )
        
        print("\n✅ MedQuAD dataset downloaded successfully!")
        print(f"Location: {medquad_dir.absolute()}")
        
        # Show dataset structure
        print("\n📊 Dataset structure:")
        collections = sorted([d for d in medquad_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
        
        if collections:
            print(f"Found {len(collections)} collections:")
            for collection in collections:
                xml_files = list(collection.glob("*.xml"))
                print(f"  - {collection.name}: {len(xml_files)} files")
        
        # Count total QA pairs (rough estimate from XML files)
        all_xml_files = list(medquad_dir.glob("*/*.xml"))
        print(f"\nTotal XML files: {len(all_xml_files)}")
        
        print("\n💡 Next steps:")
        print("1. Inspect the XML structure")
        print("2. Write a parser to extract Q&A pairs")
        print("3. Convert to causal text format for LM training")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading dataset: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("MedQuAD Dataset Downloader")
    print("=" * 60)
    download_medquad()
