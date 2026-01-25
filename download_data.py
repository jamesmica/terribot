#!/usr/bin/env python3
"""
Script to download parquet data files for Terribot.
This bypasses Git LFS limitations by hosting files externally.
"""
import os
import sys
import urllib.request
import urllib.error

# Base URL for data files (you'll need to host these files somewhere)
# Options: GitHub Releases, S3, Google Drive, or any public URL
# For now, we'll use GitHub Releases as an example
DATA_BASE_URL = "https://github.com/jamesmica/terribot/releases/download/data-v1.0/"

# Alternative: Use a different branch or repository backup
# DATA_BASE_URL = "https://example.com/terribot-data/"

DATA_DIR = "data"

# List of parquet files needed (complete list from repository)
PARQUET_FILES = [
    "aah.parquet", "acci.parquet", "act.parquet", "act_10.parquet", "act_5.parquet",
    "aeeh.parquet", "all.parquet", "all30.parquet", "apl.parquet", "artif.parquet",
    "assmat.parquet", "bas_rev.parquet", "be.parquet", "bpe.parquet", "ccas.parquet",
    "clc.parquet", "club.parquet", "cmf.parquet", "cmf_10.parquet", "cmf_5.parquet",
    "cmg.parquet", "co2.parquet", "conso.parquet", "dads.parquet", "data_es.parquet",
    "defm_an.parquet", "dgfip.parquet", "dipl.parquet", "dipl_10.parquet", "dipl_5.parquet",
    "dom_tra.parquet", "dpe.parquet", "dpetert.parquet", "dtou.parquet", "dtraco2.parquet",
    "dvf.parquet", "empl.parquet", "empl_10.parquet", "empl_5.parquet", "erp.parquet",
    "et_civil.parquet", "evo.parquet", "evo_10.parquet", "evo_5.parquet", "filo.parquet",
    "flores.parquet", "htou.parquet", "indices.parquet", "lic.parquet", "log.parquet",
    "log_10.parquet", "log_5.parquet", "ma.parquet", "menj.parquet", "mi.parquet",
    "mte.parquet", "odf.parquet", "paccueil.parquet", "paje.parquet", "pass_sport.parquet",
    "pre_par_e.parquet", "preleau.parquet", "prod.parquet", "ren.parquet", "rna.parquet",
    "rpls.parquet", "rsa.parquet", "saneau.parquet", "sitadl.parquet", "sources.parquet",
    "sup.parquet", "tcg.parquet", "type.parquet", "voit.parquet", "vote.parquet"
]

def download_file(filename, retries=3):
    """Download a single file with retry logic."""
    url = DATA_BASE_URL + filename
    filepath = os.path.join(DATA_DIR, filename)

    # Skip if file already exists and is not an LFS pointer
    if os.path.exists(filepath):
        # Check if it's an LFS pointer (ASCII text, starts with "version https://git-lfs")
        with open(filepath, 'rb') as f:
            content = f.read(100)
            if content.startswith(b'version https://git-lfs'):
                print(f"⚠️  {filename} is an LFS pointer, will re-download")
            else:
                file_size = os.path.getsize(filepath)
                if file_size > 200:  # LFS pointers are ~130 bytes
                    print(f"✅ {filename} already exists ({file_size} bytes)")
                    return True

    # Try to download
    for attempt in range(retries):
        try:
            print(f"📥 Downloading {filename} (attempt {attempt + 1}/{retries})...")
            urllib.request.urlretrieve(url, filepath)
            file_size = os.path.getsize(filepath)
            print(f"✅ Downloaded {filename} ({file_size} bytes)")
            return True
        except urllib.error.HTTPError as e:
            print(f"❌ HTTP Error {e.code}: {e.reason}")
            if attempt == retries - 1:
                print(f"⚠️  Failed to download {filename} after {retries} attempts")
                return False
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            if attempt == retries - 1:
                return False

    return False

def get_all_parquet_files():
    """Get list of all parquet files that should exist."""
    # Get from directory if it exists, otherwise use the predefined list
    if os.path.exists(DATA_DIR):
        existing = [f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')]
        return existing if existing else PARQUET_FILES
    return PARQUET_FILES

def main():
    """Main download function."""
    print("=" * 80)
    print("Terribot Data Downloader")
    print("=" * 80)
    print()

    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Get list of files to download
    parquet_files = get_all_parquet_files()
    print(f"📋 Found {len(parquet_files)} parquet files to check/download")
    print()

    # Download each file
    success_count = 0
    failed_count = 0

    for filename in parquet_files:
        if download_file(filename):
            success_count += 1
        else:
            failed_count += 1

    print()
    print("=" * 80)
    print(f"✅ Successfully verified/downloaded: {success_count} files")
    print(f"❌ Failed: {failed_count} files")
    print("=" * 80)

    if failed_count > 0:
        print()
        print("⚠️  WARNING: Some files could not be downloaded.")
        print("   Please check that the files are hosted at:")
        print(f"   {DATA_BASE_URL}")
        print()
        print("   To fix this:")
        print("   1. Create a GitHub Release named 'data-v1.0'")
        print("   2. Upload all parquet files to that release")
        print("   3. Run this script again")
        sys.exit(1)

    return 0

if __name__ == "__main__":
    sys.exit(main())
