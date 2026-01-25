# 📦 Data Setup Guide - Terribot

## 🚨 Problem: Git LFS Quota Exceeded

The Terribot repository was previously using Git LFS (Large File Storage) to store 75 parquet data files. The LFS quota has been exceeded, causing Streamlit deployment to fail with this error:

```
Error downloading data/aah.parquet: This repository exceeded its LFS budget.
The account responsible for the budget should increase it to restore access.
```

## ✅ Solution: External Data Storage

We've removed Git LFS tracking and created a system to download data files externally. This solves the quota issue permanently.

## 🔧 Setup Instructions

### Step 1: Get the Parquet Files

You need to retrieve the actual parquet files. Choose one option:

#### Option A: Download from LFS (requires temporary quota increase)

If you have access to increase the LFS quota temporarily:

```bash
# Increase LFS quota on GitHub (Settings → Billing → Git LFS Data)
# Then pull the files:
git lfs pull
```

#### Option B: Access files from backup

If you have the files backed up elsewhere (local drive, another repository, etc.), copy them to the `data/` directory.

#### Option C: Download from another source

If the data is available from a public source or API, download it using the appropriate method.

### Step 2: Create a GitHub Release with the Data

Once you have the parquet files:

1. Create a new release on GitHub:
   ```bash
   # Tag and create release using GitHub CLI
   gh release create data-v1.0 --title "Data Files v1.0" --notes "Parquet data files for Terribot"
   ```

2. Upload all parquet files to the release:
   ```bash
   # Upload all parquet files from data/ directory
   cd data
   gh release upload data-v1.0 *.parquet
   ```

   Or manually:
   - Go to: https://github.com/jamesmica/terribot/releases/new
   - Tag: `data-v1.0`
   - Title: "Data Files v1.0"
   - Upload all 75 parquet files

### Step 3: Run the Download Script

Once files are uploaded to the release:

```bash
python download_data.py
```

This will download all parquet files from the GitHub Release to the `data/` directory.

### Step 4: Verify Setup

Check that all files are present:

```bash
ls -lh data/*.parquet | wc -l
# Should show: 75
```

Test the app locally:

```bash
streamlit run app.py
```

## 🚀 Deployment to Streamlit

Once the data files are in the GitHub Release, Streamlit will be able to clone the repository successfully because:

1. Git LFS is no longer used (no quota issues)
2. The repository size is small (only contains the download script)
3. Data files are downloaded on first run from GitHub Releases

### Automatic Data Download on Streamlit

To automatically download data files when the Streamlit app starts, add this to the beginning of `app.py` (after imports):

```python
import os
import subprocess

# Download data files if not present
if not os.path.exists("data") or len([f for f in os.listdir("data") if f.endswith('.parquet')]) < 75:
    print("[SETUP] Downloading data files...")
    subprocess.run(["python", "download_data.py"], check=True)
```

## 📋 Complete File List

The repository contains 75 parquet files:

- aah.parquet, acci.parquet, act.parquet, act_10.parquet, act_5.parquet
- aeeh.parquet, all.parquet, all30.parquet, apl.parquet, artif.parquet
- assmat.parquet, bas_rev.parquet, be.parquet, bpe.parquet, ccas.parquet
- clc.parquet, club.parquet, cmf.parquet, cmf_10.parquet, cmf_5.parquet
- cmg.parquet, co2.parquet, conso.parquet, dads.parquet, data_es.parquet
- defm_an.parquet, dgfip.parquet, dipl.parquet, dipl_10.parquet, dipl_5.parquet
- dom_tra.parquet, dpe.parquet, dpetert.parquet, dtou.parquet, dtraco2.parquet
- dvf.parquet, empl.parquet, empl_10.parquet, empl_5.parquet, erp.parquet
- et_civil.parquet, evo.parquet, evo_10.parquet, evo_5.parquet, filo.parquet
- flores.parquet, htou.parquet, indices.parquet, lic.parquet, log.parquet
- log_10.parquet, log_5.parquet, ma.parquet, menj.parquet, mi.parquet
- mte.parquet, odf.parquet, paccueil.parquet, paje.parquet, pass_sport.parquet
- pre_par_e.parquet, preleau.parquet, prod.parquet, ren.parquet, rna.parquet
- rpls.parquet, rsa.parquet, saneau.parquet, sitadl.parquet, sources.parquet
- sup.parquet, tcg.parquet, type.parquet, voit.parquet, vote.parquet

## 🔄 Alternative: Different Hosting

If GitHub Releases doesn't work for your use case, you can host the files elsewhere:

### Option 1: Cloud Storage (S3, Google Cloud Storage)

```python
# Modify DATA_BASE_URL in download_data.py:
DATA_BASE_URL = "https://your-bucket.s3.amazonaws.com/terribot-data/"
```

### Option 2: External Server

```python
# Modify DATA_BASE_URL in download_data.py:
DATA_BASE_URL = "https://your-server.com/terribot-data/"
```

### Option 3: Keep Small Files in Git

Files under 50MB can be stored directly in git (without LFS). If your parquet files are small enough:

```bash
# Simply add them to git normally
git add data/*.parquet
git commit -m "Add parquet files directly to git"
git push
```

## 📊 File Size Information

- Original LFS file (aah.parquet): ~572 KB
- Total estimated size (75 files × 572 KB): ~43 MB
- This is within GitHub's recommended repository size

**Recommendation**: If all files are similar size (~500 KB each), you can skip GitHub Releases and commit them directly to git!

## ❓ FAQ

**Q: Why did we use LFS in the first place?**
A: LFS was likely set up preemptively for large files, but the parquet files are actually small enough for regular git.

**Q: What if I can't access the LFS files at all?**
A: You'll need to regenerate the data from its original source, or contact whoever has access to the files.

**Q: Can I just increase the LFS quota?**
A: Yes, but this is a recurring cost. The external download approach or direct git storage is more sustainable.

**Q: How do I update data files later?**
A: Create a new release (data-v1.1, data-v1.2, etc.) and update DATA_BASE_URL in download_data.py.

## 🆘 Support

If you encounter issues:
1. Check that the GitHub Release exists and contains all files
2. Verify the release name matches DATA_BASE_URL in download_data.py
3. Test download_data.py locally first before deploying to Streamlit
4. Check Streamlit logs for any download errors

## 📝 Summary

This setup removes the dependency on Git LFS and provides a sustainable solution for managing data files in the Terribot project. The parquet files are small enough to either:

1. **Store in GitHub Releases** (recommended for Streamlit)
2. **Store directly in git** (simpler, if total size < 100 MB)
3. **Store in cloud storage** (for very large datasets)

Choose the option that best fits your workflow and infrastructure.
