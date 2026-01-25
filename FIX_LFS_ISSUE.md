# 🔧 Git LFS Quota Issue - FIXED

## Problem

Streamlit deployment was failing with:
```
Error downloading data/aah.parquet: This repository exceeded its LFS budget.
```

## Root Cause

- 75 parquet files (~43 MB total) were tracked with Git LFS
- GitHub LFS quota exceeded, preventing repository cloning
- Streamlit couldn't deploy the app

## Solution Implemented

### ✅ Changes Made

1. **Removed Git LFS tracking** (`.gitattributes`)
   - Parquet files no longer require LFS
   - Files can be stored directly in git or downloaded externally

2. **Created data download script** (`download_data.py`)
   - Downloads all 75 parquet files from GitHub Releases
   - Handles retries and error checking
   - Can be run manually or automatically

3. **Added automatic data setup** (`app.py`)
   - Checks for missing/LFS pointer files on startup
   - Automatically runs download script if needed
   - Gracefully handles errors with helpful messages

4. **Created comprehensive documentation** (`DATA_SETUP.md`)
   - Step-by-step setup instructions
   - Multiple hosting options (GitHub Releases, S3, direct git)
   - Troubleshooting guide

### 📋 Next Steps Required

**IMPORTANT**: To complete the fix, you must do ONE of the following:

#### Option 1: Store Files in GitHub Releases (Recommended for Streamlit)

```bash
# 1. Get the parquet files (temporarily increase LFS quota or use backup)
git lfs pull

# 2. Create a GitHub release with the data
gh release create data-v1.0 --title "Data Files v1.0" --notes "Parquet data files"

# 3. Upload all parquet files
cd data
gh release upload data-v1.0 *.parquet

# 4. Test the download script
cd ..
python download_data.py
```

#### Option 2: Store Files Directly in Git (Simpler)

If your parquet files are small enough (~43 MB total is fine):

```bash
# 1. Get the parquet files
git lfs pull

# 2. Add them to git normally (without LFS)
git add data/*.parquet

# 3. Commit and push
git commit -m "Store parquet files directly in git (removed LFS)"
git push
```

#### Option 3: Use External Storage (S3, Cloud Storage)

Upload files to your cloud storage and update `DATA_BASE_URL` in `download_data.py`.

### 🚀 Deployment

Once you complete ONE of the above options:

1. **Merge this PR to main**
2. **Streamlit will automatically deploy** (no more LFS errors!)
3. **App will download data on first run** (if using GitHub Releases)

### 📊 File Information

- **Total files**: 75 parquet files
- **Example file size**: ~572 KB (aah.parquet)
- **Estimated total**: ~43 MB
- **Recommendation**: Store directly in git (well within GitHub's 100MB/file limit)

### 🎯 Benefits

- ✅ No more LFS quota issues
- ✅ Streamlit deployment works
- ✅ Automatic data download on startup
- ✅ Multiple hosting options
- ✅ Better control over data versioning

### 📝 Files Changed

- `.gitattributes` - Removed LFS tracking
- `app.py` - Added automatic data download on startup
- `download_data.py` - New script to download parquet files
- `DATA_SETUP.md` - Complete setup documentation
- `FIX_LFS_ISSUE.md` - This summary

### ❓ FAQ

**Q: Can I just increase the LFS quota?**
A: Yes, but it's a recurring cost. The new approach is more sustainable and free.

**Q: Will the app work right now?**
A: After merging and completing ONE of the next steps above, yes!

**Q: Which option should I choose?**
A: **Option 2** (direct git) is simplest if files are <50MB each. **Option 1** (GitHub Releases) keeps the repo lighter.

**Q: What if I can't access the LFS files?**
A: You'll need to regenerate the data from source or contact whoever has the files.

### 🆘 Need Help?

See `DATA_SETUP.md` for detailed instructions and troubleshooting.

---

**Status**: ✅ Code changes complete, awaiting data file setup (see Next Steps above)
