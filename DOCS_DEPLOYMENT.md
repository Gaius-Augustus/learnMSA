# Documentation Deployment Setup

This repository uses GitHub Actions to automatically build and deploy Sphinx documentation to GitHub Pages.

## Setup Instructions

### 1. Enable GitHub Pages

1. Go to your repository settings: `https://github.com/Gaius-Augustus/learnMSA/settings/pages`
2. Under "Build and deployment":
   - **Source**: Select "GitHub Actions"
3. That's it! No need to select a branch.

### 2. Workflow Configuration

The workflow file `.github/workflows/docs.yml` is already configured to:
- Trigger on pushes to the `main` branch
- Build the Sphinx documentation
- Deploy using GitHub Pages Actions
- Use the `GITHUB_TOKEN` (automatically provided by GitHub)

### 3. First Deployment

To trigger the first deployment:

```bash
git add .github/workflows/docs.yml docs/ pyproject.toml
git commit -m "Add GitHub Actions workflow for documentation deployment"
git push origin main
```

### 4. Check Deployment Status

1. Go to the "Actions" tab in your repository
2. Watch the "Build and Deploy Documentation" workflow run
3. Once complete, your documentation will be available at:
   `https://gaius-augustus.github.io/learnMSA/`

## Manual Deployment

You can also manually trigger the workflow:
1. Go to Actions → Build and Deploy Documentation
2. Click "Run workflow"
3. Select the branch and click "Run workflow"

## Local Documentation Build

To build the documentation locally:

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.

## Troubleshooting

### Workflow fails with permission error
Make sure "Workflow permissions" in Settings → Actions → General is set to "Read and write permissions".

### Documentation not updating
1. Check the Actions tab for any errors
2. Clear the cache: Settings → Pages → Remove the deployment and re-enable
3. Force a rebuild by pushing a new commit or manually triggering the workflow

### 404 error on GitHub Pages
1. Make sure you selected "GitHub Actions" as the source in Settings → Pages
2. Check that the workflow completed successfully in the Actions tab
3. Wait a few minutes for GitHub's CDN to update
