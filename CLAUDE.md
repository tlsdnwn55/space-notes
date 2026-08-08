# Space Notes Agent Instructions

This repository is Shin Woo-ju's (`tlsdnwn55`) personal tech blog & Markdown study notes site.

## How to Add New Articles & Group by Topics

When the user asks you to add a new article or study note:

1. **Create Markdown File with Topic Subfolders**:
   - Location: `src/content/docs/posts/<topic>/<slug-name>.md`
   - Example: `src/content/docs/posts/cs/operating-system.md`, `src/content/docs/posts/frontend/react.md`
   - Placing files in subfolders (`cs/`, `frontend/`, `backend/`, `ai/`) automatically groups them under collapsible topic headers in the left sidebar.
   - Ensure YAML Frontmatter at top:
     ```markdown
     ---
     title: "Article Title"
     description: "Short summary"
     ---

     # Main Content Here
     ```

2. **Local Test Files (git-ignored)**:
   - `src/content/docs/posts/local-test/` is git-ignored and used ONLY for local reference/testing.
   - Do NOT commit files inside `local-test/` to GitHub.

3. **Image Handling**:
   - Save image files inside `public/images/`
   - Reference images in Markdown using base paths:
     ```markdown
     ![Image Description](/space-notes/images/filename.png)
     ```

4. **Commit & Push to GitHub**:
   - Ask user for explicit permission before pushing.
   - Commit changes and push to `main` branch on repository `tlsdnwn55/space-notes`.
   - GitHub Actions will automatically deploy to GitHub Pages (`https://tlsdnwn55.github.io/space-notes/`).
