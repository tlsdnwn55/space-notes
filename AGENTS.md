# Space Notes Agent Instructions

This repository is Shin Woo-ju's (`tlsdnwn55`) personal tech blog & Markdown study notes site.

## How to Add New Articles

When the user asks you to add a new article or study note:

1. **Create Markdown File**:
   - Location: `D:\space-notes\src\content\docs\posts\<slug-name>.md`
   - Ensure YAML Frontmatter at top:
     ```markdown
     ---
     title: "Article Title"
     description: "Short summary"
     ---

     # Main Content Here
     ```

2. **Image Handling**:
   - Save image files inside `D:\space-notes\public\images\`
   - Reference images in Markdown using:
     ```markdown
     ![Image Description](/space-notes/images/filename.png)
     ```

3. **Commit & Push to GitHub**:
   - Commit changes and push to `main` branch on repository `tlsdnwn55/space-notes`.
   - GitHub Actions will automatically deploy to GitHub Pages (`https://tlsdnwn55.github.io/space-notes/`).
