# SPACE project page

A static, responsive page with no build step or frontend dependencies. Its layout
references the [DoRA](https://nbasyl.github.io/DoRA-project-page/) and
[EAFT](https://ymxyll.github.io/EAFT/) project pages; the HTML, CSS, and JavaScript
here are written for SPACE and use the repository's Apache-2.0 license.

Project page: <https://jianzhubao.github.io/SPACE/>.

## Preview locally

From the repository root:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory site
```

From the `experiments/` directory, use `--directory ../site` instead:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory ../site
```

Open <http://127.0.0.1:8765>. When working on a remote machine, forward port 8765
through your editor or SSH. All assets are local, and formulas use native MathML.

## Edit

- `index.html`: text, equations, and results.
- `static/css/style.css`: layout, colors, and mobile styles.
- `static/images/favicon.svg`: transparent triangular ruler favicon, redrawn as an SVG from the supplied visual reference.
- `static/js/results-tabs.js`: result tabs with mouse and keyboard navigation.
- `static/images/space-overview.png`: a copy of `assets/space-overview.png` at the
  repository root. Update both copies when replacing the figure.

The four `static/images/table-*.png` files are 375-DPI exports of Tables 1–4
from the manuscript, including the original captions, scores, and formatting.
Each is displayed in its own result tab and links to the full-resolution image;
replace the images when updating the tables. Each has a brief summary in `index.html`.

Rank selection uses squared singular-value energy, matching the package.
The author list and affiliations in `index.html` match the provided manuscript
information. The Code button links to `https://github.com/jianzhubao/SPACE`.
Add the confirmed `href` to the arXiv anchor and remove its `aria-disabled`
attribute when the paper URL is available. Add a BibTeX section once the citation
is available.
The canonical URL and Open Graph metadata use the project page URL above.

## Publish with GitHub Pages

1. Push the repository to GitHub when it is ready for release.
2. Open **Settings → Pages → Build and deployment → Source → GitHub Actions**.
3. Run **Deploy project page** from the Actions tab on `main`.

The workflow also runs when changes to `site/` or the workflow are pushed to
`main`. If your default branch has a different name, update `pages.yml`.
Only `site/` is uploaded. The project URL is
`https://jianzhubao.github.io/SPACE/`; assets use relative paths so they work
under the `/SPACE/` prefix. The root README's Project Page badge links to this URL.

See [GitHub's Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
