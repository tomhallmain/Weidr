# Weidr - Advanced Visual Media Handler

Media workflow application for browsing, comparing, analyzing, and transforming visual files at scale.
Combines embedding-driven search/classification with batch file actions, format conversion, and tool integrations for practical library curation.

---

## Usage

Clone this repository and ensure Python 3 and the required packages are installed from requirements.txt. Optionally, run `pip install -r requirements-optional.txt` for platform-specific extras (e.g. keychain integration, EXIF tools). Note: some dependencies in requirements.txt (e.g. **torch**) may need to be installed from the provider’s site (e.g. [pytorch.org](https://pytorch.org)) for GPU support specific to your system.

Run `app_qt.py` to start the PySide6 (Qt) UI. For more details, see [USAGE.md](https://www.github.com/tomhallmain/Weidr/blob/master/USAGE.md).

---

## Media Browser

The UI can be used as a media file browser. The following features are available that your OS default photo viewer application may not have:
<details>
<summary>View Features</summary>
<ul>
    <li>Auto-resize images to fill the screen</li>
    <li>Auto-refresh directory files</li>
    <li>Slideshow (customizable)</li>
    <li>Optionally play and compare video files and other media - typically will use the first image found for the comparison.</li>
    <li>Go to file by string search or by index (1-based)</li>
    <li>Mark groups of files to enable quick transitions and comparisons</li>
    <li>Mark favorite media and access them quickly via the Favorites window</li>
    <li>Move, copy, and delete marked file groups without overwriting system clipboard</li>
    <li>Revert and modify historical file action changes</li>
    <li>Quickly find directories via recent directory picker window</li>
    <li>Stores session info about seen directories (useful for directories with many media files)</li>
    <li>Can be set up to run on user-defined list of files in place of a directory</li>
    <li><a href="https://github.com/tomhallmain/sd-runner" target="_blank">sd-runner</a> integration for image generation</li>
    <li><a href="https://github.com/tomhallmain/refacdir" target="_blank">refacdir</a> intgration for file operations</li>
    <li>Find related images and prompts from embedded Stable Diffusion workflows</li>
    <li>Sort files by related images and prompts</li>
    <li>View raw image metadata</li>
    <li>Content filtering of images and videos based on their text encoding similarity (automatically hide, move to dir, delete etc)</li>
    <li>Search and install image classifier models from Hugging Face directly in-app</li>
    <li>Create PDFs from marked files with customizable quality and compression options</li>
    <li>Password protection system for sensitive operations with configurable session timeouts</li>
    <li>Extract text using OCR from images</li>
    <li>Set custom title bar colors for specific directories</li>
    <li>In-window media playback controls (timeline and play/pause) for video and animated GIF files</li>
    <li>Apply custom aspect ratio settings to image display</li>
    <li>Capture screenshots from time-based media with a keyboard shortcut and configurable save directory</li>
    <li><a href="https://github.com/vslavik/diff-pdf" target="_blank">diff-pdf</a> integration for creating visual diffs from two files (including non-PDF)</li>
</ul>
</details>

For image files, zoom and drag functionality is available in both browsing mode as well as when viewing grouped media after a comparison has been run.

Note that depending on your configuration, videos, GIFs, PDFs, SVGs, and HTML may be disabled; use the file-type configuration window (`Ctrl+J`) or edit `image_types` / `video_types` in config. Default extensions and optional dependencies (HEIC, AVIF, VLC, etc.) are listed in [USAGE.md](USAGE.md#supported-file-types-defaults).

---

### Favorites Window

You can mark any media file (image, video, etc.) as a favorite and access all favorites quickly using the Favorites window (Ctrl+F). This is especially useful when working with directories containing many files, as it allows you to keep persistent preferred items easily accessible for future searches and actions.

### Directory Notes

The Directory Notes feature allows you to maintain persistent notes and marked files for individual directories. You can add notes to specific files, mark files for later reference, and export or import your notes and marked files as text or JSON files. This is separate from the runtime marked files used for moving files, making it useful for long-term organization and documentation of your media collections.

---

## Performing Media Comparisons

Group large media sets by visual similarity using both embedding and color-comparison modes, or search by similarity to an input image or text embedding, then refine analysis with classifier models (H5/PyTorch) and rule-driven actions. For fine-point document/image review workflows, marked files can also be compared with `diff-pdf` output to highlight precise differences. Multiple embedding models are supported:

<details>
<summary>View Embedding Models</summary>
- CLIP (default): 512D embeddings, high zero-shot performance
- SigLIP: 768D or 1024D embeddings, excellent retrieval performance
- ALIGN: 640D embeddings, high accuracy for retrieval
- FLAVA: 768D embeddings, good for complex reasoning
- X-VLM: 256D embeddings, efficient for region-text tasks - requires local copy of [X-VLM](https://github.com/zengyan-97/X-VLM)
- LAION: 1024D embeddings, high-quality visual-language understanding - based on CLIP ViT-H/14 architecture
- EVA CLIP: 1024D embeddings, large-scale CLIP variant from BAAI - requires `open-clip-torch`
- MetaCLIP: 1024D embeddings, Meta's improved CLIP training data curation on ViT-H/14
- V-JEPA 2: 1024D embeddings, video-primary world model encoder - supports images, video, and GIF; media-to-media similarity only (no text search)

Each model offers different tradeoffs between accuracy, speed, and resource usage. The default CLIP model provides a good balance for most use cases.
</details>

---

## Prevalidation Rules and Classifier Actions

The application includes a flexible prevalidation system that can automatically process media before they're shown to the user, as well as classifier actions that can be run ad-hoc on selected directories. Both are managed through a unified window. This is useful for:

<details>
<summary>View Use Cases</summary>
<ul>
<li>Automatically skipping, hiding, blurring or deleting unwanted media</li>
<li>Moving or copying media to specific directories based on content</li>
<li>Filtering media using CLIP embeddings, embedding prototypes, H5 image classifiers, PyTorch image classifiers, prompt string detection</li>
<li>Setting up rules that apply to specific directories</li>
<li>Running one-off classification actions on selected directories</li>
</ul>
</details>

Prevalidation rules and classifier actions can be configured with:

<details>
<summary>View Rule Options</summary>
<ul>
<li>Multiple validation types enabled simultaneously (OR logic - any type can trigger the action)</li>
<li>Positive and negative text prompts shared across embedding and prompt validation</li>
<li><strong>Embedding prototypes</strong>: Create prototype embeddings from directories of sample images, then compare images against these prototypes. Supports both positive and negative prototypes with configurable weighting (lambda) for fine-tuning similarity matching</li>
<li>Custom thresholds for embedding-based matching</li>
<li>Different actions (skip, hide, notify, move, copy, delete, add mark, blur)</li>
<li>Directory-specific rules</li>
<li>H5 model-based classification rules</li>
<li>PyTorch model-based classification rules (supports .pth, .pt, .safetensors, and .bin formats)</li>
</ul>
</details>

Prevalidations automatically run on media as you browse, while classifier actions can be executed manually on selected media directories when needed. These features are particularly useful for maintaining clean media collections and automating local content filtering, but can be disabled at any time if desired. The classifier action management window allows copying between types of classifier action to reduce the burden of action configuration.

Classifier models can be added manually or discovered through the in-app model manager, which supports searching Hugging Face repositories, viewing model cards, and installing selected model files.

You can find example classifier models that are known to work here:
- [Coherence Detection](https://huggingface.co/reddesert/coherence_detection) - A PyTorch ResNet-34 model for classifying AI-generated images into coherent, incoherent, or semi-incoherent categories
- [NSFW Model](https://github.com/FurkanGozukara/nsfw_model) - An H5 classifier model for filtering out some types of NSFW content

---

## Limitations

**NOTE** - It is not currently possible to undo or modify a delete action, however unless the delete folder is explicitly set to null in the config it is likely the deleted items will be saved in a trash folder before being fully removed.

