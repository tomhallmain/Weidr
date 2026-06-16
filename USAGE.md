# Weidr - Usage

Clone this repository and ensure Python 3 and the required packages are installed from requirements.txt.

Run `python app_qt.py` to start the PySide6 (Qt) UI.

You can also run comparison from the command line. The compare logic lives in the `compare` package. From the project root:

```
python -m compare.compare_embeddings_clip --dir /path/to/media
python -m compare.compare_colors --dir /path/to/media
```

Supported command line options (embedding and color modules): `--dir`, `--counter`, `--include`, `--search`, `-o`/`--overwrite`, `--threshold`, `-v` (verbose), `-h`/`--help`. Embedding modules use `--threshold=float` (similarity, default 0.9); color uses `--threshold=int` (color diff) and `--use_thumb`. Multiple embedding modes exist (e.g. `compare_embeddings_clip`, `compare_embeddings_siglip`, `compare_embeddings_laion`); each shares the same CLI.

<details>
<summary>View Usage Details</summary>

Useful for detecting duplicates or finding associations between large unstructured sets of media files. File management controls are available after the analysis has completed.

Individual media files can be passed to search against the full data set by passing flag `--search` with the path of the search file, or setting a search file in the UI before running comparison.

The color matching compare mode is faster than embedding comparison but less robust. In the group comparison case, since every image must be compared to every other image the time complexity is $\mathcal{O}(n^2)$. To remedy this issue for large media sets, set the `store_checkpoints` config setting to enable process caching to close and pick up where you left off previously, but ensure no files are added or removed from the comparison directory before restarting a compare job.

When using embedding compare modes, you can search your image-based media files by text - both positive and negative. Commas will break the texts to search into multiple parts, to be combined in a final set of results. If there is a good embedding signal for the search texts it will likely return the media files you are looking for. It will take a while to load the first time as embeddings need to be generated. If a list of preset text searches is defined in your config JSON, you can cycle between them with the dedicated shortcut found below.

If a search image is set simultaneously with search text, its embedding will be factored into the search at a weight equal to a single search text part.
</details>

---

## Running Tests

Run from the project root with `pytest` (requires `pytest` and `pytest-qt` from `requirements.txt`). To run only the fast unit tests: `pytest tests/unit/`.

---

## Configuration

<details>
<summary>View Configuration Details</summary>

`locale` supports any of the following locales:

- en (English)
- de (Deutsh)
- fr (Français)
- es (Español)
- it (Italiano)
- pt (Português)
- ru (Русский)
- ja (日本語)
- ko (한국어)
- zh (中文)

`clip_model` defines the CLIP model to use for generating CLIP embeddings.

`image_types` and `video_types` define which extensions appear in directory browse lists (merged into `file_types` at startup). Additional categories are toggled separately:

| Config flag | Effect |
|-------------|--------|
| `enable_videos` | Include `video_types` in browse |
| `enable_gifs` | Include `.gif` |
| `enable_pdfs` | Include `.pdf` |
| `enable_svgs` | Include `.svg` |
| `enable_html` | Include `.html` and `.htm` |

You can change extension lists in `config.json` or the **type configuration** window (`Ctrl+J`). Defaults match `configs/config_example.json`.

### Supported file types (defaults)

**Still images** (`image_types`): `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.webp`, `.bmp`, `.heic`, `.heif`, `.avif`, `.ico`, `.jfif`, `.jpe`, `.pjpeg`, `.pjp`, `.apng`, `.jxl`, `.dib`, `.psd`

- **HEIC / HEIF / AVIF** need optional Pillow plugins (`pillow-heif`, `pillow-avif-plugin`) on some systems; Qt may decode others natively.
- **JPEG XL (`.jxl`)** is in the default browse list but has **no Qt reader**; install `pip install pillow-jxl-plugin` so Pillow can decode (libjxl is bundled with that wheel on most platforms). Without it, `.jxl` files appear in browse but may fail to display.
- **Animated WebP and APNG** play like GIF when Qt reports animation (`QMovie`); otherwise they show as a static frame. `.webp` and `.apng` are in `image_types`; animated playback does not require extra packages beyond Qt.
- **`.dib`** is the same bitstream as `.bmp`; **`.psd`** decodes via Pillow's built-in PSD reader (composite/flattened view only — no per-layer access). Neither needs an optional plugin.

**Video** (`video_types`, requires `enable_videos`): `.mp4`, `.mkv`, `.avi`, `.wmv`, `.mov`, `.flv`, `.webm`, `.m4v`, `.ogv`, `.mpeg`, `.mpg` — playback uses **VLC** when installed.

**Audio** (`audio_types`, requires `enable_audio`): `.mp3`, `.flac`, `.ogg`, `.opus`, `.wav`, `.m4a`, `.m4b`, `.aac`, `.wma`, and related extensions — playback uses **VLC**; not included in embedding compare or classifier prevalidation.

**Other browse types** (flags above): GIF, PDF (first page for compare/cache), SVG, HTML (HTML uses headless render + first-page raster for cache; can be slow on first open).

**Not in defaults:** Camera RAW (`.cr2`, `.nef`, `.dng`, etc.) and other legacy/scientific-imaging extensions (JPEG 2000, Netpbm, `.qoi`, `.xpm`, `.sgi`/`.rgb`/`.rgba`); add an extension to `image_types` if you want them in browse (decode may still need extra libraries — RAW in particular needs a dedicated develop pipeline, not just an extension).

**Go to file** can open paths outside `file_types`; browse and compare gathering respect the configured lists.

`media_volume_use_eq` controls how per-instance video volume is applied. When `true` (default), an audio equalizer preamp is used to give each player its own independent volume level. When `false`, the legacy `audio_set_volume` call is used instead, which is global across all players but has no equalizer drawbacks (preamp floor, slight transition delay).

`file_check_interval_seconds` defines the interval between auto-updates to identify recent file changes.

`slideshow_interval_seconds` defines the interval between slideshow transitions.

`sort_by` defines the default media browsing sort setting upon starting the application.

`trash_folder` defines the target folder for media deletion. If not set, deletion will send the file to your system's default trash folder.

`enable_prevalidations` enables the prevalidation system. When enabled, prevalidation rules will be applied to media before they are shown.

`image_classifier_models` defines a list of classifier model definitions used by prevalidations and classifier actions. These can now be managed directly in the UI (see the **Hugging Face Model Manager** section below), but manual config editing is still supported.

If the `sd_prompt_reader_loc` config setting is pointing to your local copy of [stable-diffusion-prompt-reader](https://github.com/receyuki/stable-diffusion-prompt-reader) then opening image details for an image with a stable diffusion prompt will give prompt information found in the image.

`tag_suggestions_file` should point to a JSON list that provides suggested tags for images for easy access in adding tags, if desired.

`file_path_json_path` should be set to the path for the file path JSON, if setting `use_file_path_json` is set to true.

`text_embedding_search_presets_exclusive` enables the search results returned by preset search texts to be exclusive of eachother to more accurately categorize. Note that since some text embeddings have a much stronger signal than others clustering on those searches can occur.

`store_checkpoints` will cache a group comparison process at certain checkpoints for later restart.
</details>

---

## Hugging Face Model Manager

Use the Hugging Face model manager window to search, install, edit, and remove classifier model entries from the UI (shortcut: `Ctrl+Shift+M`).

It keeps `image_classifier_models` in config in sync with what you manage in the window.

If editing manually, each `image_classifier_models` entry should include:

- `model_name`: unique display name for the model
- `model_location`: local model path (`.h5`, `.pth`, `.pt`, `.safetensors`, `.bin`, etc.)
- `model_categories`: non-empty list of class/category names
- `backend`: `auto`, `hdf5`/`tensorflow`, or `pytorch`
- Optional: `use_hub_keras_layers` (H5 models)
- Optional Hugging Face metadata: `hf_repo_id`, `hf_selected_filename`
- Optional advanced backend args in `model_kwargs` (for example `architecture_module_name`, `architecture_class_path`, `weights_only`, `device`, `input_shape`, `use_transformers_auto_model`, `hf_pretrained_path`)

For many `.safetensors` models, set architecture details in `model_kwargs` (at minimum `architecture_module_name` unless using transformers auto-model loading). See `configs/config_example.json` for examples.

---

## Key and Mouse Bindings

While the UI elements support normal usage in most cases, there are many bindings that enable extended functionality, mostly to minimize UI content unrelated to image viewers.

Press Shift+H to open up a help window with all key bindings. A directory with media files must be set before most of the bindings will have any effect. The group bindings are only functional in GROUP mode after a comparison has been run.

---

## Move Marks Window

This window helps with efficient filing of file marks.

<details>
<summary>View File Marks Details</summary>
<p>When the move marks window is open -- with or without GUI -- marks can be moved to a target directory by pressing the Enter key, or with the GUI elements if visible. After pressing the Enter key, a number of things can occur:</p>

<ul>
<li>If no target directories have been set, a folder picker window will open to set a new directory.</li>
<li>If a marks action has been run previously, simply pressing Enter without a filter set will use the directory last used for the move or copy action.</li>
<li>If target directories have been set and a filter is set, the move or copy operation will use the first target directory in the filtered list.</li>
<li>If shift key is pressed along with Enter, the files will be copied instead of moved.</li>
<li>If control key is pressed, any previously marked directories will be ignored and a folder picker window will open to set a new target directory.</li>
<li>If alt key is pressed, the penultimate mark target dir will be used as target directory. This is useful when you want to successively copy files to one directory and then move them to another, without having to re-filter each time.</li>
</ul>
<p>Simply typing letters while the mark window is open will filter the list of mark target directories, even if the GUI is not present. The backspace key will delete letters from the filter. You can scroll through the list of saved target directories using arrow keys.</p>
<p>To bypass the move marks window, use the Ctrl+R or Ctrl+E shortcuts to immediately run the previous and penultimate actions respectively on the current selection. You can also use number keys or Ctrl+T as hotkeys for persistent marks actions. To see the full list of file action hotkeys and their current settings open the hotkey actions window by pressing Ctrl+H on the marks window.</p>
<p>Ctrl+Z will undo the previous file marks move or copy action. If an earlier action needs to be reversed or modified, open the file actions window to verify the action in the history list and reverse it via the UI.</p>
<p>The same window includes a <code>Create PDF</code> action: it combines marked files into a single PDF and lets you choose quality/compression behavior.</p>
<p>The same window also includes a <code>Diff PDFs</code> action: with exactly two marked files, it creates a visual diff PDF using <code>diff-pdf</code>. If the marked inputs are not PDFs, they are converted to temporary PDFs first.</p>
</details>

---

## File Actions Window

The file actions window displays a certain number of completed actions, as defined in the config JSON. Similar to the move marks window, typing will add to a text filter that filters the actions by the target directory basenames.

On this window the previous file action media can be viewed and reversed or the action can be modified if desired.

---

## File Action Sets

File action sets let you combine multiple copy and/or move actions into a single operation. Open the window from the "File action sets" button in the File actions section of the sidebar.

The top section shows all actions in the pool as cards. Click a card to select or deselect it — selected cards form the current set. Only one Move action may be selected at a time. Use **Add from Hotkeys** or **Add from Recent** to populate the pool; ✕ removes a card from the pool.

**Execute** runs all selected actions against the current marks (auto-marking the active file if nothing is marked). Copies run before any move. Frequent selections can be saved as named presets and loaded by clicking a preset card.

Press **Ctrl+Shift+T** in any app window to execute the current set on the active file without opening the window.