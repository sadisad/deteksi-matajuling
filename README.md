# deteksi-matajuling

Two-stage eye screening prototype for strabismus (mata juling). Built as a teaching tool for how a screening pipeline should behave when it is not sure, not as a diagnostic device.

**Educational prototype. Not a medical device, not a diagnosis, and not validated against a clinical dataset.**

## What it does

A photo of a face goes through two models in sequence:

1. **Eye detection** — `model_mata.pt` (YOLO) locates the eye regions in the frame.
2. **Strabismus classification** — the detected eyes are merged into one horizontal strip, padded by 20 px, and `model_juling.pt` (YOLO) classifies that strip.

If stage 1 finds nothing, the pipeline falls back to OpenCV's `haarcascade_eye.xml`, then to the whole frame. If stage 2 finds nothing inside the strip, the result is reported as **inconclusive** — not normal. See [How the decision is made](#how-the-decision-is-made).

Two front ends share the same weights:

- `app.py` — Streamlit app. Offers camera capture with a face-position guide (a yellow ROI box, 20–80% horizontally and 5–90% vertically) and upload. Rejects frames with no face, more than one face, or a face outside the ROI before running inference.
- `main.py` — Tkinter desktop script. Opens a file picker, runs both stages, and shows the annotated result in an OpenCV window.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py      # browser UI with camera or upload
python main.py            # desktop picker + OpenCV window
```

CPU only (`device='cpu'` in both scripts). On macOS and Linux, `main.py` needs a display for the OpenCV window; the Streamlit app works headless.

## How the decision is made

The decision lives in `screening.py`, separate from the UI and from Ultralytics, so it can be tested without weights or a camera. `decide()` maps stage-2 detections onto one of three outcomes:

| Situation | Outcome | Confidence |
| :--- | :--- | :--- |
| Any box labelled `JULING` or `STRABISMUS` | `STRABISMUS (JULING)` | highest such box confidence |
| Boxes, none of them strabismus | `NORMAL` | highest box confidence |
| Stage 1 found no eyes (stage 2 ran on the fallback frame) | `TIDAK DAPAT DISIMPULKAN` | none |
| Stage 2 returned zero boxes | `TIDAK DAPAT DISIMPULKAN` | none |

The rule that matters: a model that produces no boxes has not said "normal", it has said nothing. Abstentions carry no number — `format_result()` returns the bare status string and never a percentage, because a percentage would imply a measurement that did not happen.

```python
from screening import Detection, decide, format_result

decide([]).status                                  # 'TIDAK DAPAT DISIMPULKAN'
decide([Detection("NORMAL", 0.87)]).status          # 'NORMAL'
decide([Detection("NORMAL", 0.95),
        Detection("JULING", 0.61)]).status          # 'STRABISMUS (JULING)'
```

Both front ends call this same function. `app.py` renders the inconclusive case as a warning with the reason; `main.py` shows it on the frame and raises a dialog.

This is still a threshold on a single model's box confidence, not a calibrated probability — see the limitations below.

## Known limitations

These are real and known. They are listed because a screening tool that hides them is worse than one that admits them.

- **Confidence is uncalibrated.** A box score of 0.7 is a detector score, not the probability that the patient has strabismus. No temperature scaling, no reliability diagram, no decision threshold tuned against outcomes.
- **The stage-2 score threshold is fixed at 0.5** in both front ends, chosen by hand and never validated.
- **A detection failure and a genuine negative now look different, but both are still uninformative.** `INCONCLUSIVE` tells the user the model declined; it does not tell them whether the eye was present and normal or absent entirely.
- **No recorded accuracy.** No evaluation set, no confusion matrix, and no held-out split ships with this repository. No number in this README would be honest, so none is given.
- **Training is not reproducible from here.** `models/` contains three weight files (about 35 MB total) but no training script, no dataset manifest, and no split definitions.
- **One face, front-facing.** The ROI gate rejects multiple faces; profile views and occlusions are untested.
- **Tests cover the decision layer only.** `tests/test_screening.py` runs nine cases over `decide()` and `format_result()` with no model involved, and CI runs them on every push. Nothing tests the models themselves: there is no fixture image whose expected label is asserted, so a weight swap that degrades accuracy would still pass.
- **Not a replacement for a cover test.** Strabismus screening in practice involves more than one photograph at more than one gaze angle.

## What this project is useful for

Reading it as a worked example of the gap between a model that answers and a pipeline that behaves. The commit history is the interesting part: the first version turned "no boxes" into "normal, 100% confident", which is the same defect that shows up in production screening systems, in a codebase small enough to read in one sitting. The fix was to give the pipeline a way to decline, and to write down the cases where it does.

## Tests

```bash
python -m unittest discover -s tests -v
```

Nine cases, no weights, no camera, no GPU. `.github/workflows/tests.yml` runs them on every push and pull request, plus a `py_compile` pass over the front ends.

## Stack

Python · Ultralytics YOLO · OpenCV (Haar cascades, drawing) · Streamlit · Tkinter · NumPy · Pillow

## License

No license file yet, which means all rights are reserved by default. Open an issue if you want to use this.

## Bahasa

Antarmuka aplikasi berbahasa Indonesia. Wajah harus berada di area kotak kuning dan menghadap lurus ke kamera. Hasil hanya untuk pembelajaran, bukan diagnosis.
