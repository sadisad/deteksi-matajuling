# deteksi-matajuling

Two-stage eye screening prototype for strabismus (mata juling). Built as a teaching tool for how a screening pipeline should behave when it is not sure, not as a diagnostic device.

**Educational prototype. Not a medical device, not a diagnosis, and not validated against a clinical dataset.**

## What it does

A photo of a face goes through two models in sequence:

1. **Eye detection** — `model_mata.pt` (YOLO) locates the eye regions in the frame.
2. **Strabismus classification** — the detected eyes are merged into one horizontal strip, padded by 20 px, and `model_juling.pt` (YOLO) classifies that strip.

If stage 1 finds nothing, the pipeline falls back to OpenCV's `haarcascade_eye.xml`, then to the whole frame. If stage 2 finds nothing inside the strip, the frame is reported as normal — which is one of the known problems listed below.

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

Stage 2 returns boxes with a class name and a confidence. If any box's class name contains `JULING` or `STRABISMUS`, the frame is labelled strabismus and the highest such confidence is shown. Otherwise the frame is labelled normal.

That is a threshold on a single model's box confidence, not a calibrated probability, and there is no abstention. A frame the model finds confusing is reported as a confident answer.

## Known limitations

These are real and known. They are listed because a screening tool that hides them is worse than one that admits them.

- **The normal path can report 100% confidence without measuring anything.** When stage 2 returns no boxes, the fallback assigns the label `NORMAL` with `conf = 1.0`, which the UI renders as "normal (100%)". The number is a placeholder, not a model output. `app.py` around the `Fallback jika model_juling tidak mendeteksi apa-apa` comment and the same block in `main.py`.
- **A blank strip reads as normal.** Failure of stage 2 is treated as absence of strabismus, so a detection failure and a genuine negative look identical to the user.
- **No calibration, no abstention, no "inconclusive".** The only outcomes are normal and strabismus. Compare with [LeafSentry AI](https://github.com/sadisad/leafsentry-ai), which is built the other way round.
- **No recorded accuracy.** No evaluation set, no confusion matrix, and no held-out split ships with this repository. No number in this README would be honest, so none is given.
- **Training is not reproducible from here.** `models/` contains three weight files (about 35 MB total) but no training script, no dataset manifest, and no split definitions.
- **One face, front-facing.** The ROI gate rejects multiple faces; profile views and occlusions are untested.
- **No tests and no CI.** There is nothing in the repository that fails when the pipeline regresses.
- **Not a replacement for a cover test.** Strabismus screening in practice involves more than one photograph at more than one gaze angle.

## What this project is useful for

Reading it as a worked example of the gap between a model that answers and a pipeline that behaves. The fix list above is the interesting part: those are the same defects that show up in production screening systems, in a codebase small enough to read in one sitting.

## Stack

Python · Ultralytics YOLO · OpenCV (Haar cascades, drawing) · Streamlit · Tkinter · NumPy · Pillow

## License

No license file yet, which means all rights are reserved by default. Open an issue if you want to use this.

## Bahasa

Antarmuka aplikasi berbahasa Indonesia. Wajah harus berada di area kotak kuning dan menghadap lurus ke kamera. Hasil hanya untuk pembelajaran, bukan diagnosis.
