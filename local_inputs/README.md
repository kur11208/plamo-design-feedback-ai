# Local Inputs

Put local-only runner photos, prototype photos, or rights-managed images here when testing the app on your own machine.

Files in this directory are ignored by Git. Do not use this folder for public demo assets, README screenshots, or Streamlit Cloud sample data.

Enable the uploader only in a Streamlit process running on your own machine:

```powershell
$env:PLAMO_ENABLE_IMAGE_UPLOAD="true"
streamlit run app.py
```

Uploads are limited to PNG, JPEG, or WebP files up to 10 MB and 20 million pixels. The image bytes are sent to the running Streamlit process, so never enable this feature on a public server for confidential images.

Recommended inputs:

- A single runner or prototype part photographed from above.
- No sticker sheet, instruction sheet, package, logo, or unrelated object covering the runner.
- Minimal plastic bag glare and strong reflections.
- Crop the image so the target runner fills most of the frame.
- If the original photo contains extra objects, check the app's automatic ROI suggestion and adjust the sliders to analyze only the target runner area.

Images with stickers, printed labels, multiple mixed runners, or heavy glare can still be uploaded for local experiments, but the lightweight image heuristic may read those visual edges as false gate or thin-part candidates.
