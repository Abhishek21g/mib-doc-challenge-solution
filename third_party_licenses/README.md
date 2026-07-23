# Third-party notices

This directory adds the license and model provenance files that are not
present in the `rapidocr==3.9.2` wheel itself. The Docker image copies this
directory to `/app/third_party_licenses`.

The remaining pinned Python wheels retain their own license and NOTICE files
inside the installed package tree. In particular, redistribution must retain:

- ONNX Runtime `onnxruntime/LICENSE` and `onnxruntime/ThirdPartyNotices.txt`;
- OpenCV headless `cv2/LICENSE.txt` and `cv2/LICENSE-3RD-PARTY.txt`;
- Requests' `.dist-info/licenses/LICENSE` and `NOTICE`;
- Shapely's `.dist-info/licenses/LICENSE.txt` and `LICENSE_GEOS`;
- every other installed `.dist-info/LICENSE*`, `.dist-info/licenses/`, and
  package-level license or notice file.

The headless OpenCV wheel includes FFmpeg components under LGPL 2.1, and the
Shapely wheel bundles GEOS under LGPL 2.1. Corresponding notices and relinking
or source obligations in the wheel files remain applicable. Upstream source is
available from:

- `https://github.com/opencv/opencv-python`
- `https://github.com/FFmpeg/FFmpeg`
- `https://github.com/shapely/shapely`
- `https://github.com/libgeos/geos`

Other pinned package license identifiers are recorded in their wheel metadata:
MPL-2.0 (certifi), MIT (charset-normalizer, colorlog, requests, pyclipper,
PyYAML, six, urllib3), BSD-family (idna, OmegaConf, protobuf, Shapely),
Apache-2.0 (FlatBuffers, ONNX Runtime, OpenCV, packaging, RapidOCR), PSF-2.0
(typing_extensions), and the license bundles shipped by NumPy, Pillow, tqdm,
and packaging.
