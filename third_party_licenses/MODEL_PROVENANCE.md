# RapidOCR model provenance

The `rapidocr==3.9.2` universal wheel contains the three default CPU ONNX
models used by this solution. RapidOCR states that the OCR model copyright is
held by Baidu and releases the project under Apache License 2.0. The upstream
RapidAI/RapidOCR model repository and PaddleOCR are also published under
Apache License 2.0. Their license texts are retained beside this file.

| Wheel member | SHA-256 | Bytes | Upstream URL |
| --- | --- | ---: | --- |
| `rapidocr/models/PP-OCRv6_det_small.onnx` | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` | 9,929,594 | `https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx` |
| `rapidocr/models/PP-OCRv6_rec_small.onnx` | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` | 21,234,383 | `https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx` |
| `rapidocr/models/ch_ppocr_mobile_v2.0_cls_mobile.onnx` | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` | 585,532 | `https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx` |

The model URLs and expected hashes are embedded in the wheel's
`rapidocr/default_models.yaml`. The recognition model embeds its 18,708-entry
character dictionary in ONNX metadata, so no separate dictionary download is
needed at runtime. The models have not been modified by this solution.

Attribution and redistribution notes:

- RapidOCR: Copyright (c) 2021 RapidOCR Authors / RapidAI.
- OCR model copyright: Baidu; models converted from PaddleOCR releases.
- PaddleOCR: Copyright (c) 2016 PaddlePaddle Authors.
- Retain this provenance, both adjacent license files, and applicable upstream
  copyright, patent, trademark, and NOTICE terms when distributing the image.
