# Anti-spoof model provenance

- File: `antispoof_minifasnet.onnx`
- Architecture: MiniFASNetV2, input `1x3x80x80`, BGR values in the `0..255` range.
- Output: two class logits; class index `1` is `real`, class index `0` is `spoof`.
- Source: [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing)
- Model release: [MiniFASNetV2.onnx](https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx)
- SHA-256: `b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907`
- License: Apache-2.0. The source repository describes its pretrained model as evaluated on CelebA-Spoof; test it with this project's webcams and phone-display attacks before relying on it operationally.

`model_assets.py` downloads this pinned release when the file is missing or its
checksum does not match, then verifies and installs it atomically. Desktop
model warm-up and backend startup both trigger the download. Network or checksum
errors keep anti-spoofing fail-closed.
