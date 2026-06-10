# custom-seed-vc

This repostitory contains a custom implementation of the [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc).

## Requirements

- Python 3.10+
- uv 0.10+
- Nvidia Driver with CUDA 11.8 support
- Nvidia GPU
- Docker (optional)

This repository is tested on the following environment:

- Ubuntu 24.04
- Python 3.10.16
- uv 0.10.6
- Nvidia Driver 570.133.20
- Nvidia RTX6000 & L40S
- Docker 28.0.4

## Setup

```bash
$ make install
```

## Usage

See the following examples for usage:

- [Realtime VC with SocketIO](./examples/realtime-vc-via-socketio)
- [Fine-tuning with Custom Dataset](./examples/fine-tuning)

## Acknowledgements

- [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc)
- [open-mmlab/Amphion](https://github.com/open-mmlab/Amphion)
- [open-mmlab/Vevo](https://github.com/open-mmlab/Amphion/tree/main/models/vc/vevo)
- [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3)
- [Plachta/ASTRAL-quantiztion](https://github.com/Plachtaa/ASTRAL-quantization)
- [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [SEED-TTS](https://arxiv.org/abs/2406.02430)
