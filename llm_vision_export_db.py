import os
import paddle

_fused_linear = paddle.incubate.nn.functional.fused_linear
paddle.incubate.nn.functional.fused_linear = paddle.nn.functional.linear

_cuda = paddle.version.cuda


def cuda_fn():
    return 11.2


paddle.version.cuda = cuda_fn

import paddle.nn as nn
import numpy as np

# dtype = paddle.float16
# paddle.set_default_dtype(dtype)
dtype = paddle.float32
paddle.set_device('gpu')

mean = paddle.to_tensor([[[0.48145466]], [[0.4578275]], [[0.40821073]]], dtype=dtype)
std = paddle.to_tensor([[[0.26862954]], [[0.26130258]], [[0.27577711]]], dtype=dtype)

root = '/mnt/data/cangshui/triplemu/mgen_vllm/baseline/paddle_weights/fft_h20_a800_stb_img'
save_root = '/mnt/data/cangshui/triplemu/mgen_vllm/acceletrate/submission/weights/fft_h20_a800_stb_img-7000'


class ExportVision(nn.Layer):
    def __init__(self, model: nn.Layer):
        super().__init__()
        self.visual = model.visual
        input_ids = paddle.to_tensor([
            151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 24669, 220, 16, 25, 220,
            151857,
            48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 51, 52, 46, 106, 112, 103, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
            151859,
            151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151858, 198, 14880, 53481,
            43288,
            99708, 45930, 151645, 198, 151644, 77091, 198], dtype='int32')

        self.mean = mean
        self.std = std
        self.part1 = model.transformer.llm.wte(input_ids[None, :20])
        self.part2 = model.transformer.llm.wte(input_ids[None, 276:])

    def forward(self, x):
        bs = paddle.shape(x)[0]
        x = x / 255.
        x = x - self.mean
        x = x / self.std
        x = x.cast(dtype)
        y = self.visual(x)
        y = paddle.concat([self.part1.tile([bs, 1, 1]), y, self.part2.tile([bs, 1, 1])], axis=1)
        y = y.cast(paddle.float32)
        return y


def export_llm():
    from paddlenlp.transformers import AutoConfig
    from paddlenlp.experimental.transformers import MGenForMGenVLInferenceModel
    paddle.set_default_dtype(paddle.float16)

    model_name_or_path = os.path.join(root, 'llm')
    config = AutoConfig.from_pretrained(model_name_or_path)

    model = MGenForMGenVLInferenceModel.from_pretrained(
        model_name_or_path,
        config=config,
        dtype='float16',
        use_safetensors=False
    )

    model.eval()

    model.to_static_v2(
        os.path.join(save_root, 'llm', 'mgen'),
        {
            "dtype": 'float16',
            "export_precache": False,
            "use_cachekv_int8": False,
        },
    )
    print(f"static model has been to {os.path.join(save_root, 'llm')}")


def export_vision():
    from paddlemix import MGenLMHeadModel
    paddle.set_default_dtype(paddle.float16)

    size = 448
    model_name_or_path = root
    model = MGenLMHeadModel.from_pretrained(model_name_or_path, dtype="float16")
    model.eval()

    model = ExportVision(model)
    model.eval()

    model.mean = mean.cast(paddle.float32)
    model.std = std.cast(paddle.float32)

    x = paddle.randn((2, 3, size, size), dtype=paddle.float32)
    y = model(x)
    print(f'{y.shape=}')

    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, 3, size, size], dtype="float32", name="images"),  # images
        ],
        full_graph=True
    )

    save_path = os.path.join(save_root, 'vision', 'vit')
    paddle.jit.save(model, save_path)
    print(f"static model has been to {save_path}")


if __name__ == "__main__":
    export_llm()
    export_vision()
