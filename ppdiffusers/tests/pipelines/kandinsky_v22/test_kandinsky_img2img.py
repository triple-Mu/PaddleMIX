# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import random
import unittest

import numpy as np
import paddle
from PIL import Image

from ppdiffusers import (
    DDIMScheduler,
    KandinskyV22Img2ImgPipeline,
    KandinskyV22PriorPipeline,
    UNet2DConditionModel,
    VQModel,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, slow
from ppdiffusers.utils.testing_utils import enable_full_determinism, require_paddle_gpu

from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference

enable_full_determinism()


class Dummies:
    @property
    def text_embedder_hidden_size(self):
        return 32

    @property
    def time_input_dim(self):
        return 32

    @property
    def block_out_channels_0(self):
        return self.time_input_dim

    @property
    def time_embed_dim(self):
        return self.time_input_dim * 4

    @property
    def cross_attention_dim(self):
        return 32

    @property
    def dummy_unet(self):
        paddle.seed(seed=0)
        model_kwargs = {
            "in_channels": 4,
            "out_channels": 8,
            "addition_embed_type": "image",
            "down_block_types": ("ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
            "up_block_types": ("SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
            "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
            "block_out_channels": (self.block_out_channels_0, self.block_out_channels_0 * 2),
            "layers_per_block": 1,
            "encoder_hid_dim": self.text_embedder_hidden_size,
            "encoder_hid_dim_type": "image_proj",
            "cross_attention_dim": self.cross_attention_dim,
            "attention_head_dim": 4,
            "resnet_time_scale_shift": "scale_shift",
            "class_embed_type": None,
        }
        model = UNet2DConditionModel(**model_kwargs)
        return model

    @property
    def dummy_movq_kwargs(self):
        return {
            "block_out_channels": [32, 64],
            "down_block_types": ["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
            "in_channels": 3,
            "latent_channels": 4,
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "norm_type": "spatial",
            "num_vq_embeddings": 12,
            "out_channels": 3,
            "up_block_types": ["AttnUpDecoderBlock2D", "UpDecoderBlock2D"],
            "vq_embed_dim": 4,
        }

    @property
    def dummy_movq(self):
        paddle.seed(seed=0)
        model = VQModel(**self.dummy_movq_kwargs)
        return model

    def get_dummy_components(self):
        unet = self.dummy_unet
        movq = self.dummy_movq
        ddim_config = {
            "num_train_timesteps": 1000,
            "beta_schedule": "linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "clip_sample": False,
            "set_alpha_to_one": False,
            "steps_offset": 0,
            "prediction_type": "epsilon",
            "thresholding": False,
        }
        scheduler = DDIMScheduler(**ddim_config)
        components = {"unet": unet, "scheduler": scheduler, "movq": movq}
        return components

    def get_dummy_inputs(self, seed=0):
        image_embeds = floats_tensor((1, self.text_embedder_hidden_size), rng=random.Random(seed))
        negative_image_embeds = floats_tensor((1, self.text_embedder_hidden_size), rng=random.Random(seed + 1))
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed))
        image = image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((256, 256))
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "image": init_image,
            "image_embeds": image_embeds,
            "negative_image_embeds": negative_image_embeds,
            "generator": generator,
            "height": 64,
            "width": 64,
            "num_inference_steps": 10,
            "guidance_scale": 7.0,
            "strength": 0.2,
            "output_type": "np",
        }
        return inputs


class KandinskyV22Img2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22Img2ImgPipeline
    params = ["image_embeds", "negative_image_embeds", "image"]
    batch_params = ["image_embeds", "negative_image_embeds", "image"]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "strength",
        "guidance_scale",
        "num_inference_steps",
        "return_dict",
        "guidance_scale",
        "num_images_per_prompt",
        "output_type",
        "return_dict",
    ]

    def get_dummy_components(self):
        dummies = Dummies()
        return dummies.get_dummy_components()

    def get_dummy_inputs(self, seed=0):
        dummies = Dummies()
        return dummies.get_dummy_inputs(seed=seed)

    def test_kandinsky_img2img(self):

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        pipe.set_progress_bar_config(disable=None)
        output = pipe(**self.get_dummy_inputs())
        image = output.images
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]
        image_slice = image[(0), -3:, -3:, (-1)]
        image_from_tuple_slice = image_from_tuple[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5712, 0.5443, 0.4725, 0.6195, 0.5184, 0.4651, 0.4473, 0.459, 0.5016])
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"


@slow
@require_paddle_gpu
class KandinskyV22Img2ImgPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_kandinsky_img2img(self):
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/kandinskyv22_img2img_frog.npy"
        )
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
        )
        prompt = "A red cartoon frog, 4k"
        pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", paddle_dtype="float16"
        )

        pipeline = KandinskyV22Img2ImgPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", paddle_dtype="float16"
        )

        pipeline.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
        image_emb, zero_image_emb = pipe_prior(
            prompt, generator=generator, num_inference_steps=5, negative_prompt=""
        ).to_tuple()
        output = pipeline(
            image=init_image,
            image_embeds=image_emb,
            negative_image_embeds=zero_image_emb,
            generator=generator,
            num_inference_steps=100,
            height=768,
            width=768,
            strength=0.2,
            output_type="np",
        )
        image = output.images[0]
        assert image.shape == (768, 768, 3)
        assert_mean_pixel_difference(image, expected_image)
