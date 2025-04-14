# reducto/RolmOCR · HF Mirror

原文链接: https://hf-mirror.com/reducto/RolmOCR


# [reducto](/reducto) / [RolmOCR](/reducto/RolmOCR) like 330 Follow Reducto 88

[Image-Text-to-Text](/models?pipeline_tag=image-text-to-text)[Transformers](/models?library=transformers)[Safetensors](/models?library=safetensors)

allenai/olmOCR-mix-0225


[qwen2\_5\_vl](/models?other=qwen2_5_vl)[conversational](/models?other=conversational)[text-generation-inference](/models?other=text-generation-inference)

License:
apache-2.0



[Model card](/reducto/RolmOCR)[Files
Files and versions](/reducto/RolmOCR/tree/main)[Community
7](/reducto/RolmOCR/discussions)



Train



Deploy



Use this model





# RolmOCR by [Reducto AI](https://reducto.ai/)

Earlier this year, the [Allen Institute for AI](https://allenai.org/) released olmOCR, an open-source tool that performs document OCR using the Qwen2-VL-7B vision language model (VLM). We were excited to see a high-quality, openly available approach to parsing PDFs and other complex documents — and curious to explore what else might be possible using newer foundation models and some lightweight optimizations.

The result is **RolmOCR**, a drop-in alternative to olmOCR that’s faster, uses less memory, and still performs well on a variety of document types. We're releasing it under **Apache 2.0** for anyone to try out, explore, or build on.

This model is a fine-tuned version of [Qwen/Qwen2.5-VL-7B-Instruct](https://hf-mirror.com/Qwen/Qwen2.5-VL-7B-Instruct) on the full [allenai/olmOCR-mix-0225](https://hf-mirror.com/datasets/allenai/olmOCR-mix-0225) dataset.

## Key changes

We made three notable changes:

1. **New Base Model**: We swapped in a more recent version of the existing model (Qwen2.5-VL-7B) as the foundation.
2. **No Metadata inputs**: Unlike the original, we don’t use metadata extracted from PDFs. This significantly reduces prompt length, which in turn lowers both processing time and VRAM usage — without hurting accuracy in most cases.
3. **Rotation of training data:** About 15% of the training data was rotated to enhance robustness to off-angle documents. We otherwise use the same training set.

## Usage

Host your model with vLLM:

```
export VLLM_USE_V1=1
vllm serve reducto/RolmOCR 

```

Call the model via openai compatible server:

```
# HOST YOUR OPENAI COMPATIBLE API WITH THE FOLLOWING COMMAND in VLLM:
# export VLLM_USE_V1=1
# vllm serve reducto/RolmOCR 

from openai import OpenAI
import base64

client = OpenAI(api_key="123", base_url="http://localhost:8000/v1")

model = "reducto/RolmOCR-7b"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ocr_page_with_rolm(img_base64):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "Return the plain text representation of this document as if you were reading it naturally.\n",
                    },
                ],
            }
        ],
        temperature=0.2,
        max_tokens=4096
    )
    return response.choices[0].message.content

test_img_path = "path/to/image.png"
img_base64 = encode_image(test_img_path)
print(ocr_page_with_rolm(img_base64))

```
## Limitations

* RolmOCR, like other VLM-based OCR solutions, still suffer from hallucination or dropping contents.
* Unlike the [Reducto Parsing API](https://app.reducto.ai/), RolmOCR cannot output layout bounding boxes.
* We have not evaluated the performance of any quantized versions.

## BibTex and citation info

```
@misc{RolmOCR,
  author = {Reducto AI},
  title = {RolmOCR: A Faster, Lighter Open Source OCR Model},
  year = {2025},
}

```


Downloads last month5,439









Safetensors
Model size
8.29B params
Tensor type
BF16
·

Chat template



Files info





Inference Providers
[NEW](https://hf-mirror.com/blog/inference-providers)


[Image-Text-to-Text](/tasks/image-text-to-text "Learn more about image-text-to-text")

This model isn't deployed by any Inference Provider.
[🙋
14
Ask for provider support](/spaces/huggingface/InferenceSupport/discussions/397)

## Model tree for reducto/RolmOCR

Base model


[Qwen/Qwen2.5-VL-7B-Instruct](/Qwen/Qwen2.5-VL-7B-Instruct)

Finetuned
([117](/models?other=base_model:finetune:Qwen/Qwen2.5-VL-7B-Instruct))


this model


Quantizations

[2 models](/models?other=base_model:quantized:reducto/RolmOCR)

## Dataset used to train reducto/RolmOCR


