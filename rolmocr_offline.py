# rolmocr_offline.py

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, AutoImageProcessor, GenerationConfig
from PIL import Image
import gradio as gr
import logging
from typing import Union, Dict, List, Optional
import os
from pathlib import Path
import time
import socket
import subprocess
import argparse
from tqdm import tqdm  # 添加进度条支持
import threading
import signal
import sys
import math
import gc
import json


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # 强制重新配置日志
)
logger = logging.getLogger(__name__)


class RolmOCR:
    def __init__(self, 
                 model_path: str = "Qwen/Qwen2.5-VL",
                 use_half: bool = True,
                 max_new_tokens: int = 512,
                 temperature: float = 0.2,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 default_prompt: str = "请识别这张图片中的所有文字内容。"):
        """
        初始化 RolmOCR
        
        Args:
            model_path: 模型路径或huggingface模型名
            use_half: 是否使用半精度(float16)
            max_new_tokens: 生成的最大token数
            temperature: 采样温度
            top_p: 采样的累积概率阈值
            do_sample: 是否使用采样
            default_prompt: 默认的提示词
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.default_prompt = default_prompt
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_path, use_half)
        self._setup_memory_tracking()

    def _load_model(self, model_path: str, use_half: bool) -> None:
        """加载模型和处理器"""
        try:
            logger.info("开始加载模型...")
            
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if torch.cuda.is_available() else None,
                "torch_dtype": torch.float16 if use_half and torch.cuda.is_available() else torch.float32,
            }
            
            # 添加版本检查
            import transformers
            logger.info(f"Transformers版本: {transformers.__version__}")
            
            # 尝试不同的导入方式，增加兼容性
            try:
                # 尝试直接导入Qwen2_5_VLForConditionalGeneration
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
                logger.info("成功导入Qwen2_5_VLForConditionalGeneration")
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    **model_kwargs
                )
            except ImportError:
                # 如果直接导入失败，尝试使用AutoModel
                logger.info("尝试使用AutoModelForVision2Seq加载模型...")
                from transformers import AutoProcessor, AutoModelForVision2Seq
                
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    **model_kwargs
                )
            
            # 加载处理器
            logger.info("加载处理器...")
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # 输出更多处理器信息
            logger.info(f"处理器类型: {type(self.processor).__name__}")
            if hasattr(self.processor, 'image_processor'):
                logger.info(f"图像处理器类型: {type(self.processor.image_processor).__name__}")
            if hasattr(self.processor, 'tokenizer'):
                logger.info(f"分词器类型: {type(self.processor.tokenizer).__name__}")
            
            logger.info(f"模型类型: {type(self.model).__name__}")
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            import traceback
            logger.error(f"加载失败堆栈: {traceback.format_exc()}")
            raise

    def _setup_memory_tracking(self) -> None:
        """设置内存追踪"""
        self.peak_memory = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _get_gpu_memory_usage(self) -> float:
        """获取GPU显存使用情况"""
        if not torch.cuda.is_available():
            return 0
        # 确保获取最新的内存使用情况
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated(0)/1024**2

    def _get_processed_image_data(self, image):
        """使用官方方法处理图像获取所需数据"""
        logger.info("内部方法：正确处理图像...")
        
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")
        
        try:
            # 尝试Qwen2.5-VL风格的处理
            try:
                logger.info("尝试Qwen2.5-VL风格的处理...")
                prompt = "请识别这张图片中的文字"
                
                # 准备消息
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"image": image},
                            {"text": prompt}
                        ]
                    }
                ]
                
                # 使用自定义的Qwen处理函数
                text, processed_images = process_qwen_messages(messages, self.processor)
                logger.info(f"Qwen2.5-VL风格处理成功，文本长度: {len(text)}, 图像数量: {len(processed_images)}")
                
                # 使用processor处理处理好的文本和图像
                inputs = self.processor(text=[text], images=processed_images, return_tensors="pt").to(self.device)
                logger.info(f"Qwen2.5-VL处理返回的键: {list(inputs.keys())}")
                return inputs
            except Exception as qwen_err:
                logger.error(f"Qwen2.5-VL风格处理失败: {str(qwen_err)}")
                logger.info("回退到其他处理方法...")
            
            # 其他处理方法
            # 通用处理方法
            prompt = "请识别这张图片中的文字"
            logger.info(f"使用通用处理方法，提示词: {prompt}")
            
            # 尝试多种处理方式，从简单到复杂
            methods = [
                # 方法1: 最基础的处理方式 - 单独处理文本和图像
                lambda: self._process_basic(image, prompt),
                
                # 方法2: 标准的processor处理
                lambda: self._process_standard(image, prompt),
                
                # 方法3: 使用消息格式
                lambda: self._process_with_messages(image, prompt)
            ]
            
            # 依次尝试各种方法
            last_error = None
            for i, method in enumerate(methods):
                try:
                    logger.info(f"尝试处理方法 {i+1}/{len(methods)}...")
                    return method()
                except Exception as e:
                    logger.error(f"方法 {i+1} 失败: {str(e)}")
                    last_error = e
            
            # 如果所有方法都失败，抛出最后一个错误
            if last_error:
                raise last_error
            return None
                
        except Exception as e:
            logger.error(f"所有图像处理方法均失败: {str(e)}")
            import traceback
            logger.error(f"图像处理错误堆栈: {traceback.format_exc()}")
            return None
    
    def _process_basic(self, image, prompt):
        """最基础的处理方式 - 单独处理文本和图像"""
        logger.info("使用基础处理方式...")
        
        # 使用图像处理器单独处理图像
        try:
            image_processor = getattr(self.processor, 'image_processor', None)
            if image_processor is None:
                # 尝试查找其他可能的图像处理器属性名
                for attr in dir(self.processor):
                    if 'image' in attr.lower() and callable(getattr(self.processor, attr)):
                        image_processor = getattr(self.processor, attr)
                        logger.info(f"找到备选图像处理器: {attr}")
                        break
            
            if image_processor is None:
                # 如果仍找不到，直接用processor处理图像
                logger.info("使用processor直接处理图像")
                image_inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = image_inputs.get('pixel_values', image_inputs.get('visual_feats'))
            else:
                logger.info(f"使用图像处理器: {type(image_processor).__name__}")
                image_inputs = image_processor(image, return_tensors="pt")
                pixel_values = image_inputs.get('pixel_values', image_inputs.get('visual_feats'))
            
            # 使用分词器处理文本
            tokenizer = getattr(self.processor, 'tokenizer', None) 
            if tokenizer is None:
                logger.info("使用processor作为分词器")
                text_inputs = self.processor(text=prompt, return_tensors="pt")
                input_ids = text_inputs.get('input_ids')
            else:
                logger.info(f"使用分词器: {type(tokenizer).__name__}")
                text_inputs = tokenizer(prompt, return_tensors="pt")  
                input_ids = text_inputs.get('input_ids')
            
            # 确保我们有必要的键
            if pixel_values is None:
                raise ValueError("无法获取pixel_values")
            if input_ids is None:
                raise ValueError("无法获取input_ids")
                
            # 移动到正确的设备
            pixel_values = pixel_values.to(self.device) 
            input_ids = input_ids.to(self.device)
            
            # 创建输入字典
            inputs = {
                "pixel_values": pixel_values,
                "input_ids": input_ids
            }
            
            # 可选：添加attention_mask如果存在
            if 'attention_mask' in text_inputs:
                inputs['attention_mask'] = text_inputs['attention_mask'].to(self.device)
                
            logger.info(f"基础处理返回的键: {list(inputs.keys())}")
            return inputs
        except Exception as e:
            logger.error(f"基础处理失败: {str(e)}")
            raise
    
    def _process_standard(self, image, prompt):
        """使用标准处理方式"""
        logger.info("使用标准处理方式...")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        logger.info(f"标准处理返回的键: {list(inputs.keys())}")
        return inputs
    
    def _process_with_messages(self, image, prompt):
        """使用消息格式处理"""
        logger.info("使用消息格式处理...")
        
        # 尝试两种消息格式
        try:
            # 格式1
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            inputs = self.processor(messages=messages, return_tensors="pt").to(self.device)
        except Exception as e:
            logger.error(f"消息格式1失败: {str(e)}")
            
            # 格式2
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image},
                        {"text": prompt}
                    ]
                }
            ]
            inputs = self.processor(messages=messages, return_tensors="pt").to(self.device)
        
        logger.info(f"消息格式处理返回的键: {list(inputs.keys())}")
        return inputs

    def process_image(self, img: Image.Image, prompt=None, temperature=None, top_p=None, do_sample=None, max_new_tokens=None) -> str:
        """处理图像并获取识别结果"""
        try:
            print(f"开始处理图像: {img.size}...")
            # 检查图像模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print(f"已将图像转换为RGB模式")
            
            # 使用特殊方法处理Qwen模型
            if "Qwen" in type(self.model).__name__:
                print(f"检测到Qwen模型，使用特殊处理方法...")
                result = self.special_process_for_qwen(img, prompt, temperature, top_p, do_sample, max_new_tokens)
                return result
            
            # 使用_get_processed_image_data方法获取处理后的图像数据
            inputs = self._get_processed_image_data(img)
            if inputs is None:
                return "处理图像失败，无法生成模型输入"
            
            print(f"模型输入准备完成，开始生成...")
            # 使用生成方法处理输入并获取结果
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=do_sample if do_sample is not None else self.do_sample,
                temperature=temperature or self.temperature,
                top_p=top_p or self.top_p
            )
            
            logger.info(f"生成配置: max_new_tokens={generation_config.max_new_tokens}, "
                      f"temperature={generation_config.temperature}, "
                      f"top_p={generation_config.top_p}, "
                      f"do_sample={generation_config.do_sample}")
            
            with torch.no_grad():
                # 检查输入键是否存在input_ids
                if "input_ids" not in inputs:
                    logger.error(f"输入数据不包含input_ids键，当前键: {list(inputs.keys())}")
                    return "处理图像失败，输入数据格式不匹配"
                
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # 解码模型输出
            try:
                # 尝试使用processor.decode
                response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            except Exception as decode_err:
                logger.error(f"使用processor.batch_decode解码失败: {str(decode_err)}")
                logger.info("尝试使用tokenizer.decode...")
                # 备用解码方法
                response = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"模型生成结果: {response[:100]}...")
            logger.info(f"生成结果完成，长度: {len(response)}")
            
            # 记录内存使用情况
            if torch.cuda.is_available():
                current_memory = self._get_gpu_memory_usage()
                logger.info(f"当前显存使用: {current_memory:.2f} MB")
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                    logger.info(f"新的峰值显存: {self.peak_memory:.2f} MB")
            
            return response
        
        except Exception as e:
            error_msg = f"图像处理过程中出错: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
            logger.error(traceback.format_exc())
            return f"错误: {str(e)}"

    def special_process_for_qwen(self, img: Image.Image, prompt=None, temperature=None, top_p=None, do_sample=None, max_new_tokens=None) -> str:
        """专门针对Qwen2.5-VL模型的处理方法"""
        try:
            logger.info("使用Qwen2.5-VL专用处理方法...")
            if prompt is None:
                prompt = self.default_prompt
                
            # 严格按照官方README中的示例来处理
            logger.info("使用官方示例方式处理图像...")
            
            # 1. 准备消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},  # 直接传递PIL图像
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 2. 准备文本 - 使用官方示例的方式
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            logger.info(f"应用聊天模板后文本长度: {len(text)}")
            
            # 3. 处理图像 - 从消息中提取
            # 此处模仿process_vision_info函数
            image_inputs = []
            
            # 确保图像是RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # 调整图像大小 - 确保符合模型要求
            width, height = img.size
            # 使用smart_resize来获取合适的尺寸
            min_pixels = 4 * 28 * 28  # 最小像素数
            max_pixels = 1280 * 28 * 28  # 最大像素数
            resized_height, resized_width = smart_resize(
                height, width, 
                factor=28,  # 调整因子 
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            processed_img = img.resize((resized_width, resized_height))
            image_inputs.append(processed_img)
                        
            logger.info(f"处理后图像尺寸: {processed_img.size}")
            
            # 4. 使用处理器处理文本和图像
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # 5. 移动到设备上
            inputs = inputs.to(self.device)
            logger.info(f"输入数据键: {list(inputs.keys())}")
            
            # 6. 生成配置
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=do_sample if do_sample is not None else self.do_sample,
                temperature=temperature or self.temperature,
                top_p=top_p or self.top_p
            )
            
            logger.info(f"生成配置: max_new_tokens={generation_config.max_new_tokens}")
            
            # 7. 进行生成
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, generation_config=generation_config)
            
            # 8. 解码模型输出，仅保留新生成的部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # 9. 使用处理器解码结果
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
            
            logger.info(f"生成完成，结果长度: {len(response)}")
            return response
            
        except Exception as e:
            logger.error(f"Qwen专用处理失败: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return f"Qwen专用处理失败: {str(e)}"

    def process_batch(self, images: List[Union[str, Image.Image]], batch_size: int = 4) -> Dict[str, str]:
        """批量处理图片"""
        print(f"\n=== 开始批量处理 {len(images)} 张图片，批次大小: {batch_size} ===")  # 添加print
        logger.info(f"\n=== 开始批量处理 {len(images)} 张图片，批次大小: {batch_size} ===")
        
        results = {}
        for i in tqdm(range(0, len(images), batch_size), desc="处理图片"):
            batch = images[i:i + batch_size]
            print(f"处理批次 {i//batch_size + 1}/{math.ceil(len(images)/batch_size)}，包含 {len(batch)} 张图片")  # 添加print
            logger.info(f"处理批次 {i//batch_size + 1}/{math.ceil(len(images)/batch_size)}，包含 {len(batch)} 张图片")
            
            for j, img in enumerate(batch):
                if isinstance(img, str) or isinstance(img, Path):
                    print(f"  处理图片 {j+1}/{len(batch)}: {img}")  # 添加print
                    logger.info(f"  处理图片 {j+1}/{len(batch)}: {img}")
                else:
                    print(f"  处理图片 {j+1}/{len(batch)}: <内存中的图片>")  # 添加print
                    logger.info(f"  处理图片 {j+1}/{len(batch)}: <内存中的图片>")
                
                try:
                    key = str(img) if isinstance(img, Path) else img
                    results[key] = self.process_image(img)
                    print(f"  ✅ 图片 {j+1}/{len(batch)} 处理完成")  # 添加print
                    logger.info(f"  ✅ 图片 {j+1}/{len(batch)} 处理完成")
                except Exception as e:
                    print(f"  ❌ 图片 {j+1}/{len(batch)} 处理失败: {str(e)}")  # 添加print
                    logger.error(f"  ❌ 图片 {j+1}/{len(batch)} 处理失败: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    results[key] = f"处理失败: {str(e)}"
            
            print(f"批次 {i//batch_size + 1} 处理完成，清理GPU内存...")  # 添加print
            logger.info(f"批次 {i//batch_size + 1} 处理完成，清理GPU内存...")
            self.clear_gpu_memory()
        
        print(f"\n=== 批量处理完成，共处理 {len(images)} 张图片 ===")  # 添加print
        logger.info(f"\n=== 批量处理完成，共处理 {len(images)} 张图片 ===")
        return results

    def clear_gpu_memory(self) -> None:
        """清理GPU显存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"清理后显存: {self._get_gpu_memory_usage():.2f} MB")

    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存使用统计"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "current_memory": self._get_gpu_memory_usage(),
            "peak_memory": self.peak_memory
        }

    def safe_process_image(self, image: Union[str, Image.Image, Path], max_retries: int = 3) -> str:
        """带有重试机制的图片处理"""
        for attempt in range(max_retries):
            try:
                return self.process_image(image)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"处理失败，已重试{max_retries}次: {str(e)}")
                    return f"处理失败: {str(e)}"
                logger.warning(f"处理失败，正在重试 ({attempt + 1}/{max_retries}): {str(e)}")
                self.clear_gpu_memory()
                time.sleep(1)  # 等待一秒后重试

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """图片预处理"""
        try:
            logger.info(f"开始图像预处理，原始尺寸: {image.size}, 模式: {image.mode}")
            
            # 调整大小（如果太大）
            max_size = 2048
            if max(image.size) > max_size:
                logger.info(f"图像尺寸过大，需要调整大小，最大边长: {max(image.size)} > {max_size}")
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                logger.info(f"调整尺寸: {image.size} -> {new_size}")
                try:
                    image = image.resize(new_size, Image.LANCZOS)
                    logger.info(f"尺寸调整完成")
                except Exception as resize_err:
                    logger.error(f"尺寸调整失败: {str(resize_err)}")
                    raise
            else:
                logger.info(f"图像尺寸在允许范围内，无需调整")
            
            # 基本优化
            try:
                logger.info("开始图像增强处理...")
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                orig_image = image  # 保存原始图像以便比较
                image = enhancer.enhance(1.2)
                logger.info(f"对比度增强完成 (增强系数: 1.2)")
                
                # 可以添加其他图像增强步骤
                # ...
                
                logger.info(f"图像预处理完成，最终尺寸: {image.size}")
                return image
            except Exception as enhance_err:
                logger.error(f"图像增强处理失败: {str(enhance_err)}")
                # 如果增强失败，返回原始图像
                logger.warning("图像增强失败，将使用原始图像")
                return orig_image if 'orig_image' in locals() else image
                
        except Exception as e:
            logger.error(f"图像预处理过程中出错: {str(e)}")
            # 如果预处理失败，返回原始图像
            logger.warning("预处理失败，将使用原始图像")
            return image

# 全局变量
ocr = None

def process_image_interface(image, prompt, temperature, top_p, do_sample, max_new_tokens):
    print("\n=== 接收到新的处理请求 ===")  # 添加print
    logger.info("\n=== 接收到新的处理请求 ===")
    
    if image is None:
        print("❌ 错误：未接收到图片")  # 添加print
        logger.info("❌ 错误：未接收到图片")
        return "请上传图片", "等待上传图片..."
    
    try:
        print("✅ 成功接收图片")  # 添加print
        print(f"图片信息: {type(image)}")  # 添加print
        logger.info("✅ 成功接收图片")
        logger.info(f"图片信息:")
        logger.info(f"- 类型: {type(image)}")
        if isinstance(image, Image.Image):
            print(f"- 尺寸: {image.size}")  # 添加print
            logger.info(f"- 尺寸: {image.size}")
            logger.info(f"- 模式: {image.mode}")
        
        # 确保图片是RGB模式
        if isinstance(image, Image.Image) and image.mode != "RGB":
            print(f"转换图片模式从 {image.mode} 到 RGB")  # 添加print
            logger.info(f"转换图片模式从 {image.mode} 到 RGB")
            image = image.convert("RGB")
        
        print("开始处理图片...")  # 添加print
        logger.info("开始处理图片...")
        logger.info(f"参数配置:")
        logger.info(f"- 提示词: {prompt}")
        logger.info(f"- 温度: {temperature}")
        logger.info(f"- top_p: {top_p}")
        logger.info(f"- 采样: {do_sample}")
        logger.info(f"- 最大token数: {max_new_tokens}")
        
        result = ocr.process_image(
            image,
            prompt=prompt,
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=do_sample,
            max_new_tokens=int(max_new_tokens)
        )
        
        print("✅ 处理完成")  # 添加print
        logger.info("✅ 处理完成")
        logger.info(f"结果长度: {len(result)}")
        logger.info(f"结果预览: {result[:100]}...")
        return result, "处理完成"
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        print(f"❌ {error_msg}")  # 添加print
        logger.error(f"❌ {error_msg}")
        import traceback
        logger.error(f"错误详情:\n{traceback.format_exc()}")
        return error_msg, "处理出错"

def on_image_upload(image):
    if image is not None:
        print("✅ 图片上传成功")  # 添加print
        logger.info("✅ 图片上传成功")
        return "图片已上传，请点击开始识别"
    return "等待上传图片..."

def create_gradio_interface(model_path="Qwen/Qwen2.5-VL"):
    logger.info(f"\n=== 创建Gradio界面 ===")
    logger.info(f"使用模型: {model_path}")
    
    global ocr
    ocr = RolmOCR(model_path=model_path)
    logger.info("✅ RolmOCR实例创建完成")

    with gr.Blocks() as iface:
        gr.Markdown("# RolmOCR 文字识别系统")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 修改图片组件的配置，使用更基础的参数
                input_image = gr.Image(
                    type="pil",
                    label="上传图片",
                    height=400
                )
                
                with gr.Accordion("高级设置", open=False):
                    prompt = gr.Textbox(
                        value=ocr.default_prompt,
                        label="提示词",
                        lines=2
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=ocr.temperature,
                        step=0.1,
                        label="温度 (Temperature)"
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=ocr.top_p,
                        step=0.05,
                        label="Top P"
                    )
                    do_sample = gr.Checkbox(
                        value=ocr.do_sample,
                        label="使用采样"
                    )
                    max_new_tokens = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=ocr.max_new_tokens,
                        step=64,
                        label="最大生成Token数"
                    )
                
                submit_btn = gr.Button("开始识别", variant="primary")
                
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="识别结果",
                    lines=10,
                    show_label=True
                )
                status_text = gr.Textbox(
                    label="状态",
                    lines=1,
                    value="等待上传图片..."
                )

        # 添加图片上传事件处理
        input_image.change(
            fn=on_image_upload,
            inputs=[input_image],
            outputs=[status_text]
        )

        # 绑定识别按钮事件
        submit_btn.click(
            fn=process_image_interface,
            inputs=[input_image, prompt, temperature, top_p, do_sample, max_new_tokens],
            outputs=[output_text, status_text]
        )

    return iface

def check_network_config():
    """检查网络配置"""
    try:
        # 检查是否能解析主机名
        socket.gethostbyname(socket.gethostname())
        return True
    except:
        return False

def get_available_port(start_port=12319):
    """获取可用端口"""
    port = start_port
    while port < 65535:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except:
            port += 1
    raise RuntimeError("没有可用的端口")

def install_dependencies():
    """安装必要的依赖"""
    try:
        import nest_asyncio
    except ImportError:
        logger.info("安装必要的依赖：nest_asyncio")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio"])
        import nest_asyncio

def cli_interface(model_path="reducto/RolmOCR"):
    """命令行界面"""
    parser = argparse.ArgumentParser(description='RolmOCR 文字识别工具')
    parser.add_argument('image', help='图片路径')
    parser.add_argument('--batch', help='批量处理模式，提供包含图片路径的文件', type=str)
    parser.add_argument('--output', help='输出结果到文件', type=str)
    parser.add_argument('--verbose', help='显示详细信息', action='store_true')
    
    args = parser.parse_args()
    
    ocr = RolmOCR(model_path=model_path)
    
    if args.batch:
        with open(args.batch) as f:
            image_paths = f.read().splitlines()
        results = ocr.process_batch(image_paths)
        
        if args.output:
            with open(args.output, 'w') as f:
                for path, text in results.items():
                    f.write(f"=== {path} ===\n{text}\n\n")
        else:
            for path, text in results.items():
                print(f"=== {path} ===\n{text}\n")
    else:
        result = ocr.process_image(args.image)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result)
        else:
            print(result)
            
    if args.verbose:
        stats = ocr.get_memory_stats()
        print(f"\n性能统计:")
        print(f"当前显存使用: {stats['current_memory']:.2f} MB")
        print(f"峰值显存使用: {stats['peak_memory']:.2f} MB")

# 添加Qwen2.5-VL图像处理工具函数
def to_rgb(pil_image: Image.Image) -> Image.Image:
    """转换图像到RGB模式"""
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # 使用alpha通道作为蒙版
        return white_background
    else:
        return pil_image.convert("RGB")

def round_by_factor(number: int, factor: int) -> int:
    """将数字四舍五入到最接近的factor的倍数"""
    return round(number / factor) * factor

def smart_resize(height, width, factor=28, min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28):
    """智能调整图像大小，保持纵横比"""
    max_ratio = 200
    
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width should be positive, but got {height=}, {width=}")
    
    total_pixels = height * width
    ratio = max(height / width, width / height)
    
    if ratio >= max_ratio:
        # 处理极端高宽比
        if height > width:
            height, width = round_by_factor(max_ratio * width, factor), round_by_factor(width, factor)
        else:
            height, width = round_by_factor(height, factor), round_by_factor(max_ratio * height, factor)
        return height, width
    
    if total_pixels <= min_pixels:
        # 如果图像太小，放大到最小尺寸
        scale = math.sqrt(min_pixels / total_pixels)
        return round_by_factor(height * scale, factor), round_by_factor(width * scale, factor)
    
    if total_pixels > max_pixels:
        # 如果图像太大，缩小到最大尺寸
        scale = math.sqrt(max_pixels / total_pixels)
        return round_by_factor(height * scale, factor), round_by_factor(width * scale, factor)
    
    # 保持原始大小，但确保是factor的倍数
    return round_by_factor(height, factor), round_by_factor(width, factor)

def process_qwen_messages(messages, processor):
    """使用Qwen2.5-VL的方式处理消息，从消息中提取图像信息"""
    # 提取文本和图像
    images = []
    for message in messages:
        for content in message["content"]:
            if "image" in content:
                image = content["image"]
                if isinstance(image, Image.Image):
                    # 处理PIL图像
                    image = to_rgb(image)
                    # 调整大小
                    height, width = image.size[1], image.size[0]
                    resized_height, resized_width = smart_resize(height, width)
                    image = image.resize((resized_width, resized_height))
                    images.append(image)
    
    # 构建文本
    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    return text, images

if __name__ == "__main__":
    import socket
    import argparse
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("正在关闭服务...")  # 添加print
        logger.info("正在关闭服务...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 输出Python和包版本信息
    logger.info(f"\n=== 环境信息 ===")
    logger.info(f"Python版本: {sys.version}")
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("无法导入PyTorch")
    
    try:
        import transformers
        logger.info(f"Transformers版本: {transformers.__version__}")
    except ImportError:
        logger.warning("无法导入Transformers")
    
    try:
        import gradio
        logger.info(f"Gradio版本: {gradio.__version__}")
    except ImportError:
        logger.warning("无法导入Gradio")
    
    parser = argparse.ArgumentParser(description='RolmOCR 文字识别系统')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=12319, help='服务器端口')
    parser.add_argument('--share', action='store_true', help='是否创建公共链接')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-VL", help='模型路径或名称')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
        logger.setLevel(logging.DEBUG)
        logger.info("已启用调试模式")
    
    try:
        print("\n=== 启动服务 ===")
        print(f"主机: {args.host}")
        print(f"端口: {args.port}")
        print(f"模型: {args.model}")
        
        logger.info(f"\n=== 启动服务 ===")
        logger.info(f"主机: {args.host}")
        logger.info(f"端口: {args.port}")
        logger.info(f"模型: {args.model}")
        
        # 检查端口是否可用
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((args.host, args.port))
                logger.info(f"✅ 端口 {args.port} 可用")
        except Exception as e:
            logger.warning(f"指定端口 {args.port} 不可用，尝试获取其他可用端口")
            args.port = get_available_port(start_port=7860)
            logger.info(f"将使用新端口: {args.port}")
        
        iface = create_gradio_interface(model_path=args.model)
        
        # 启动服务
        try:
            logger.info("尝试使用 Gradio launch()...")
            iface.launch(
                server_name=args.host,
                server_port=args.port,
                share=args.share,
                show_error=True,
                debug=True # Keep debug True for now
            )
        except Exception as launch_err:
             logger.error(f"Gradio launch() 失败: {launch_err}")
             logger.info("请确保 Gradio 版本兼容或尝试其他启动方式 (如 uvicorn)")
             
             # 尝试使用不同的Gradio参数
             try:
                 logger.info("尝试使用备用启动参数...")
                 iface.launch(
                     server_name=args.host,
                     server_port=args.port,
                     share=args.share,
                     quiet=False
                 )
             except Exception as alt_launch_err:
                 logger.error(f"备用启动也失败: {alt_launch_err}")
             
    except Exception as e:
        error_msg = f"❌ 启动或运行失败: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        import traceback
        logger.error(f"错误详情:\n{traceback.format_exc()}")
        print("\n请尝试以下解决方案:")
        print(f"1. 检查模型路径/名称: --model {args.model}")
        print(f"2. 检查依赖版本 (transformers, torch, gradio)")
        print(f"3. 使用不同的主机/端口")
        print(f"4. 添加 --debug 参数获取更多日志信息")
        sys.exit(1)