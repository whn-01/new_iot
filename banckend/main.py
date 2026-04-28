import os
import uuid
import json
import pandas as pd
import shutil
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Body
from starlette.responses import JSONResponse 
from fastapi.middleware.cors import CORSMiddleware
from ludwig.api import LudwigModel
from pycaret.classification import setup, create_model, pull, save_model, tune_model
from ludwig.hyperopt.run import hyperopt
import logging
import glob
import sys
import traceback
import anyio
import numpy as np 
import pynvml
import subprocess
import threading
import time
import ray
import multiprocessing
from typing import Dict, Any
import re
import warnings
from functools import partial
from ray.tune.utils.util import wait_for_gpu as ray_wait_for_gpu
import ludwig.hyperopt.execution as ludwig_hyperopt_execution
import glob # 用于查找文件
from datetime import datetime
import joblib  # 用于加载PyCaret模型
import pickle
import joblib
# 将 Python 默认的 1000 层递归限制调高到 100000 层，防止 Ray 打包时堆栈溢出崩溃！
sys.setrecursionlimit(100000)

# --- 配置日志 ---
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*no more leaves that meet the split requirements.*", category=UserWarning)

def patch_ludwig_gpu_wait(target_util: float = 0.2, retry: int = 20, delay_s: int = 2):
    """
    放宽 Ludwig/Ray 的 GPU 空闲等待阈值，避免因显存常驻占用(如 8%-10%)导致 trial 直接失败。
    Ray 默认 target_util=0.01 在桌面环境常常过严。
    """
    ludwig_hyperopt_execution.wait_for_gpu = partial(
        ray_wait_for_gpu, target_util=target_util, retry=retry, delay_s=delay_s
    )
    logger.warning(
        f"⚙️ 已放宽 Ludwig GPU 等待阈值: target_util={target_util}, retry={retry}, delay_s={delay_s}"
    )

# ==========================================
# 原有逻辑保留：GPU 信息获取相关函数
# ==========================================
def get_gpu_info_via_nvidia_smi():
    gpu_info = []
    try:
        command =[
            "nvidia-smi", 
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw", 
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if not line:
                continue
            parts =[part.strip() for part in line.split(',')]
            if len(parts) >= 9:
                try:
                    index = int(parts[0])
                    name = parts[1]
                    total_memory_mb = int(parts[2]) * 1024 * 1024
                    used_memory_mb = int(parts[3]) * 1024 * 1024
                    free_memory_mb = int(parts[4]) * 1024 * 1024
                    gpu_util = int(parts[5]) if parts[5] != '' else 0
                    mem_util = int(parts[6]) if parts[6] != '' else 0
                    temperature = int(parts[7]) if parts[7] != '' else None
                    power_draw = float(parts[8]) if parts[8] != '' else None
                    
                    gpu_info.append({
                        "index": index, "name": name, "total_memory": total_memory_mb,
                        "used_memory": used_memory_mb, "free_memory": free_memory_mb,
                        "memory_util_percent": mem_util, "gpu_util_percent": gpu_util,
                        "memory_bandwidth_util_percent": mem_util, "temperature": temperature,
                        "power_usage_watts": power_draw
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"解析 nvidia-smi 输出行失败: {line}, 错误: {e}")
            else:
                logger.warning(f"nvidia-smi 输出格式不匹配: {line}")
                
    except subprocess.CalledProcessError as e:
        gpu_info = [{"error": "nvidia-smi command failed"}]
    except FileNotFoundError:
        gpu_info = [{"error": "nvidia-smi command not found"}]
    except Exception as e:
        gpu_info = [{"error": f"Unknown error: {e}"}]
    
    return gpu_info

def get_gpu_info():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        gpu_info =[]
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            raw_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(raw_name, bytes):
                name = raw_name.decode('utf-8')
            else:
                name = str(raw_name) 
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = mem_info.total
            used_memory = mem_info.used
            memory_util = 100 * used_memory / total_memory if total_memory > 0 else 0

            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util_rates.gpu
            
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temperature = None
            
            try:
                power_status = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except pynvml.NVMLError:
                power_status = None

            gpu_info.append({
                "index": i, "name": name, "total_memory": total_memory,
                "used_memory": used_memory, "free_memory": total_memory - used_memory,
                "memory_util_percent": round(memory_util, 2), "gpu_util_percent": gpu_util,
                "memory_bandwidth_util_percent": util_rates.memory, "temperature": temperature,
                "power_usage_watts": power_status
            })

        pynvml.nvmlShutdown()
        return gpu_info
    except pynvml.NVMLError as e:
        return get_gpu_info_via_nvidia_smi()
    except Exception as e:
        return get_gpu_info_via_nvidia_smi()

def detect_gpu_runtime_profile() -> Dict[str, Any]:
    """探测当前机器 GPU 资源与可用的 Ray accelerator_type 资源名。"""
    profile = {
        "has_gpu": False,
        "gpu_count": 0,
        "gpu_names": [],
        "max_gpu_total_memory_gb": 0.0,
        "accelerator_resource_key": None,
    }
    try:
        info = get_gpu_info()
        gpu_names = [g.get("name", "") for g in info if g.get("name")]
        profile["gpu_names"] = gpu_names
        profile["gpu_count"] = len(gpu_names)
        profile["has_gpu"] = len(gpu_names) > 0
        total_memories = [float(g.get("total_memory", 0)) / (1024 ** 3) for g in info if g.get("total_memory")]
        if total_memories:
            profile["max_gpu_total_memory_gb"] = max(total_memories)
    except Exception:
        pass
    return profile


def initialize_ray_runtime() -> Dict[str, Any]:
    """
    动态初始化 Ray：
    - CPU 根据机器核数自动分配（保留至少 2 核，避免挤占系统）
    - GPU 根据实际数量自动分配
    - 尝试识别并记录 accelerator_type:* 资源，供 trial_driver_resources 使用
    """
    profile = detect_gpu_runtime_profile()
    total_cpus = multiprocessing.cpu_count()
    ray_cpus = max(2, total_cpus - 2) if total_cpus > 4 else total_cpus
    ray_gpus = profile["gpu_count"] if profile["has_gpu"] else 0

    ray.init(num_cpus=ray_cpus, num_gpus=ray_gpus, ignore_reinit_error=True)
    patch_ludwig_gpu_wait()

    try:
        available_resources = ray.available_resources()
        accelerator_keys = [k for k in available_resources.keys() if str(k).startswith("accelerator_type:")]
        if accelerator_keys:
            # 优先使用 RTX 资源；否则使用第一个可用 accelerator_type
            rtx_keys = [k for k in accelerator_keys if "RTX" in str(k).upper()]
            profile["accelerator_resource_key"] = rtx_keys[0] if rtx_keys else accelerator_keys[0]
        logger.warning(
            "Ray initialized with CPUs=%s, GPUs=%s, accelerator_key=%s, resources=%s",
            ray_cpus,
            ray_gpus,
            profile["accelerator_resource_key"],
            available_resources,
        )
    except Exception as e:
        logger.warning(f"读取 Ray 资源失败: {e}")

    profile["ray_cpus"] = ray_cpus
    profile["ray_gpus"] = ray_gpus
    return profile


# --- 初始化 Ray（动态） ---
RAY_RUNTIME_PROFILE = initialize_ray_runtime()

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# 初始化存储目录
for d in ["datasets", "tasks", "custom_models", "trained_models/latest"]:
    os.makedirs(f"./storage/{d}", exist_ok=True)

# 全局变量
DEPLOYED_MODEL = None
DATASET_ROOT_PATH = "/home/yhz/iot_shujuji"
PREDICT_DATASET_ROOT_PATH = "/home/yhz/iot_yuceshuju"
os.makedirs(PREDICT_DATASET_ROOT_PATH, exist_ok=True)
# 混合了序列模型（CNN/LSTM）和高级表格模型（TabNet/Transformer）的超级组合
LUDWIG_ALGORITHMS = {
    # --- 新增的表格专用模型 ---
    "TabNet": {"type": "tabnet"},
    "Transformer": {"type": "tabtransformer"},
    "Deep-MLP": {"type": "concat", "num_fc_layers": 3},
    
    # --- 序列模型（基于 timeseries 输入特征） ---
    # NOTE: 不能用 combiner=sequence 来“代表”CNN/LSTM/GRU，否则在没有真正 sequence/timeseries
    # 输入特征时会在 SequenceConcatCombiner 内部因 seq_size=None 直接崩溃（你终端里看到的错误）。
    # 我们仍使用 tabular combiner（concat），并在训练时动态生成一个 timeseries 特征列来承载所有数值列。
    "CNN": {"type": "concat"},
    "LSTM": {"type": "concat"},
    "GRU": {"type": "concat"}
}

def has_available_gpu() -> bool:
    return bool(RAY_RUNTIME_PROFILE.get("has_gpu", False))

def build_gpu_train_config(use_gpu: bool) -> Dict[str, Any]:
    ludwig_executor = {
        "type": "ray",
        "num_samples": 6,  # 从 3 增加到 6，给超参搜索更多机会
        "max_concurrent_trials": 2 if use_gpu else 1, # 如果有 GPU，可以并发 2 个 trial
        "cpu_resources_per_trial": max(1, min(4, int(RAY_RUNTIME_PROFILE.get("ray_cpus", 2) // 2))),
        "gpu_resources_per_trial": 1 if use_gpu else 0,
    }

    accelerator_resource_key = RAY_RUNTIME_PROFILE.get("accelerator_resource_key")
    if use_gpu and accelerator_resource_key:
        ludwig_executor["trial_driver_resources"] = {accelerator_resource_key: 1.0}

    return {
        "pycaret_use_gpu": use_gpu,
        "ludwig_executor": ludwig_executor,
        "ludwig_scaling_config": {"use_gpu": use_gpu}
    }


def get_ludwig_dynamic_train_profile(use_gpu: bool) -> Dict[str, Any]:
    """
    基于当前机器显存动态设定训练强度，提高 GPU 利用率。
    """
    if not use_gpu:
        return {
            "batch_candidates": [8, 16], # 降低默认 batch size
            "default_batch_size": 8,
            "epochs": 20,
            "early_stop": 8,
        }

    max_mem_gb = float(RAY_RUNTIME_PROFILE.get("max_gpu_total_memory_gb", 0.0))
    if max_mem_gb >= 20:
        return {
            "batch_candidates": [32, 64],
            "default_batch_size": 32,
            "epochs": 40,
            "early_stop": 10,
        }
    if max_mem_gb >= 10:
        return {
            "batch_candidates": [16, 32],
            "default_batch_size": 16,
            "epochs": 30,
            "early_stop": 10,
        }
    return {
        "batch_candidates": [8, 16],
        "default_batch_size": 8,
        "epochs": 25,
        "early_stop": 10,
    }


# ==========================================
# 【核心修改】自动特征检测函数
# ==========================================
def detect_column_types(df, target_col):
    """自动检测 DataFrame 中各列的类型（文本、数值、类别），并强制排除 target_col"""
    column_types = {}
    id_like_name_pattern = re.compile(r"(?:^|_)(id|uuid|identifier|sn|serial|imei|imsi|mac)(?:$|_)")
    time_like_name_pattern = re.compile(r"(?:^|_)(time|timestamp|date|created|updated|ts)(?:$|_)")
    for col in df.columns:
        # ⭐⭐⭐ 关键修复：强制跳过目标列，防止其被用作输入特征导致数据泄露
        if col == target_col:
            column_types[col] = 'skip'
            continue

        normalized_col = str(col).strip().lower()
        # 关键：ID/时间类字段无论是字符串还是数值型，都应该跳过（否则会严重伤泛化，深度模型尤其明显）
        if id_like_name_pattern.search(normalized_col) or time_like_name_pattern.search(normalized_col):
            column_types[col] = 'skip'
            continue

        series = df[col].dropna()
        if series.empty:
            column_types[col] = 'skip'
            continue

        dtype = series.dtype
        unique_count = series.nunique()

        # 常量列无信息量，且数值常量列会触发 Ludwig zscore 异常
        if unique_count <= 1:
            column_types[col] = 'skip'
            continue

        if dtype == 'object' or dtype.name == 'string':
            series_as_str = series.astype(str)
            ratio_unique = unique_count / len(series) if len(series) > 0 else 0
            avg_length = series_as_str.str.len().mean() if len(series) > 0 else 0
            json_like_ratio = (
                series_as_str.str.contains(r"^\s*[\{\[]", regex=True, na=False).mean()
                if len(series) > 0 else 0
            )

            # 保留业务文本 value 字段；其余大JSON字段跳过
            if normalized_col == "value":
                column_types[col] = 'text'
            elif avg_length > 200 and json_like_ratio > 0.3:
                column_types[col] = 'skip'
            elif ratio_unique > 0.95 and avg_length < 50:
                column_types[col] = 'category'
            else:
                # 规则：唯一值占比高且平均长度>10，判定为长文本
                if ratio_unique > 0.5 and avg_length > 10:
                    column_types[col] = 'text'
                else:
                    column_types[col] = 'category'
        elif pd.api.types.is_numeric_dtype(dtype):
            # -------- 不依赖列名的“伪特征”识别（数值型ID/时间戳/流水号）--------
            # 这些列通常“几乎全唯一”，并且经常是整数/近似整数，或呈现强单调性。
            # 若不剔除，会严重伤泛化，深度表格模型尤为明显。
            n = len(series)
            if n > 0:
                ratio_unique = unique_count / n
            else:
                ratio_unique = 0.0

            # 近似整数判断：绝大多数值接近整数（例如设备号/时间戳/计数器）
            s_float = series.astype(float)
            finite = s_float.replace([np.inf, -np.inf], np.nan).dropna()
            near_int_ratio = 0.0
            is_monotonic_like = False
            if len(finite) >= 20:
                frac = finite.iloc[: min(len(finite), 2000)]  # 限制采样，避免大表过慢
                near_int_ratio = (np.abs(frac - np.round(frac)) < 1e-6).mean()

                # 单调性：用相邻差分符号占比估计（强趋势的时间戳/索引列会非常高）
                diffs = frac.diff().dropna()
                if len(diffs) >= 10:
                    pos = (diffs > 0).mean()
                    neg = (diffs < 0).mean()
                    is_monotonic_like = max(pos, neg) > 0.98

            # 规则：几乎全唯一 + (近似整数 或 强单调) => skip
            # 这会自动跳过数值型 timestamp / id / 自增序号
            if ratio_unique > 0.98 and (near_int_ratio > 0.98 or is_monotonic_like):
                column_types[col] = 'skip'
                continue

            if np.issubdtype(dtype, np.integer) and unique_count <= 20:
                column_types[col] = 'category'
            else:
                column_types[col] = 'number'
        else:
            column_types[col] = 'category'
    return column_types


def _detect_time_like_columns(df: pd.DataFrame) -> list:
    """Detect time-like columns using both names and value statistics.

    Works across datasets with different factory naming conventions.
    """
    if df is None or df.empty:
        return []

    name_pattern = re.compile(r"(?:^|_)(time|timestamp|datetime|date|created|updated|ts)(?:$|_)")
    cols = list(df.columns)

    # 1) Name-based candidates (fast path)
    name_candidates = [c for c in cols if name_pattern.search(str(c).strip().lower())]

    # 2) Value-based candidates: epoch-like numeric or parseable datetime strings
    value_candidates = []
    for c in cols:
        if c in name_candidates:
            value_candidates.append(c)
            continue

        s = df[c].dropna()
        if s.empty:
            continue

        # Sample to keep runtime bounded on huge datasets
        s = s.iloc[: min(len(s), 3000)]

        # Datetime-like strings
        if s.dtype == "object" or getattr(s.dtype, "name", "") == "string":
            try:
                parsed = pd.to_datetime(s.astype(str), errors="coerce", utc=True, infer_datetime_format=True)
                ok = parsed.notna().mean()
                if ok > 0.95:
                    value_candidates.append(c)
            except Exception:
                pass
            continue

        # Numeric epoch-like / monotonic sequence
        if pd.api.types.is_numeric_dtype(s.dtype):
            x = s.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) < 50:
                continue

            # High uniqueness suggests id/time rather than sensor readings (but we validate further).
            ratio_unique = x.nunique() / len(x)

            # Monotonicity score: most diffs have the same sign
            diffs = x.diff().dropna()
            if len(diffs) < 10:
                continue
            pos = (diffs > 0).mean()
            neg = (diffs < 0).mean()
            mono = max(pos, neg)

            # Epoch magnitude heuristics: seconds ~1e9, ms ~1e12, us ~1e15
            med_abs = float(np.median(np.abs(x)))
            epoch_like = med_abs > 1e9

            # Time-like if it is both mostly unique and strongly monotonic, or clearly epoch-like.
            if (ratio_unique > 0.95 and mono > 0.98) or (epoch_like and mono > 0.9):
                value_candidates.append(c)

    # De-dup preserving order
    seen = set()
    out = []
    for c in name_candidates + value_candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out
# ==========================================
# 后台异步训练引擎 (全动态多模态融合版) - 增加调试日志
# ==========================================
def run_hybrid_pipeline(task_id: str, file_paths: list, train_ratio: float, label_column: str, selected_models_list: list):
    model_root = "./storage/trained_models/latest/"

    if os.path.exists(model_root):
        logger.info(f"🧹 开始清理旧模型目录: {model_root}")
        try:
            # 尝试直接删除目录树，忽略常见错误（如权限或临时锁定）
            shutil.rmtree(model_root, ignore_errors=True)
            logger.info(f"✅ 旧模型目录已清理: {model_root}")
        except Exception as e:
            # 如果 ignore_errors=True 仍失败，则记录更详细的错误
            logger.warning(f"⚠️ 清理旧模型目录时遇到问题 (可能文件被占用): {model_root}, 错误: {e}")
    os.makedirs(model_root, exist_ok=True)
    task_file = f"./storage/tasks/{task_id}.json"
    leaderboard = []
    def write_task(status: str, msg: str = "", models=None):
        payload = {"status": status, "models": models if models is not None else leaderboard}
        if msg:
            payload["msg"] = msg
        with open(task_file, "w") as wf:
            json.dump(payload, wf)

    with open(task_file, "w") as f:
        json.dump({"status": "running", "msg": "任务已创建，准备读取数据", "models":[]}, f)

    dfs =[]
    for fp in file_paths:
        if not os.path.exists(fp): continue
        sep = '\t' if fp.lower().endswith('.tsv') else ','
        try:
            dfs.append(pd.read_csv(fp, sep=sep))
        except Exception as e:
            logger.error(f"读取文件失败 {fp}: {str(e)}")

    if not dfs:
        with open(task_file, "w") as f:
            json.dump({"status": "failed", "msg": "未找到有效数据", "models":[]}, f)
        return

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    label_column = label_column.strip()

    if label_column not in df.columns:
        with open(task_file, "w") as f:
            json.dump({"status": "failed", "msg": f"缺少标签列: '{label_column}'", "models":[]}, f)
        return

    # 统一清洗：移除标签缺失行（PyCaret 对目标列缺失会直接报错）
    before_rows = len(df)
    df = df.dropna(subset=[label_column]).copy()
    after_rows = len(df)
    dropped_rows = before_rows - after_rows
    if dropped_rows > 0:
        logger.warning(f"⚠️ Task {task_id}: 目标列 '{label_column}' 存在缺失，已移除 {dropped_rows} 行。")
    if after_rows == 0:
        with open(task_file, "w") as f:
            json.dump({"status": "failed", "msg": f"标签列 '{label_column}' 全为空，无法训练。", "models":[]}, f)
        return

    # 自动推断特征类型
    column_types = detect_column_types(df, label_column) # 使用修复后的函数
    text_columns = [col for col, typ in column_types.items() if typ == 'text']
    first_text_col = text_columns[0] if text_columns else None
    use_gpu = has_available_gpu()
    gpu_cfg = build_gpu_train_config(use_gpu)
    ludwig_train_profile = get_ludwig_dynamic_train_profile(use_gpu)

    # 是否存在“真实时间顺序”信号：用于决定是否启用基于 timeseries 的 CNN/LSTM/GRU。
    # 不同工厂数据集列名不统一，因此同时使用“列名 + 数值统计特征”来自动识别时间列。
    time_like_cols = _detect_time_like_columns(df)
    has_time_like_col = len(time_like_cols) > 0

    # --- 新增调试日志 ---
    print(f"[DEBUG] Task {task_id} - Target column: '{label_column}'")
    print(f"[DEBUG] Task {task_id} - Detected column types: {column_types}")
    input_feature_names = [col for col, typ in column_types.items() if typ != 'skip']
    print(f"[DEBUG] Task {task_id} - Potential input features: {input_feature_names}")
    print(f"[DEBUG] Task {task_id} - GPU available: {use_gpu}")
    print(f"[DEBUG] Task {task_id} - Ray runtime profile: {RAY_RUNTIME_PROFILE}")
    print(f"[DEBUG] Task {task_id} - Ludwig executor config: {gpu_cfg['ludwig_executor']}")
    print(f"[DEBUG] Task {task_id} - Ludwig dynamic train profile: {ludwig_train_profile}")
    print(f"[DEBUG] Task {task_id} - Detected time-like columns: {time_like_cols}")
    
    if label_column in input_feature_names:
        logger.error(f"[CRITICAL ERROR] Task {task_id}: Label column '{label_column}' is included in input features! This indicates a bug in detect_column_types.")
        with open(task_file, "w") as f:
            json.dump({"status": "failed", "msg": f"Internal error: Label column was incorrectly added as an input feature.", "models":[]}, f)
        return
    else:
        logger.info(f"[INFO] Task {task_id}: Label column '{label_column}' correctly excluded from input features.")
    print(f"[DEBUG] Task {task_id} - Data shape after cleaning: {df.shape}")
    print(f"[DEBUG] Task {task_id} - Label distribution:\n{df[label_column].value_counts()}")
    print(f"[DEBUG] Task {task_id} - Feature columns (excluding label): {input_feature_names}")
    
    # 检查输入特征是否有 NaN 或 Inf
    input_df_subset = df[input_feature_names]
    if input_df_subset.isnull().any().any():
        logger.warning(f"⚠️ Task {task_id}: Input features contain NaN values.")
    if input_df_subset.isin([np.inf, -np.inf]).any().any():
        logger.warning(f"⚠️ Task {task_id}: Input features contain Inf/-Inf values.")
    # --- END 新增调试日志 ---

    try:
        # ================== 引擎 1：PyCaret 传统机器学习 ==================
        write_task("running", "开始 PyCaret 训练")
        # 为避免文本列导致 OOM，PyCaret 统一使用非文本特征子集；这样既稳定又能保留 PyCaret 全流程
        pycaret_feature_cols = [c for c, t in column_types.items() if t in ("number", "category")]
        if not pycaret_feature_cols:
            logger.warning(f"⚠️ Task {task_id}: 无可用非文本特征，PyCaret 跳过。")
        else:
            pycaret_df = df[pycaret_feature_cols + [label_column]].copy()
            setup_kwargs = {
                "data": pycaret_df, "target": label_column, "train_size": train_ratio,
                "verbose": False, "session_id": 42,
                "n_jobs": 1, "use_gpu": gpu_cfg["pycaret_use_gpu"], "fold": 3
            }
            setup(**setup_kwargs)

            # --- 修改：只训练选中的 PyCaret 模型 ---
            # 注意：这里的 py_caret_map 需要根据 selected_models_list 中的名称进行匹配
            # 假设 selected_models_list 中包含的是 'rf', 'nb' 等
            # 那么需要确保前端发送的名称与这里匹配
            py_caret_codes_map = {'rf': 'RF', 'nb': 'NB'} # 内部代码到显示名的映射
            for internal_code, display_name in py_caret_codes_map.items():
                if internal_code in selected_models_list: # 检查内部代码是否被选中
                    model_name = internal_code # 使用基础模型名作为模型名称
                    try:
                        trained_model = create_model(internal_code, verbose=False)
                        tuned_model = tune_model(trained_model, optimize='F1', n_iter=10, verbose=False)
                        metrics_df = pull()

                        f1_score = metrics_df.loc['Mean', 'F1']
                        accuracy = metrics_df.loc['Mean', 'Accuracy']
                        
                        # --- 关键修改：保存路径直接使用基础模型名 ---
                        out_path = f"./storage/trained_models/latest/{model_name}" # 例如 ./storage/trained_models/latest/rf
                        save_model(tuned_model, out_path) # 这会生成 rf.pkl

                        leaderboard.append({
                            "model_name": model_name, # 使用基础模型名
                            "f1_score": round(float(f1_score), 4),
                            "accuracy": round(float(accuracy), 4), 
                            # --- 修改 model_path 以反映实际的 .pkl 文件名 ---
                            "model_path": out_path + ".pkl" # 例如 rf.pkl
                        })
                        # 实时更新状态，报告单个模型完成
                        write_task("running", f"PyCaret 模型完成: {model_name}", leaderboard)
                    except Exception as e:
                        logger.error(f"❌ {model_name} 训练异常: {str(e)}")
                        # 失败模型不再伪造 0 分，避免与真实低分混淆
                        leaderboard.append({
                            "model_name": model_name, "f1_score": None,
                            "accuracy": None, "model_path": None, "status": "failed",
                            "error": str(e)
                        })
                        write_task("running", f"PyCaret 模型失败: {model_name}", leaderboard)


        # ================== 引擎 2：Ludwig 深度学习超参搜索 ==================
        write_task("running", "开始 Ludwig 训练")
        # 计算类别权重（应对类别不平衡，否则很容易塌缩成“永远预测多数类”，导致所有模型分数完全一样）
        label_counts = df[label_column].value_counts()
        if label_counts.empty:
            class_weights = None
        else:
            max_count = float(label_counts.max())
            # Ludwig 接受 dict 或 list；这里用 dict，key 用字符串以保证 JSON 序列化稳定
            class_weights = {str(k): float(max_count / float(v)) for k, v in label_counts.items()}
        # --- 修改：只训练选中的 Ludwig 模型 ---
        # 假设 selected_models_list 中包含的是 'TabNet', 'Transformer', 'Deep-MLP', 'CNN', 'LSTM', 'GRU' 等
        # LUDWIG_ALGORITHMS 应该包含这些键
        for base_model_name in LUDWIG_ALGORITHMS.keys(): # 遍历所有支持的 Ludwig 算法
            if base_model_name in selected_models_list: # 检查是否被选中
                name = base_model_name # 使用基础模型名作为模型名称
                write_task("running", f"正在训练 Ludwig 模型: {name}")
                # 为每个 Ludwig 模型构建独立数据副本（避免不同模型间互相污染特征列）
                df_ludwig = df.copy()

                # 对 CNN/LSTM/GRU：把所有数值列打包成一个 timeseries 列，让 encoder 真正跑 CNN/RNN
                seq_models = {"CNN", "LSTM", "GRU"}
                numeric_cols = [c for c, t in column_types.items() if t == "number" and c != label_column]
                timeseries_col = "__numeric_timeseries__"
                # 只有检测到时间列时才启用（否则是“伪序列”，通常会导致 F1 低得离谱）
                use_timeseries_pack = base_model_name in seq_models and len(numeric_cols) > 0 and has_time_like_col
                # 没有时间列时，不跳过：退化为“表格深度模型变体”继续训练，
                # 但会使用不同的 combiner 超参搜索空间，避免与 Deep-MLP 同构同分。
                is_tabular_seq_fallback = base_model_name in seq_models and not use_timeseries_pack
                if use_timeseries_pack:
                    # space tokenizer：每行是一个数值序列字符串，例如 "0.1 3.5 8.0 ..."
                    df_ludwig[timeseries_col] = (
                        df_ludwig[numeric_cols]
                        .astype(float)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0.0)
                        .agg(lambda r: " ".join(f"{v:.8g}" for v in r.values), axis=1)
                    )

                temp_dataset_path = f"./storage/datasets/temp_{task_id}_{uuid.uuid4()}.csv"
                df_ludwig.to_csv(temp_dataset_path, index=False)

                # 动态组装当前模型的输入特征
                # --- 防御性修复：再次确保 label_column 不被加入 input_features ---
                input_features =[]
                for col_name, col_type in column_types.items():
                    if col_type == 'skip' or col_name == label_column: 
                        continue
                    # 如果启用 timeseries 打包，则单独的数值列不再作为独立输入特征加入
                    if use_timeseries_pack and col_type == "number":
                        continue
                    elif col_type == 'text':
                        input_features.append({
                            "name": col_name, "type": "text", 
                            "preprocessing": {"tokenizer": "characters"}
                        })
                    elif col_type == 'number':
                        # ⭐ 增加 zscore，防止传感器数据（如压力几千，震动零点几）导致 CNN/LSTM 梯度爆炸
                        input_features.append({
                            "name": col_name, "type": "number",
                            "preprocessing": {"normalization": "zscore"}
                        })
                    elif col_type == 'category':
                        input_features.append({"name": col_name, "type": "category"})

                # 如果启用 timeseries 打包，追加一个 timeseries 输入特征，并按模型选择 encoder
                if use_timeseries_pack:
                    if base_model_name == "CNN":
                        ts_encoder = {"type": "parallel_cnn"}
                    elif base_model_name == "LSTM":
                        ts_encoder = {"type": "rnn", "cell_type": "lstm", "bidirectional": True}
                    else:  # GRU
                        ts_encoder = {"type": "rnn", "cell_type": "gru", "bidirectional": True}

                    input_features.append(
                        {
                            "name": timeseries_col,
                            "type": "timeseries",
                            "preprocessing": {
                                "tokenizer": "space",
                                # 每行序列长度=数值列个数；设得更精确避免 padding/cutoff 影响
                                "timeseries_length_limit": max(1, len(numeric_cols)),
                                "padding_value": 0.0,
                                "padding": "right",
                            },
                            "encoder": ts_encoder,
                        }
                    )
                # --- 防御性修复：再次确保 label_column 不被加入 input_features ---
                # 这是一个额外的安全网，以防 detect_column_types 有任何遗漏
                filtered_input_features = []
                for feat in input_features:
                    if feat["name"] == label_column:
                        logger.warning(f"⚠️ [DEFENSE] Column '{label_column}' was found in raw input_features but has been removed. This should ideally be handled by detect_column_types.")
                        continue # 跳过这个特征
                    filtered_input_features.append(feat)

                input_features = filtered_input_features # 更新 input_features 列表

                # --- 调试日志：确认最终的 input_features ---
                final_input_names = [f["name"] for f in input_features]
                print(f"[DEBUG] Task {task_id} - Final Ludwig input features for {base_model_name}: {final_input_names}")
                print(f"[DEBUG] Task {task_id} - Text features for {base_model_name}: {[f['name'] for f in input_features if f['type'] == 'text']}")
                if label_column in final_input_names:
                    logger.critical(f"🚨 [CRITICAL ERROR] Task {task_id} - {base_model_name}: Label column '{label_column}' is STILL present in final input features after ALL filters! Aborting this model.")
                    continue # 如果发生，跳过此模型训练
                else:
                    logger.info(f"✅ [INFO] Task {task_id} - {base_model_name}: Label column '{label_column}' correctly excluded from final input features. Proceeding.")
                # --- END 调试日志 ---
                
                hyperopt_params = {
                    # 扩大学习率搜索范围：TabNet/Transformer 在表格上经常需要更大的 LR 才能收敛到更优解
                    "trainer.learning_rate": {"space": "choice", "categories":[0.001, 0.0005, 0.0001]},
                    "trainer.batch_size": {"space": "choice", "categories": ludwig_train_profile["batch_candidates"]}
                }
                if first_text_col:
                    hyperopt_params[f"input_features.{first_text_col}.encoder.dropout"] = {"space": "uniform", "lower": 0.1, "upper": 0.5} # 稍微增加 dropout

                # —— 模型专属的结构超参搜索空间（这是 Transformer/MLP 分数“不动”的主要原因）——
                if base_model_name == "Deep-MLP":
                    hyperopt_params.update(
                        {
                            "combiner.num_fc_layers": {"space": "choice", "categories": [1, 2, 3, 4]},
                            "combiner.output_size": {"space": "choice", "categories": [64, 128, 256, 512]},
                            "combiner.dropout": {"space": "uniform", "lower": 0.0, "upper": 0.5},
                        }
                    )
                elif base_model_name in {"CNN", "LSTM", "GRU"} and is_tabular_seq_fallback:
                    # 无时间列时：把“序列模型”当作不同的表格 MLP 结构族去搜索（避免同分）
                    if base_model_name == "CNN":
                        hyperopt_params.update(
                            {
                                "combiner.num_fc_layers": {"space": "choice", "categories": [3, 4, 5]},
                                "combiner.output_size": {"space": "choice", "categories": [256, 512]},
                                "combiner.dropout": {"space": "uniform", "lower": 0.0, "upper": 0.3},
                            }
                        )
                    elif base_model_name == "LSTM":
                        hyperopt_params.update(
                            {
                                "combiner.num_fc_layers": {"space": "choice", "categories": [2, 3, 4]},
                                "combiner.output_size": {"space": "choice", "categories": [64, 128, 256]},
                                "combiner.dropout": {"space": "uniform", "lower": 0.1, "upper": 0.5},
                            }
                        )
                    else:  # GRU
                        hyperopt_params.update(
                            {
                                "combiner.num_fc_layers": {"space": "choice", "categories": [1, 2, 3]},
                                "combiner.output_size": {"space": "choice", "categories": [128, 256, 512]},
                                "combiner.dropout": {"space": "uniform", "lower": 0.0, "upper": 0.4},
                            }
                        )
                elif base_model_name in {"CNN", "LSTM", "GRU"} and use_timeseries_pack:
                    # 有时间列时：序列模型主要靠 timeseries encoder；combiner 仅做轻量搜索
                    hyperopt_params.update(
                        {
                            "combiner.num_fc_layers": {"space": "choice", "categories": [0, 1, 2]},
                            "combiner.output_size": {"space": "choice", "categories": [64, 128, 256]},
                            "combiner.dropout": {"space": "uniform", "lower": 0.0, "upper": 0.3},
                        }
                    )
                elif base_model_name == "TabNet":
                    hyperopt_params.update(
                        {
                            "combiner.size": {"space": "choice", "categories": [16, 32, 64]},
                            "combiner.output_size": {"space": "choice", "categories": [64, 128, 256]},
                            "combiner.num_steps": {"space": "choice", "categories": [3, 4, 5]},
                            "combiner.dropout": {"space": "uniform", "lower": 0.0, "upper": 0.3},
                            "combiner.sparsity": {"space": "choice", "categories": [1e-5, 1e-4, 1e-3]},
                            "combiner.bn_virtual_bs": {"space": "choice", "categories": [128, 256, 512, 1024]},
                        }
                    )
                elif base_model_name == "Transformer":
                    # tabtransformer/common transformer options
                    hyperopt_params.update(
                        {
                            "combiner.hidden_size": {"space": "choice", "categories": [64, 128, 256]},
                            "combiner.transformer_output_size": {"space": "choice", "categories": [64, 128, 256]},
                            "combiner.num_layers": {"space": "choice", "categories": [1, 2, 3]},
                            "combiner.num_heads": {"space": "choice", "categories": [2, 4, 8]},
                            "combiner.dropout": {"space": "uniform", "lower": 0.0, "upper": 0.3},
                            "combiner.fc_dropout": {"space": "uniform", "lower": 0.0, "upper": 0.5},
                            "combiner.num_fc_layers": {"space": "choice", "categories": [0, 1, 2]},
                            "combiner.output_size": {"space": "choice", "categories": [64, 128, 256]},
                        }
                    )

                config = {
                    "input_features": input_features,
                    "output_features": [
                        {
                            "name": label_column,
                            "type": "category",
                            # 使用带 class_weights 的 softmax loss，缓解类别不平衡导致的多数类塌缩
                            "loss": (
                                {"type": "softmax_cross_entropy", "class_weights": class_weights}
                                if class_weights
                                else {"type": "softmax_cross_entropy"}
                            ),
                        }
                    ],
                    # IMPORTANT: use the selected Ludwig model (combiner).
                    # Previously LUDWIG_ALGORITHMS was only used to build display names,
                    # causing every "different" Ludwig model to train with the same default combiner.
                    "combiner": LUDWIG_ALGORITHMS[base_model_name],
                    "backend": {"type": "local"},
                    "preprocessing": {
                        "split": {
                            # 分层切分，保证 train/val/test 都包含各类样本，避免评估被多数类主导
                            "type": "stratify",
                            "column": label_column,
                            "probabilities": [train_ratio, (1.0 - train_ratio) / 2.0, (1.0 - train_ratio) / 2.0]
                        }
                    },
                    "trainer": {
                            "epochs": ludwig_train_profile["epochs"],
                            "early_stop": ludwig_train_profile["early_stop"],
                            "batch_size": ludwig_train_profile["default_batch_size"],  
                        },
                    "hyperopt": {
                        # NOTE: 当前 Ludwig 版本的 hyperopt config schema 不支持 `avg_f1_score_macro` 作为 metric。
                        # 为了在类别不平衡下避免“全预测多数类”的塌缩解，这里改为最小化验证集 loss，
                        # 并配合 output feature 的 class_weights（上方已注入）来提升少数类关注度。
                        "goal": "minimize", "metric": "loss", "output_feature": label_column,
                        "search_alg": {"type": "hyperopt"},
                    "executor": gpu_cfg["ludwig_executor"],
                        "parameters": hyperopt_params
                    }
                }
                
                # --- 新增：启用 Ludwig 详细日志 ---
                import logging as ludwig_logging
                ludwig_logger = ludwig_logging.getLogger("ludwig")
                ludwig_logger.setLevel(ludwig_logging.INFO) 
                # --- END 新增：启用 Ludwig 详细日志 ---

                # --- 关键修改：输出目录直接使用基础模型名 ---
                out_dir = f"./storage/trained_models/latest/{name}" # 例如 ./storage/trained_models/latest/TabNet
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir, ignore_errors=True)
                try:
                    hyperopt_results = hyperopt(config=config, dataset=temp_dataset_path, output_directory=out_dir)
                    
                    # --- 更稳健的解析：优先读取 hyperopt_statistics.json ---
                    # Ray 的 results_df 在 trial 全部失败/或版本差异时可能没有 metric_score 列，导致 KeyError。
                    stats_path = os.path.join(out_dir, "hyperopt", "hyperopt_statistics.json")
                    best_f1 = None
                    best_acc = None

                    if os.path.exists(stats_path):
                        with open(stats_path, "r") as sf:
                            stats_json = json.load(sf)
                        results_list = stats_json.get("hyperopt_results", []) or []
                        if not results_list:
                            raise ValueError(f"{name} hyperopt returned no successful trials (hyperopt_results empty).")

                        # Even if hyperopt optimizes `loss`, what we care about is F1.
                        # Prefer selecting the trial with the best macro F1 from eval_stats (when available),
                        # and fall back to metric_score respecting goal if macro F1 is missing.
                        def _macro_f1(entry: dict) -> float:
                            try:
                                s = (entry.get("eval_stats", {}) or {}).get(label_column, {}) or {}
                                overall = (s.get("overall_stats", {}) or {})
                                v = overall.get("avg_f1_score_macro") or overall.get("f1_score_macro")
                                return float(v) if v is not None else float("-inf")
                            except Exception:
                                return float("-inf")

                        best_by_f1 = max(results_list, key=_macro_f1)
                        if _macro_f1(best_by_f1) != float("-inf"):
                            best_entry = best_by_f1
                        else:
                            hyperopt_goal = (stats_json.get("hyperopt_config", {}) or {}).get("goal", "maximize")
                            if str(hyperopt_goal).lower() == "minimize":
                                best_entry = min(results_list, key=lambda r: float(r.get("metric_score", float("inf"))))
                            else:
                                best_entry = max(results_list, key=lambda r: float(r.get("metric_score", float("-inf"))))
                        eval_stats = best_entry.get("eval_stats", {}) or {}
                        feature_stats = eval_stats.get(label_column, {}) or {}
                        overall_stats = feature_stats.get("overall_stats", {}) or {}

                        best_f1 = (
                            overall_stats.get("avg_f1_score_macro")
                            or overall_stats.get("avg_f1_score_weighted")
                            or overall_stats.get("avg_f1_score_micro")
                            or overall_stats.get("f1_score_macro")
                            or overall_stats.get("f1_score_weighted")
                            or overall_stats.get("f1_score_micro")
                            or feature_stats.get("f1_score_macro")
                            or feature_stats.get("f1_score_weighted")
                            or feature_stats.get("f1_score_micro")
                            or feature_stats.get("f1_score")
                        )

                        # Ludwig 分类的整体准确率常用 token_accuracy / accuracy_micro
                        best_acc = (
                            overall_stats.get("token_accuracy")
                            or feature_stats.get("accuracy_micro")
                            or feature_stats.get("accuracy")
                            or feature_stats.get("token_accuracy")
                        )
                    else:
                        # fallback：旧逻辑（尽量少依赖列名）
                        experiment_analysis = hyperopt_results.experiment_analysis
                        trials_df = experiment_analysis.results_df
                        if trials_df is None or trials_df.empty:
                            raise ValueError(f"{name} hyperopt returned no results.")

                        score_col = "metric_score" if "metric_score" in trials_df.columns else None
                        if score_col is None:
                            raise ValueError(f"{name} missing metric_score in results_df. columns={list(trials_df.columns)}")

                        # Respect goal if present on the returned object (fallback to maximize).
                        try:
                            hyperopt_goal = str(getattr(hyperopt_results, "hyperopt_config", {}).get("goal", "maximize")).lower()
                        except Exception:
                            hyperopt_goal = "maximize"
                        best_trial_row = (
                            trials_df.loc[trials_df[score_col].idxmin()]
                            if hyperopt_goal == "minimize"
                            else trials_df.loc[trials_df[score_col].idxmax()]
                        )
                        eval_stats_raw = best_trial_row.get("eval_stats", "{}")
                        if isinstance(eval_stats_raw, str):
                            eval_stats = json.loads(eval_stats_raw)
                        else:
                            eval_stats = eval_stats_raw
                        feature_stats = eval_stats.get(label_column, {}) or {}
                        overall_stats = feature_stats.get("overall_stats", {}) or {}
                        best_f1 = overall_stats.get("avg_f1_score_macro") or overall_stats.get("avg_f1_score_weighted")
                        best_acc = overall_stats.get("token_accuracy") or feature_stats.get("accuracy_micro") or feature_stats.get("accuracy")

                    if best_f1 is None:
                        raise ValueError(f"{name} 缺少可用的 F1 指标(best_f1 is None)")
                    if best_acc is None:
                        best_acc = 0.0

                    logger.info(f"✅ {name}: Best trial F1={float(best_f1):.4f}, Acc={float(best_acc):.4f}")

                    leaderboard.append(
                        {
                            "model_name": name, # 使用基础模型名
                            "f1_score": round(float(best_f1), 4),
                            "accuracy": round(float(best_acc), 4),
                            "model_path": out_dir, # 例如 ./storage/trained_models/latest/TabNet
                        }
                    )
                    write_task("running", f"Ludwig 模型完成: {name}", leaderboard)

                    # ================== ⭐ 提取最佳模型（核心修改） ==================
                    import glob

                    try:
                        # 1. 找到 hyperopt 目录下的所有 trial
                        hyperopt_dir = os.path.join(out_dir, "hyperopt")
                        trial_dirs = glob.glob(os.path.join(hyperopt_dir, "trial_*"))
                        if not trial_dirs:
                            raise Exception("No trial directories found inside hyperopt folder after run.")

                        # 2. 策略：找到包含 'checkpoint_*' 子目录且该子目录下有 'model' 目录的 trial
                        best_trial_dir = None
                        final_model_source = None
                        for trial_dir in trial_dirs:
                             # 检查 trial 目录下是否有 checkpoint_* 文件夹
                             checkpoint_dirs = glob.glob(os.path.join(trial_dir, "checkpoint_*"))
                             if checkpoint_dirs:
                                 # 选择最新的 checkpoint (通常是最终模型)
                                 latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
                                 candidate_model_dir = os.path.join(latest_checkpoint, "model")
                                 if os.path.exists(candidate_model_dir):
                                     best_trial_dir = trial_dir
                                     final_model_source = candidate_model_dir
                                     break # 找到第一个有效的就停止，通常是最佳的

                        if not best_trial_dir or not final_model_source:
                            raise Exception("No valid trial directory with a complete 'model' subdirectory found.")

                        # 3. 目标模型路径 (固定在 out_dir 下)
                        final_model_dst = os.path.join(out_dir, "model")

                        # 4. 删除旧的目标模型目录 (如果存在)
                        if os.path.exists(final_model_dst):
                            shutil.rmtree(final_model_dst, ignore_errors=True)
                            logger.info(f"🧹 Cleaned up old final model at {final_model_dst}")

                        # 5. 复制最佳模型到固定位置
                        shutil.copytree(final_model_source, final_model_dst)

                        logger.info(f"✅ Extracted final model from '{best_trial_dir}' to: {final_model_dst}")

                    except Exception as extract_e:
                        logger.error(f"❌ Failed to extract final model for {name}: {str(extract_e)}")
                        # 如果提取失败，后续的部署肯定会失败。这里可以抛出异常中断，或记录错误。
                        # 为了流程继续，可以选择不中断，但部署时会失败。
                        # raise # 可选：中断当前模型训练
                    # ================== ⭐ 清理 hyperopt（节省空间，可选） ==================
                    try:
                        hyperopt_dir_to_remove = os.path.join(out_dir, "hyperopt")
                        if os.path.exists(hyperopt_dir_to_remove):
                            shutil.rmtree(hyperopt_dir_to_remove, ignore_errors=True)
                            logger.info(f"🧹 Cleaned up hyperopt temp files for {name}")
                    except Exception as cleanup_e:
                        logger.warning(f"Failed to cleanup hyperopt for {name}: {str(cleanup_e)}")


                except Exception as e:
                    logger.error(f"❌ {name} 异常: {str(e)}")
                    # 失败模型不再写入 0 分，明确标注失败原因
                    leaderboard.append({
                        "model_name": name, "f1_score": None, # 使用基础模型名
                        "accuracy": None, "model_path": None, "status": "failed",
                        "error": str(e)
                    })
                    write_task("running", f"Ludwig 模型失败: {name}，原因: {str(e)}", leaderboard)
                finally:
                    if os.path.exists(temp_dataset_path): os.remove(temp_dataset_path)
        # 训练结束后，对排行榜进行排序
        leaderboard[:] = sorted(
            leaderboard,
            key=lambda x: (x.get('f1_score') is not None, x.get('f1_score') or -1),
            reverse=True
        ) # 使用 slice [:] 来就地修改
        write_task("completed", "训练完成", leaderboard)

    except Exception as e:
        logger.error(traceback.format_exc())
        write_task("failed", str(e), [])
# ==========================================
# 新增：获取所有已训练模型列表
# ==========================================
@app.get("/api/models/list")
async def list_trained_models():
    """获取 ./storage/trained_models/latest 目录下的所有模型文件夹和模型文件"""
    # --- 添加这行日志 ---
    logger.info("--- DEBUG: list_trained_models function called ---")
    
    # --- 从 latest 目录获取 ---
    latest_model_dir = "./storage/trained_models/latest"
    latest_items = []
    if os.path.exists(latest_model_dir):
        for item in os.listdir(latest_model_dir):
            item_path = os.path.join(latest_model_dir, item)
            # 如果是文件夹，添加文件夹名 (例如 TabNet, Transformer)
            if os.path.isdir(item_path):
                latest_items.append(item)
            # 如果是 pkl 文件且是 rf/nb，直接添加基础模型名
            elif item.endswith('.pkl') and item in ['rf.pkl', 'nb.pkl']: # 这一行是关键
                # --- 修改：直接添加基础模型名，不再添加 _AutoFeat 后缀 ---
                model_name = item.replace('.pkl', '') # 例如 'rf.pkl' -> 'rf'
                latest_items.append(model_name)
    
    # --- 从 uploads 目录获取 (保持不变) ---
    upload_dir = "./uploads"
    upload_items = []
    if os.path.exists(upload_dir):
        for item in os.listdir(upload_dir):
            item_path = os.path.join(upload_dir, item)
            if os.path.isfile(item_path):
                # 提取文件名（不含扩展名）作为模型名
                base_name = os.path.splitext(item)[0]
                upload_items.append(base_name)
    
    # 合并两个列表
    all_models = latest_items + upload_items
    logger.info(f"Listing models: {all_models}") 
    return {"code": 200, "data": all_models}

UPLOAD_DIR = "/home/yhz/local_iot/banckend/uploads" # 模型上传目录
os.makedirs(UPLOAD_DIR, exist_ok=True) 
# ==========================================
# 原有逻辑保留：API 路由
# ==========================================
@app.get("/api/gpu/status")
async def get_gpu_status():
    try:
        return {"gpus": get_gpu_info()}
    except Exception as e:
        return {"gpus": [{"error": f"Internal API Error: {e}"}]}

@app.get("/api/datasets/folders")
async def list_dataset_folders():
    if not os.path.exists(DATASET_ROOT_PATH):
        return {"code": 200, "folders":[]}
    items = os.listdir(DATASET_ROOT_PATH)
    folders =[item for item in items if os.path.isdir(os.path.join(DATASET_ROOT_PATH, item))]
    return {"code": 200, "folders": folders}

@app.get("/api/datasets/files/{folder_name}")
async def list_dataset_files(folder_name: str):
    folder_path = os.path.join(DATASET_ROOT_PATH, folder_name)
    if not os.path.exists(folder_path):
        return {"code": 404, "msg": f"子目录不存在: {folder_name}", "files":[]}
    
    allowed_extensions = {".csv", ".tsv"}
    all_items = os.listdir(folder_path)
    files =[f for f in all_items if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in allowed_extensions]
    return {"code": 200, "files": files}

# 在 app.post("/api/train/start") 中添加参数
@app.post("/api/train/start")
async def start_training(
    background_tasks: BackgroundTasks, 
    folder_name: str = Form(...),
    file_names: str = Form(...),
    train_ratio: float = Form(...),
    label_column: str = Form(...),
    selected_models: str = Form(...)  # 接收前端传递的 JSON 字符串
):
    folder_path = os.path.join(DATASET_ROOT_PATH, folder_name)
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"文件夹不存在: {folder_path}")
    
    requested_files = [f.strip() for f in file_names.split(",") if f.strip()]
    if not requested_files:
        all_items = os.listdir(folder_path)
        requested_files =[f for f in all_items if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in {".csv", ".tsv"}]
    
    file_paths =[os.path.join(folder_path, fname) for fname in requested_files if os.path.exists(os.path.join(folder_path, fname))]
    if not file_paths:
        raise HTTPException(status_code=400, detail="没有有效的文件路径用于训练。")

    # 解析选中的模型
    try:
        selected_models_list = json.loads(selected_models)
        if not isinstance(selected_models_list, list):
            raise ValueError("selected_models 必须是一个列表")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"selected_models 格式错误: {e}")

    task_id = str(uuid.uuid4())
    # 将 selected_models_list 传递给后台任务
    background_tasks.add_task(run_hybrid_pipeline, task_id, file_paths, train_ratio, label_column, selected_models_list)
    return {"code": 200, "task_id": task_id}
@app.post("/api/market/custom_upload")
async def upload_custom_model(file: UploadFile = File(...), model_name: str = Form(...)):
    """
    上传自定义模型文件到指定模型目录。
    文件会被保存到 /home/yhz/local_iot/banckend/uploads/{model_name}.{ext}
    """
    try:
        if not model_name or not file.filename:
            raise HTTPException(status_code=400, detail="模型名称和文件不能为空")

        # 验证文件扩展名
        allowed_extensions = {".pkl", ".h5", ".pt", ".pth", ".joblib"}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"不允许的文件类型: {file_ext}. 支持的类型: {', '.join(allowed_extensions)}")

        # 构建目标文件路径 (直接在 UPLOAD_DIR 下)
        # 文件名为 {model_name}.{ext}
        target_file_path = os.path.join(UPLOAD_DIR, f"{model_name}{file_ext}")

        # 检查文件是否已存在
        if os.path.exists(target_file_path):
            # 生成带时间戳的新文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename_with_ext = f"{model_name}_{timestamp}{file_ext}"
            target_file_path = os.path.join(UPLOAD_DIR, new_filename_with_ext)
            logger.info(f"Custom upload file exists, renamed to: {new_filename_with_ext}")

        # 保存上传的文件
        with open(target_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"✅ 自定义模型上传成功: {model_name}, 文件路径: {target_file_path}")

        return JSONResponse(
            content={
                "code": 200,
                "msg": f"自定义模型 '{model_name}' 上传成功",
                "data": {"saved_path": target_file_path}
            }
        )

    except HTTPException:
        # 重新抛出 HTTPException，FastAPI 会处理
        raise
    except Exception as e:
        logger.error(f"❌ 自定义模型上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.get("/api/train/status/{task_id}")
async def get_train_status(task_id: str):
    task_file = f"./storage/tasks/{task_id}.json"
    if not os.path.exists(task_file):
        return {"code": 404, "data": {"status": "not_found"}}
    
    try:
        with open(task_file, "r") as f:
            data = json.load(f)
        return {"code": 200, "data": data}
    except json.JSONDecodeError:
        # 文件读写冲突时返回 running 让前端继续等待
        return {"code": 200, "data": {"status": "running"}}

@app.post("/api/deploy")
async def deploy_model(model_name: str = Form(...)):
    """
    部署模型：直接查找 ./storage/trained_models/latest 和 ./uploads 目录下的文件/文件夹
    """
    try:
        logger.info(f"--- Deploying model (new logic): {model_name} ---")
        
        model_path = None
        model_type = None

        # --- 1. 在 latest 目录下查找 ---
        latest_model_dir = "./storage/trained_models/latest"
        if os.path.exists(latest_model_dir):
            # --- 修改：查找直接的 pkl 文件 ---
            # 首先检查是否是 PyCaret 基础模型名，如果是，则查找对应的 .pkl
            if model_name in ['rf', 'nb']: 
                direct_pkl_path = os.path.join(latest_model_dir, f"{model_name}.pkl")
                if os.path.exists(direct_pkl_path):
                    model_path = direct_pkl_path
                    model_type = 'pycaret_direct'
                    logger.info(f"Found PyCaret direct file: {model_path}")
            
            # 如果没找到 PyCaret 文件，或者不是 PyCaret 模型，查找子目录 (Ludwig 模型等)
            if model_path is None:
                subdir_path = os.path.join(latest_model_dir, model_name)
                if os.path.isdir(subdir_path):
                    # 检查是否有 Ludwig 的 model 子目录
                    ludwig_model_dir = os.path.join(subdir_path, "model")
                    if os.path.exists(ludwig_model_dir):
                        model_path = ludwig_model_dir
                        model_type = 'ludwig_subdir'
                        logger.info(f"Found Ludwig model dir: {model_path}")
                    # 如果 model 子目录不存在，但 model_name 是 PyCaret 模型，可能意味着文件在错误的位置
                    # 但这不太可能发生，因为我们优先查找了 .pkl 文件
                    # 除非有其他逻辑创建了名为 rf/ 或 nb/ 的空文件夹，这种情况可以忽略或记录警告

        # --- 2. 如果 latest 中没找到，在 uploads 目录下查找 ---
        if model_path is None:
            upload_dir = "./uploads"
            if os.path.exists(upload_dir):
                # 查找直接文件 (去掉扩展名匹配，然后加上扩展名验证)
                # 优先查找 .pkl 文件
                upload_pkl_path = os.path.join(upload_dir, f"{model_name}.pkl")
                if os.path.exists(upload_pkl_path):
                    model_path = upload_pkl_path
                    model_type = 'generic_upload_pkl' 
                    logger.info(f"Found uploaded .pkl file: {model_path}")

        # --- 检查是否找到模型 ---
        if model_path is None or not os.path.exists(model_path):
            error_msg = f"未找到模型 '{model_name}' 在 ./storage/trained_models/latest 或 ./uploads 目录下。"
            logger.error(error_msg)
            return {"code": 500, "msg": error_msg}

        # --- 加载模型 ---
        global DEPLOYED_MODEL
        if model_type == 'pycaret_direct':
            try:
                # --- 修改：load_model 的路径需要去掉 .pkl 后缀 ---
                # model_path 是 ./storage/trained_models/latest/rf.pkl
                # load_model 需要的是 ./storage/trained_models/latest/rf
                model_path_for_load = model_path.replace('.pkl', '')
                
                from pycaret.classification import load_model
                loaded_model = load_model(model_path_for_load) # 传入去掉 .pkl 的路径
                DEPLOYED_MODEL = (loaded_model, 'pycaret', model_path)
                logger.info(f"✅ PyCaret 模型部署成功: {model_name}, Path: {model_path}")
            except Exception as e:
                logger.error(f"❌ 加载 PyCaret 模型失败: {e}")
                logger.error(traceback.format_exc())
                return {"code": 500, "msg": f"加载 PyCaret 模型失败: {str(e)}"}
        
        elif model_type == 'ludwig_subdir':
            try:
                # 假设 LudwigModel.load 需要的是包含 model.yaml, config.yaml 等的 'model' 目录路径
                # model_path 此时就是那个 'model' 目录的路径
                loaded_model = LudwigModel.load(model_path, backend='local')
                DEPLOYED_MODEL = (loaded_model, 'ludwig', model_path)
                logger.info(f"✅ Ludwig 模型部署成功: {model_name}, Path: {model_path}")
            except Exception as e:
                logger.error(f"❌ 加载 Ludwig 模型失败: {e}")
                logger.error(traceback.format_exc())
                return {"code": 500, "msg": f"加载 Ludwig 模型失败: {str(e)}"}
        
        elif model_type == 'generic_upload_pkl': 
            try:
                logger.info(f"Attempting to load generic .pkl model from {model_path}")
                # Try joblib first
                try:
                    loaded_model = joblib.load(model_path)
                    logger.info(f"Loaded with joblib from {model_path}")
                except Exception as e_joblib:
                    logger.warning(f"Joblib load failed for {model_path}: {e_joblib}")
                    # If joblib fails, try pickle
                    with open(model_path, 'rb') as f:
                        loaded_model = pickle.load(f)
                    logger.info(f"Loaded with pickle from {model_path}")
                
                DEPLOYED_MODEL = (loaded_model, 'generic_pkl', model_path)
                logger.info(f"✅ 上传的 .pkl 模型部署成功: {model_name}, Path: {model_path}")
            except Exception as e:
                logger.error(f"❌ 加载上传的 .pkl 模型失败: {e}")
                logger.error(traceback.format_exc())
                return {"code": 500, "msg": f"加载上传的 .pkl 模型失败: {str(e)}"}
        
        else:
            # 理论上不应到达这里
            error_msg = f"未知的模型类型: {model_type} for path: {model_path}"
            logger.error(error_msg)
            return {"code": 500, "msg": error_msg}

        return {"code": 200, "msg": f"模型 {model_name} 部署成功!"}

    except Exception as e:
        logger.error(f"❌ 模型部署失败 (顶层): {str(e)}")
        logger.error(traceback.format_exc())
        return {"code": 500, "msg": f"模型部署失败: {str(e)}"}
# ==========================================
# 【核心修改】多属性在线推理预测接口
# ==========================================
@app.post("/api/predict")
async def predict(payload: Dict[str, Any] = Body(...)):
    if DEPLOYED_MODEL is None:
        return {"error": "No model deployed. Please deploy a model first."}
    
    loaded_model, model_type, _ = DEPLOYED_MODEL
    try:
        # 动态接收前端任意 JSON (哪怕有十几个属性列) 并转为 DataFrame
        df_new = pd.DataFrame([payload])
        
        if model_type == 'pycaret':
            from pycaret.classification import predict_model
            prediction_df = predict_model(loaded_model, data=df_new)
            pred_val = prediction_df['prediction_label'].iloc[0]
            return {"prediction": str(pred_val)}
        else:
            pred_df, _ = loaded_model.predict(df_new)
            pred_col =[c for c in pred_df.columns if c.endswith('_predictions')][0]
            pred_val = pred_df[pred_col].iloc[0]
            return {"prediction": str(pred_val)}
    except Exception as e:
        logger.error(traceback.format_exc())
        return {"error": f"Prediction failed: {e}"}

# ==========================================
# 【新增模块】获取无标签预测数据集目录和文件
# ==========================================
@app.get("/api/predict_datasets/folders")
async def list_predict_dataset_folders():
    """获取 /home/yhz/iot_yuceshuju 目录下的所有子文件夹"""
    if not os.path.exists(PREDICT_DATASET_ROOT_PATH):
        return {"code": 200, "folders":[]}
    items = os.listdir(PREDICT_DATASET_ROOT_PATH)
    folders =[item for item in items if os.path.isdir(os.path.join(PREDICT_DATASET_ROOT_PATH, item))]
    return {"code": 200, "folders": folders}

@app.get("/api/predict_datasets/files/{folder_name}")
async def list_predict_dataset_files(folder_name: str):
    """获取预测子目录下的无标签文件列表"""
    folder_path = os.path.join(PREDICT_DATASET_ROOT_PATH, folder_name)
    if not os.path.exists(folder_path):
        return {"code": 404, "msg": f"预测子目录不存在: {folder_name}", "files":[]}
    
    allowed_extensions = {".csv", ".tsv", ".xlsx", ".xls"}
    all_items = os.listdir(folder_path)
    files =[f for f in all_items if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in allowed_extensions]
    return {"code": 200, "files": files}

# ==========================================
# 【新增模块】针对无标签文件的批量推理接口 (通过指定路径)
# ==========================================
@app.post("/api/predict/batch_file")
async def predict_batch_from_file(
    folder_name: str = Form(...),
    file_name: str = Form(...)
):
    if DEPLOYED_MODEL is None:
        return {"code": 400, "msg": "没有任何模型被部署运行！请先在排行榜点击一键部署。"}
    
    loaded_model, model_type, model_path = DEPLOYED_MODEL
    
    file_path = os.path.join(PREDICT_DATASET_ROOT_PATH, folder_name, file_name)
    if not os.path.exists(file_path):
        return {"code": 404, "msg": f"待预测文件不存在: {file_path}"}
        
    try:
        # 1. 动态读取无标签文件
        file_ext = file_path.split('.')[-1].lower()
        if file_ext in ['xls', 'xlsx']:
            df_new = pd.read_excel(file_path)
        elif file_ext == 'tsv':
            df_new = pd.read_csv(file_path, sep='\t')
        else:
            df_new = pd.read_csv(file_path)
            
        df_new = df_new.dropna(how='all') # 清洗全空行
        if df_new.empty:
            return {"code": 400, "msg": "文件内容为空。"}
            
        # 2. 走部署模型的推理逻辑
        if model_type == 'pycaret':
            from pycaret.classification import predict_model
            prediction_df = predict_model(loaded_model, data=df_new)
            df_new['模型预测结果'] = prediction_df['prediction_label']
        else:
            pred_df, _ = loaded_model.predict(df_new)
            pred_col = [c for c in pred_df.columns if c.endswith('_predictions')][0]
            df_new['模型预测结果'] = pred_df[pred_col]
            
        # 3. 将 DataFrame 转回字典数组返回给前端渲染 (处理 NaN)
        df_new = df_new.replace({np.nan: None}) 
        records = df_new.to_dict(orient='records')
        
        return {
            "code": 200, 
            "msg": f"批量预测完成，共诊断 {len(records)} 条数据！",
            "data": records,
            "model_used": model_path.split('/')[-1] if model_type=='pycaret' else model_path.split('/')[-2]
        }
        
    except Exception as e:
        logger.error(traceback.format_exc())
        return {"code": 500, "msg": f"批量预测失败，请检查该无标签文件的列特征是否与训练时一致！错误: {str(e)}"}

# ==========================================
# 【新增模块】通过上传文件进行批量推理接口
# ==========================================
@app.post("/api/predict/batch_upload")
async def predict_batch_from_uploaded_file(
    file: UploadFile = File(...)
):
    if DEPLOYED_MODEL is None:
        return {"code": 400, "msg": "没有任何模型被部署运行！请先在排行榜点击一键部署。"}
    
    loaded_model, model_type, model_path = DEPLOYED_MODEL

    # 1. 保存上传的临时文件
    temp_file_path = os.path.join(PREDICT_DATASET_ROOT_PATH, f"temp_{file.filename}")
    try:
        contents = await file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(contents)
        logger.info(f"Uploaded temporary file: {temp_file_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {str(e)}")
        return {"code": 500, "msg": f"Failed to save uploaded file: {str(e)}"}

    try:
        # 2. 动态读取无标签文件
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext in ['xls', 'xlsx']:
            df_new = pd.read_excel(temp_file_path)
        elif file_ext == 'tsv':
            df_new = pd.read_csv(temp_file_path, sep='\t')
        else:
            df_new = pd.read_csv(temp_file_path)
            
        df_new = df_new.dropna(how='all') # 清洗全空行
        if df_new.empty:
            os.remove(temp_file_path) # 清理临时文件
            return {"code": 400, "msg": "上传的文件内容为空。"}
            
        # 3. 走部署模型的推理逻辑
        if model_type == 'pycaret':
            from pycaret.classification import predict_model
            prediction_df = predict_model(loaded_model, data=df_new)
            df_new['模型预测结果'] = prediction_df['prediction_label']
        else:
            pred_df, _ = loaded_model.predict(df_new)
            pred_col = [c for c in pred_df.columns if c.endswith('_predictions')][0]
            df_new['模型预测结果'] = pred_df[pred_col]
            
        # 4. 将 DataFrame 转回字典数组返回给前端渲染 (处理 NaN)
        df_new = df_new.replace({np.nan: None}) 
        records = df_new.to_dict(orient='records')

        # 5. 删除临时文件
        os.remove(temp_file_path)
        logger.info(f"Deleted temporary file: {temp_file_path}")
        
        return {
            "code": 200, 
            "msg": f"批量预测完成，共诊断 {len(records)} 条数据！",
            "data": records,
            "model_used": model_path.split('/')[-1] if model_type=='pycaret' else model_path.split('/')[-2]
        }
        
    except Exception as e:
        # 确保即使预测失败也删除临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.warning(f"Deleted temporary file after error: {temp_file_path}")
        logger.error(traceback.format_exc())
        return {"code": 500, "msg": f"批量预测失败，请检查该无标签文件的列特征是否与训练时一致！错误: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning", reload_excludes=["storage/*"])