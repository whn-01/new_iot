import os
import uuid
import json
import pandas as pd
import shutil
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Body
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
from typing import Dict, Any
import re

# 将 Python 默认的 1000 层递归限制调高到 100000 层，防止 Ray 打包时堆栈溢出崩溃！
sys.setrecursionlimit(100000)

# --- 配置日志 ---
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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

# --- 初始化 Ray ---
ray.init(num_cpus=2, ignore_reinit_error=True)

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

LUDWIG_ALGORITHMS = {
    "TextCNN": {"type": "parallel_cnn"},
    "Bi-LSTM": {"type": "rnn", "cell_type": "lstm", "bidirectional": True},
    "Bi-GRU":  {"type": "rnn", "cell_type": "gru", "bidirectional": True},
    "Vanilla-RNN": {"type": "rnn", "cell_type": "rnn"},
    "FastText-Embed": {"type": "embed"}
}

# ==========================================
# 【核心修改】自动特征检测函数
# ==========================================
def detect_column_types(df, target_col):
    """自动检测 DataFrame 中各列的类型（文本、数值、类别），并强制排除 target_col"""
    column_types = {}
    id_like_name_pattern = re.compile(r"(?:^|_)(id|uuid|identifier|sn|serial|imei|imsi|mac)(?:$|_)")
    time_like_name_pattern = re.compile(r"(?:^|_)(time|timestamp|date|created|updated)(?:$|_)")
    for col in df.columns:
        # ⭐⭐⭐ 关键修复：强制跳过目标列，防止其被用作输入特征导致数据泄露
        if col == target_col:
            column_types[col] = 'skip'
            continue

        series = df[col].dropna()
        if series.empty:
            column_types[col] = 'skip'
            continue

        dtype = series.dtype
        unique_count = series.nunique()
        normalized_col = str(col).strip().lower()

        # 常量列无信息量，且数值常量列会触发 Ludwig zscore 异常
        if unique_count <= 1:
            column_types[col] = 'skip'
            continue

        if dtype == 'object' or dtype.name == 'string':
            # 明显ID/时间字段直接跳过
            if id_like_name_pattern.search(normalized_col) or time_like_name_pattern.search(normalized_col):
                column_types[col] = 'skip'
                continue

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
            if np.issubdtype(dtype, np.integer) and unique_count <= 20:
                column_types[col] = 'category'
            else:
                column_types[col] = 'number'
        else:
            column_types[col] = 'category'
    return column_types
# ==========================================
# 后台异步训练引擎 (全动态多模态融合版) - 增加调试日志
# ==========================================
def run_hybrid_pipeline(task_id: str, file_paths: list, train_ratio: float, label_column: str):
    model_root = "./storage/trained_models/latest"
    if os.path.exists(model_root):
        shutil.rmtree(model_root)   # ⭐直接删整个 latest（最干净）
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

    # --- 新增调试日志 ---
    print(f"[DEBUG] Task {task_id} - Target column: '{label_column}'")
    print(f"[DEBUG] Task {task_id} - Detected column types: {column_types}")
    input_feature_names = [col for col, typ in column_types.items() if typ != 'skip']
    print(f"[DEBUG] Task {task_id} - Potential input features: {input_feature_names}")
    
    if label_column in input_feature_names:
        logger.error(f"[CRITICAL ERROR] Task {task_id}: Label column '{label_column}' is included in input features! This indicates a bug in detect_column_types.")
        with open(task_file, "w") as f:
            json.dump({"status": "failed", "msg": f"Internal error: Label column was incorrectly added as an input feature.", "models":[]}, f)
        return
    else:
        logger.info(f"[INFO] Task {task_id}: Label column '{label_column}' correctly excluded from input features.")
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
                "n_jobs": 1, "use_gpu": False, "fold": 3
            }
            setup(**setup_kwargs)

            for model_code in ['rf', 'nb']:
                model_name = f"{model_code.upper()}_AutoFeat"
                try:
                    trained_model = create_model(model_code, verbose=False)
                    tuned_model = tune_model(trained_model, optimize='F1', n_iter=10, verbose=False)
                    metrics_df = pull()

                    f1_score = metrics_df.loc['Mean', 'F1']
                    accuracy = metrics_df.loc['Mean', 'Accuracy']
                    out_path = f"./storage/trained_models/latest/{model_code.lower()}"
                    save_model(tuned_model, out_path)

                    leaderboard.append({
                        "model_name": model_name, "f1_score": round(float(f1_score), 4),
                        "accuracy": round(float(accuracy), 4), "model_path": out_path + ".pkl"
                    })
                except Exception as e:
                    logger.error(f"❌ {model_name} 训练异常: {str(e)}")

        # ================== 引擎 2：Ludwig 深度学习超参搜索 ==================
        write_task("running", "开始 Ludwig 训练")
        for base_model_name, base_encoder_config in LUDWIG_ALGORITHMS.items():
            name = f"{base_model_name}_Hyperopt_AutoFeat"
            write_task("running", f"正在训练 Ludwig 模型: {name}")
            temp_dataset_path = f"./storage/datasets/temp_{task_id}_{uuid.uuid4()}.csv"
            df.to_csv(temp_dataset_path, index=False)

            # 动态组装当前模型的输入特征
                        # --- 防御性修复：再次确保 label_column 不被加入 input_features ---
            input_features = []
            for col_name, col_type in column_types.items():
                if col_type == 'skip' or col_name == label_column: # ⭐ 额外检查
                    continue
                elif col_type == 'text':
                    input_features.append({
                        "name": col_name, "type": "text", "encoder": base_encoder_config,
                        "preprocessing": {"tokenizer": "characters"} # 中文必须按字切分
                    })
                elif col_type == 'number':
                    input_features.append({"name": col_name, "type": "number"})
                elif col_type == 'category':
                    input_features.append({"name": col_name, "type": "category"})

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
                logger.critical(f"🚨 [CRITICAL ERROR] Task {task_id} - {base_model_name}: Label column '{label_column}' is STILL present in final input_features after ALL filters! Aborting this model.")
                continue # 如果发生，跳过此模型训练
            else:
                logger.info(f"✅ [INFO] Task {task_id} - {base_model_name}: Label column '{label_column}' correctly excluded from final input features. Proceeding.")
            # --- END 调试日志 ---
            # --- END 调试日志 ---
            hyperopt_params = {
                "trainer.learning_rate": {"space": "choice", "categories":[0.001, 0.0005]},
                "trainer.batch_size": {"space": "choice", "categories": [8, 16]}
            }
            if first_text_col:
                hyperopt_params[f"input_features.{first_text_col}.encoder.dropout"] = {"space": "uniform", "lower": 0.0, "upper": 0.3}

            config = {
                "input_features": input_features,
                "output_features":[{"name": label_column, "type": "category"}],
                "backend": {"type": "local"}, # 强行拦截 Ray Data 防止 OOM
                "preprocessing": {
                    "split": {
                        "type": "random",
                        "probabilities": [train_ratio, (1.0 - train_ratio) / 2.0, (1.0 - train_ratio) / 2.0]
                    }
                },
                "trainer": {
                    # 控制单模型训练时长，避免“看起来一直不结束”
                    "epochs": 2,
                    "early_stop": 2
                },
                "hyperopt": {
                    # 当前 Ludwig 版本不支持 f1_score_macro 作为 hyperopt metric，改用兼容指标避免 schema 校验失败
                    "goal": "maximize", "metric": "accuracy_micro", "output_feature": label_column,
                    "search_alg": {"type": "hyperopt"},
                    "executor": {"type": "ray", "num_samples": 1, "max_concurrent_trials": 1},
                    "parameters": hyperopt_params
                }
            }
            out_dir = f"./storage/trained_models/latest/{name}"
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir, ignore_errors=True)
            try:
                hyperopt_results = hyperopt(config=config, dataset=temp_dataset_path, output_directory=out_dir)
                # ================== ⭐ 固定模型输出（核心修改） ==================
                import glob

                try:
                    # 找到所有 trial
                    trial_dirs = glob.glob(os.path.join(out_dir, "hyperopt", "trial_*"))
                    if not trial_dirs:
                        raise Exception("没有找到 trial 目录")

                    # 取最新 trial
                    latest_trial = max(trial_dirs, key=os.path.getctime)

                    # 找 checkpoint
                    checkpoint_dirs = glob.glob(os.path.join(latest_trial, "checkpoint_*"))
                    if not checkpoint_dirs:
                        raise Exception("没有找到 checkpoint")

                    # 取最新 checkpoint（通常就是最终模型）
                    best_checkpoint = max(checkpoint_dirs, key=os.path.getctime)

                    # 源模型路径
                    src_model = os.path.join(best_checkpoint, "model")

                    # ⭐ 固定输出路径（关键）
                    dst_model = os.path.join(out_dir, "model")

                    # 删除旧模型
                    if os.path.exists(dst_model):
                        shutil.rmtree(dst_model, ignore_errors=True)

                    # 拷贝最终模型
                    shutil.copytree(src_model, dst_model)

                    logger.info(f"✅ 已提取最终模型到: {dst_model}")

                except Exception as e:
                    logger.error(f"❌ 提取最终模型失败: {str(e)}")
                                    # ================== ⭐ 清理 hyperopt（节省空间） ==================
                try:
                    hyperopt_dir = os.path.join(out_dir, "hyperopt")
                    if os.path.exists(hyperopt_dir):
                        shutil.rmtree(hyperopt_dir, ignore_errors=True)
                        logger.info("🧹 已清理 hyperopt 临时文件")
                except Exception as e:
                    logger.warning(f"清理 hyperopt 失败: {str(e)}")
                best_trial_result = hyperopt_results.experiment_analysis.best_result
                if not isinstance(best_trial_result, dict): raise TypeError("返回格式异常")

                eval_stats_raw = best_trial_result.get('eval_stats', {})
                eval_stats = json.loads(eval_stats_raw) if isinstance(eval_stats_raw, str) else eval_stats_raw

                overall_stats = eval_stats.get(label_column, {}).get('overall_stats', {})
                best_f1 = (
                    overall_stats.get('avg_f1_score_macro')
                    or overall_stats.get('f1_score_macro')
                    or overall_stats.get('avg_f1_score_weighted')
                    or overall_stats.get('f1_score')
                    or overall_stats.get('avg_f1_score_micro')
                    or 0.0
                )
                best_acc = overall_stats.get('accuracy') or overall_stats.get('accuracy_micro') or 0.0

                leaderboard.append({
                    "model_name": name, "f1_score": round(float(best_f1), 4),
                    "accuracy": round(float(best_acc), 4), "model_path": out_dir
                })
                write_task("running", f"模型完成: {name}", leaderboard)
            except Exception as e:
                logger.error(f"❌ {name} 异常: {str(e)}")
                write_task("running", f"模型失败并跳过: {name}，原因: {str(e)}", leaderboard)
            finally:
                if os.path.exists(temp_dataset_path): os.remove(temp_dataset_path)
        # 统一写入排行榜
        leaderboard = sorted(leaderboard, key=lambda x: x['f1_score'], reverse=True)
        write_task("completed", "训练完成", leaderboard)

    except Exception as e:
        logger.error(traceback.format_exc())
        write_task("failed", str(e), [])
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

@app.post("/api/train/start")
async def start_training(
    background_tasks: BackgroundTasks, 
    folder_name: str = Form(...),
    file_names: str = Form(...),
    train_ratio: float = Form(...),
    label_column: str = Form(...)  # 前端传入的自定义标签列
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

    task_id = str(uuid.uuid4())
    background_tasks.add_task(run_hybrid_pipeline, task_id, file_paths, train_ratio, label_column)
    return {"code": 200, "task_id": task_id}

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
async def deploy_model_from_table(model_name: str = Form(...), model_id: str = Form(None)):
    latest_task_files = glob.glob("./storage/tasks/*.json")
    if not latest_task_files:
         return {"code": 500, "msg": "未找到任何训练结果。"}
    
    latest_task_file = max(latest_task_files, key=os.path.getctime)
    with open(latest_task_file, "r") as f:
        task_data = json.load(f)
    
    models = task_data.get("models",[])
    model_to_deploy = next((m for m in models if m["model_name"] == model_name), None)
    if not model_to_deploy:
        return {"code": 500, "msg": f"未找到模型 '{model_name}'。"}
    
    model_path = model_to_deploy["model_path"]
    
    try:
        global DEPLOYED_MODEL
        if "pycaret" in model_path:
            from pycaret.classification import load_model
            loaded_model = load_model(model_path.replace('.pkl', ''))
            DEPLOYED_MODEL = (loaded_model, 'pycaret', model_path)
        else:
            loaded_model = LudwigModel.load(model_path, backend='local')
            DEPLOYED_MODEL = (loaded_model, 'ludwig', model_path)
        return {"code": 200, "msg": f"{model_name} 部署成功！"}
    except Exception as e:
        return {"code": 500, "msg": f"加载失败: {str(e)}"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning", reload_excludes=["storage/*"])