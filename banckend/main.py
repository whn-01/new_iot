# 现在代码如下，给出修改后的代码
import os
import uuid
import json
import pandas as pd
import shutil
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
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

# 将 Python 默认的 1000 层递归限制调高到 100000 层，防止 Ray 打包时堆栈溢出崩溃！
sys.setrecursionlimit(100000)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gpu_info_via_nvidia_smi():
    """
    通过 nvidia-smi 命令获取 GPU 信息，作为 pynvml 的替代方案，
    可能更能抵抗其他进程对 GPU 的干扰。
    """
    gpu_info = []
    try:
        # 使用 nvidia-smi 查询所有 GPU 的详细信息
        command = [
            "nvidia-smi", 
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw", 
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if not line:
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 9: # index, name, total_mem, used_mem, free_mem, gpu_util, mem_util, temp, power
                try:
                    index = int(parts[0])
                    name = parts[1]
                    total_memory_mb = int(parts[2]) * 1024 * 1024 # MB to Bytes
                    used_memory_mb = int(parts[3]) * 1024 * 1024
                    free_memory_mb = int(parts[4]) * 1024 * 1024
                    gpu_util = int(parts[5]) if parts[5] != '' else 0
                    mem_util = int(parts[6]) if parts[6] != '' else 0
                    temperature = int(parts[7]) if parts[7] != '' else None
                    power_draw = float(parts[8]) if parts[8] != '' else None
                    
                    gpu_info.append({
                        "index": index,
                        "name": name,
                        "total_memory": total_memory_mb,
                        "used_memory": used_memory_mb,
                        "free_memory": free_memory_mb,
                        "memory_util_percent": mem_util,
                        "gpu_util_percent": gpu_util,
                        "memory_bandwidth_util_percent": mem_util, # SMI 报告的 memory util 通常是带宽利用率
                        "temperature": temperature,
                        "power_usage_watts": power_draw
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"解析 nvidia-smi 输出行失败: {line}, 错误: {e}")
            else:
                logger.warning(f"nvidia-smi 输出格式不匹配: {line}")
                
    except subprocess.CalledProcessError as e:
        logger.warning(f"nvidia-smi 命令执行失败: {e}")
        gpu_info = [{"error": "nvidia-smi command failed"}]
    except FileNotFoundError:
        logger.warning("nvidia-smi 命令未找到，可能没有安装 NVIDIA 驱动或 nvidia-smi 不在 PATH 中。")
        gpu_info = [{"error": "nvidia-smi command not found"}]
    except Exception as e:
        logger.error(f"获取 GPU 信息时发生未知错误: {e}")
        gpu_info = [{"error": f"Unknown error when getting GPU info: {e}"}]
    
    return gpu_info

def get_gpu_info():
    """
    获取 GPU 信息。优先使用 pynvml，失败时回退到 nvidia-smi。
    """
    # 首先尝试 pynvml
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        gpu_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # --- 修改此处 ---
            raw_name = pynvml.nvmlDeviceGetName(handle)
            # 检查 raw_name 是否已经是字符串，如果不是则进行解码
            if isinstance(raw_name, bytes):
                name = raw_name.decode('utf-8')
            else:
                # 如果 raw_name 是字符串或其他类型（根据新API）
                name = str(raw_name) 
            # --- 修改结束 ---
            
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
                "index": i,
                "name": name,
                "total_memory": total_memory,
                "used_memory": used_memory,
                "free_memory": total_memory - used_memory,
                "memory_util_percent": round(memory_util, 2),
                "gpu_util_percent": gpu_util,
                "memory_bandwidth_util_percent": util_rates.memory, # 注意：这里也应确保是整数
                "temperature": temperature,
                "power_usage_watts": power_status
            })

        pynvml.nvmlShutdown()
        return gpu_info
    except pynvml.NVMLError as e:
        logger.warning(f"pynvml 获取 GPU 信息失败: {e}. 尝试使用 nvidia-smi...")
        # pynvml 失败后，尝试 nvidia-smi
        return get_gpu_info_via_nvidia_smi()
    except Exception as e:
        logger.error(f"pynvml 初始化或执行过程中发生未知错误: {e}. 尝试使用 nvidia-smi...")
        # 发生其他错误也回退
        return get_gpu_info_via_nvidia_smi()
# --- 新增：限制 Ray 资源 ---
import ray
ray.init(num_cpus=2, ignore_reinit_error=True)
# --- END 新增 ---

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化存储目录
for d in ["datasets", "tasks", "custom_models", "trained_models/latest"]:
    os.makedirs(f"./storage/{d}", exist_ok=True)

# 全局变量
DEPLOYED_MODEL = None

# 设置数据集根目录
DATASET_ROOT_PATH = "/home/yhz/iot_shujuji"

# ==========================================
# 定义 5 个 Ludwig 深度学习算法基线
# ==========================================
LUDWIG_ALGORITHMS = {
    "TextCNN": {"type": "parallel_cnn"},
    "Bi-LSTM": {"type": "rnn", "cell_type": "lstm", "bidirectional": True},
    "Bi-GRU":  {"type": "rnn", "cell_type": "gru", "bidirectional": True},
    "Vanilla-RNN": {"type": "rnn", "cell_type": "rnn"},
    "FastText-Embed": {"type": "embed"}
}

# ==========================================
# 后台异步训练引擎 (融合版 - 已修改，支持合并多个文件)
# ==========================================

def run_hybrid_pipeline(task_id: str, file_paths: list, train_ratio: float): # 参数改为 file_paths 列表
    """双引擎混合训练管道：PyCaret (传统ML) + Ludwig (深度学习)"""
    
    task_file = f"./storage/tasks/{task_id}.json"
    
    # 初始化任务状态
    with open(task_file, "w") as f:
        json.dump({"status": "running", "models": []}, f)
        
    leaderboard = []
    
    # 读取并合并所有选定的文件
    dfs = []
    for fp in file_paths:
        if not os.path.exists(fp):
            error_msg = f"数据集文件不存在: {fp}"
            logger.error(error_msg)
            with open(task_file, "w") as f:
                json.dump({"status": "failed", "msg": error_msg, "models": []}, f)
            return
            
        file_ext = fp.split('.')[-1].lower()
        if file_ext == 'tsv':
            sep = '\t'
        elif file_ext == 'csv':
            sep = ','
        else:
            sep = ','
            
        try:
            df_temp = pd.read_csv(fp, sep=sep)
            dfs.append(df_temp)
        except Exception as e:
            error_msg = f"读取文件失败 {fp}: {str(e)}"
            logger.error(error_msg)
            with open(task_file, "w") as f:
                json.dump({"status": "failed", "msg": error_msg, "models": []}, f)
            return

    if not dfs:
        error_msg = "没有找到任何有效的数据文件进行训练。"
        logger.error(error_msg)
        with open(task_file, "w") as f:
            json.dump({"status": "failed", "msg": error_msg, "models": []}, f)
        return

    # 合并所有 DataFrame
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"成功合并 {len(file_paths)} 个文件，总行数: {len(df)}")

    try:
        # 【新增】打印列名和前几行数据，用于调试
        print("--- 合并后数据集信息 (用于调试) ---")
        print(f"列名: {list(df.columns)}")
        print(f"数据形状: {df.shape}")
        print(f"前5行:\n{df.head()}")
        print("-----------------------------")
        
        #  检查并处理列名
        required_columns = ['text', 'label']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
             raise ValueError(f"数据集中缺少必需的列: {missing_cols}. 当前列名为: {list(df.columns)}. 请确保您的数据集包含 'text' 和 'label' 列。")

        # ==========================================
        # 引擎 1：PyCaret 处理传统机器学习 (极速)
        # ==========================================
        print("       [引擎 1] 启动 PyCaret 传统机器学习流水线...")
        logger.info("       [引擎 1] 启动 PyCaret 传统机器学习流水线...")
        
        setup(data=df, target='label', train_size=train_ratio, text_features=['text'], verbose=False, session_id=42)
        
        for model_code in ['rf', 'nb']:
            model_name = f"{model_code.upper()}"
            logger.info(f"         训练 {model_name}")
            trained_model = create_model(model_code, verbose=False)
            tuned_model = tune_model(trained_model, optimize='F1', n_iter=10, verbose=False)
            metrics_df = pull()
            
            f1_score = metrics_df.loc['Mean', 'F1']
            accuracy = metrics_df.loc['Mean', 'Accuracy']
            
            out_path = f"./storage/trained_models/latest/{model_code.lower()}"
            save_model(trained_model, out_path)
            
            leaderboard.append({
                "model_name": model_name,
                "f1_score": round(float(f1_score), 4),
                "accuracy": round(float(accuracy), 4),
                "model_path": out_path + ".pkl"
            })
            logger.info(f"  ✅ {model_name} - F1: {f1_score:.4f}, Acc: {accuracy:.4f}")

        # ==========================================
        # 引擎 2：Ludwig 超参数优化 (Hyperopt)
        # ==========================================
        print("       [引擎 2] 启动 Ludwig 超参数优化流水线...")
        logger.info("       [引擎 2] 启动 Ludwig 超参数优化流水线...")

        for base_model_name, base_encoder_config in LUDWIG_ALGORITHMS.items():
            name = f"{base_model_name}_Hyperopt"
            logger.info(f"       [引擎 2] 启动 Ludwig {base_model_name} Hyperopt 流水线...")

            # 为了给 Ludwig 使用，我们需要将合并后的 DataFrame 临时保存为一个文件
            temp_dataset_path = f"./storage/datasets/temp_{task_id}_{uuid.uuid4()}.csv"
            df.to_csv(temp_dataset_path, index=False)

            config = {
                "input_features": [{"name": "text", "type": "text", "encoder": base_encoder_config}],
                "output_features": [{"name": "label", "type": "category"}],
                "hyperopt": {
                    "goal": "maximize",
                    "metric": "accuracy",
                    "output_feature": "label",
                    "search_alg": {"type": "hyperopt"},
                    "executor": {
                        "type": "ray", 
                        "num_samples": 5,
                        "cpu_resources_per_trial": 1,
                    },
                    "parameters": {
                        "trainer.learning_rate": {"space": "choice", "categories":[0.001, 0.0001, 0.01]},
                        "trainer.batch_size": {"space": "choice", "categories": [16, 32, 64]},
                        "input_features.text.encoder.dropout": {"space": "uniform", "lower": 0.0, "upper": 0.3}
                    }
                }
            }

            out_dir = f"./storage/trained_models/latest/{name}"

            try:
                hyperopt_results = hyperopt(
                    config=config, 
                    dataset=temp_dataset_path, # 使用临时合并文件
                    output_directory=out_dir
                )

                best_trial_result = hyperopt_results.experiment_analysis.best_result
                if not isinstance(best_trial_result, dict):
                    raise TypeError(f"Ray 返回的搜索结果格式错误: {type(best_trial_result)}")
                
                eval_stats_raw = best_trial_result.get('eval_stats', {})
                if isinstance(eval_stats_raw, str):
                    eval_stats = json.loads(eval_stats_raw)
                else:
                    eval_stats = eval_stats_raw
                
                if not isinstance(eval_stats, dict):
                    raise TypeError(f"评测数据格式异常: {eval_stats}")
                
                label_metrics = eval_stats.get('label', {})
                overall_stats = label_metrics.get('overall_stats', {})
                
                best_f1 = (
                    overall_stats.get('avg_f1_score_micro') or
                    overall_stats.get('avg_f1_score_macro') or
                    overall_stats.get('avg_f1_score_weighted') or
                    overall_stats.get('f1_score') or 
                    0.0
                )
                best_accuracy = (
                    overall_stats.get('accuracy_micro') or 
                    overall_stats.get('accuracy') or
                    0.0
                )

                leaderboard.append({
                    "model_name": f"{name}",
                    "f1_score": round(float(best_f1), 4),
                    "accuracy": round(float(best_accuracy), 4),
                    "model_path": out_dir 
                })
                logger.info(f"  ✅ {name} (Hyperopt) - Best F1: {best_f1:.4f}, Best Acc: {best_accuracy:.4f}")
            except Exception as e:
                logger.error(f"❌ {name} (Hyperopt) 训练发生异常: {str(e)}")
                logger.error(traceback.format_exc())
                leaderboard.append({
                    "model_name": f"{name}",
                    "f1_score": 0.0,
                    "accuracy": 0.0,
                    "model_path": "",
                    "status": "failed"
                })
            finally:
                # 训练完成后删除临时文件
                if os.path.exists(temp_dataset_path):
                    os.remove(temp_dataset_path)
                    logger.info(f"已删除临时数据集文件: {temp_dataset_path}")


        # ==========================================
        # 收尾：统一合并榜单并写入文件
        # ==========================================
        leaderboard = sorted(leaderboard, key=lambda x: x['f1_score'], reverse=True)
        
        with open(task_file, "w") as f:
            json.dump({
                "status": "completed", 
                "models": leaderboard
            }, f)
            
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        logger.error(traceback.format_exc())
        with open(task_file, "w") as f:
            json.dump({"status": "failed", "msg": traceback.format_exc(), "models": []}, f)


# ==========================================
# API 路由
# ==========================================
@app.get("/api/gpu/status")
async def get_gpu_status():
    """
    获取 GPU 状态信息。
    """
    gpu_info = get_gpu_info()
    # 即使 gpu_info 包含 error，我们也返回 200 OK 和 JSON
    # 只有在发生 FastAPI 无法处理的严重内部错误时才返回 500
    return {"gpus": gpu_info}

@app.get("/api/datasets/folders")
async def list_dataset_folders():
    """获取 /home/yhz/iot_shujuji 目录下的所有子文件夹列表"""
    if not os.path.exists(DATASET_ROOT_PATH):
        logger.warning(f"数据集根目录不存在: {DATASET_ROOT_PATH}")
        return {"code": 200, "folders": []}
    
    if not os.path.isdir(DATASET_ROOT_PATH):
        logger.error(f"数据集根路径不是一个目录: {DATASET_ROOT_PATH}")
        return {"code": 500, "msg": f"根路径不是一个目录: {DATASET_ROOT_PATH}"}
    
    items = os.listdir(DATASET_ROOT_PATH)
    folders = [item for item in items if os.path.isdir(os.path.join(DATASET_ROOT_PATH, item))]
    
    return {"code": 200, "folders": folders}

@app.get("/api/datasets/files/{folder_name}")
async def list_dataset_files(folder_name: str):
    """获取 /home/yhz/iot_shujuji/{folder_name} 目录下的所有 .csv 和 .tsv 文件列表"""
    folder_path = os.path.join(DATASET_ROOT_PATH, folder_name)
    
    if not os.path.exists(folder_path):
        logger.warning(f"数据集子目录不存在: {folder_path}")
        return {"code": 404, "msg": f"数据集子目录不存在: {folder_name}", "files": []}
    
    if not os.path.isdir(folder_path):
        logger.error(f"路径不是一个目录: {folder_path}")
        return {"code": 500, "msg": f"路径不是一个目录: {folder_path}"}
    
    allowed_extensions = {".csv", ".tsv"}
    all_items = os.listdir(folder_path)
    files = [f for f in all_items if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in allowed_extensions]
    
    return {"code": 200, "files": files}


@app.post("/api/train/start")
async def start_training(
    background_tasks: BackgroundTasks, 
    folder_name: str = Form(...),
    file_names: str = Form(...),
    train_ratio: float = Form(...)
):
    # 1. 构建文件路径列表
    folder_path = os.path.join(DATASET_ROOT_PATH, folder_name)
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"数据集文件夹不存在: {folder_name} (Path: {folder_path})")
    
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail=f"路径不是一个目录: {folder_path}")
    
    # 解析文件名列表
    requested_files = [f.strip() for f in file_names.split(",") if f.strip()]
    
    # 如果文件名列表为空，则表示选择整个文件夹
    if not requested_files:
        allowed_extensions = {".csv", ".tsv"}
        all_items = os.listdir(folder_path)
        requested_files = [f for f in all_items if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in allowed_extensions]
        if not requested_files:
             raise HTTPException(status_code=400, detail=f"文件夹 '{folder_name}' 中没有找到任何 .csv 或 .tsv 文件。")

    file_paths = []
    for fname in requested_files:
        # 验证文件扩展名
        allowed_extensions = {".csv", ".tsv"}
        file_ext = os.path.splitext(fname)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {fname}. 仅支持: {allowed_extensions}")
        
        full_file_path = os.path.join(folder_path, fname)
        if not os.path.exists(full_file_path):
             raise HTTPException(status_code=404, detail=f"指定的文件不存在: {full_file_path}")
        
        file_paths.append(full_file_path)
    
    if not file_paths:
        raise HTTPException(status_code=400, detail="没有有效的文件路径用于训练。")

    task_id = str(uuid.uuid4())
    # <--- 3. 移除全局 bg_tasks 变量声明
    # <--- 4. 使用传入的 background_tasks 参数
    background_tasks.add_task(run_hybrid_pipeline, task_id, file_paths, train_ratio)
    return {"code": 200, "task_id": task_id}



@app.get("/api/train/status/{task_id}")
async def get_train_status(task_id: str):
    task_file = f"./storage/tasks/{task_id}.json"
    if not os.path.exists(task_file):
        return {"code": 404, "data": {"status": "not_found"}}
    
    with open(task_file, "r") as f:
        content = f.read().strip()
        if not content:
            logger.error(f"Task file {task_file} is empty!")
            return {"code": 500, "data": {"status": "error", "msg": "Task file is empty."}}
        
        import io
        try:
            data = json.load(io.StringIO(content))
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse JSON from {task_file}: {je}")
            return {"code": 500, "data": {"status": "error", "msg": f"Failed to parse task result: {str(je)}"}}
    
    return {"code": 200, "data": data}

@app.post("/api/model/deploy")
async def deploy_model(model_path: str = Form(...), model_name: str = Form(...)):
    global DEPLOYED_MODEL
    try:
        if "pycaret" in model_path:
            from pycaret.classification import load_model
            DEPLOYED_MODEL = load_model(model_path.replace('.pkl', ''))
            model_type = 'pycaret'
        else:
            DEPLOYED_MODEL = LudwigModel.load(model_path, backend='local')
            model_type = 'ludwig'
        
        DEPLOYED_MODEL = (DEPLOYED_MODEL, model_type, model_path)
        return {"code": 200, "msg": f"{model_name} 部署成功！"}
    except Exception as e:
        logger.error(f"模型部署失败: {e}")
        return {"code": 500, "msg": f"加载模型失败: {str(e)}"}

@app.post("/api/deploy")
async def deploy_model_from_table(model_name: str = Form(...), model_id: str = Form(...)):
    latest_task_files = glob.glob("./storage/tasks/*.json")
    if not latest_task_files:
         return {"code": 500, "msg": "未找到任何训练任务结果，无法部署。"}
    
    latest_task_file = max(latest_task_files, key=os.path.getctime)
    with open(latest_task_file, "r") as f:
        content = f.read().strip()
        if not content:
            logger.error(f"Latest task file {latest_task_file} is empty!")
            return {"code": 500, "msg": "Latest task file is empty."}
        
        import io
        try:
            task_data = json.load(io.StringIO(content))
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse JSON from {latest_task_file}: {je}")
            return {"code": 500, "msg": f"Failed to parse latest task result: {str(je)}"}
    
    if task_data.get("status") != "completed":
        return {"code": 500, "msg": "最新任务未完成或失败，无法部署。"}
    
    models = task_data.get("models", [])
    model_to_deploy = next((m for m in models if m["model_name"] == model_name), None)
    
    if not model_to_deploy:
        return {"code": 500, "msg": f"在最新任务结果中未找到模型 '{model_name}'。"}
    
    model_path = model_to_deploy["model_path"]
    
    try:
        if "pycaret" in model_path:
            from pycaret.classification import load_model
            loaded_model = load_model(model_path.replace('.pkl', ''))
            model_type = 'pycaret'
        else:
            loaded_model = LudwigModel.load(model_path, backend='local')
            model_type = 'ludwig'
        
        # 更新全局变量
        global DEPLOYED_MODEL
        DEPLOYED_MODEL = (loaded_model, model_type, model_path)
        return {"code": 200, "msg": f"{model_name} 部署成功！"}
    except Exception as e:
        logger.error(f"模型部署失败: {e}")
        return {"code": 500, "msg": f"加载模型失败: {str(e)}"}
@app.post("/api/predict")
def predict(text: str, model_path: str):
    if DEPLOYED_MODEL is None:
        return {"error": "No model deployed. Please deploy a model first."}
    
    loaded_model, model_type, _ = DEPLOYED_MODEL
    
    try:
        if "pycaret" in model_path:
            from pycaret.classification import predict_model
            df_new = pd.DataFrame([{"text": text}])
            prediction_df = predict_model(loaded_model, data=df_new)
            return prediction_df['prediction_label'].iloc[0]
        else:
            df_new = pd.DataFrame([{"text": text}])
            pred, _ = loaded_model.predict(df_new)
            return pred['label_predictions'].iloc[0]
    except Exception as e:
        logger.error(f"预测失败: {e}")
        return {"error": f"Prediction failed: {e}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)