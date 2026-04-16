<template>
  <div class="ai-platform">
    <h2>IoT 智能体 - AI 模型管理中心</h2>

    <el-tabs v-model="activeTab" type="border-card">

      <!-- ================= 模块 1：训练与部署闭环 ================= -->
      <el-tab-pane label="模型训练与部署" name="train">
        <el-card shadow="never" class="box-card">
          <h3>第一步：选择历史数据集</h3>

          <!-- 第一层：选择数据集文件夹 -->
          <div style="margin-bottom: 15px;">
            <label>选择数据集文件夹:</label>
            <el-select
              v-model="selectedFolder"
              placeholder="请选择一个数据集文件夹"
              style="width: 300px; margin-left: 10px;"
              @change="onFolderChange"
              clearable
            >
              <el-option
                v-for="folder in availableFolders"
                :key="folder"
                :label="folder"
                :value="folder"
              />
            </el-select>
          </div>

          <!-- 第二层：选择文件或全选 -->
          <div v-if="selectedFolder">
            <label>选择数据文件:</label>
            <el-checkbox-group
              v-model="selectedFiles"
              style="display: inline-block; margin-left: 10px;"
              :disabled="loadingFiles"
            >
              <el-checkbox
                v-for="file in availableFiles"
                :key="file"
                :label="file"
                :value="file"
              >
                {{ file }}
              </el-checkbox>
            </el-checkbox-group>

            <!-- 全选按钮 -->
            <el-button
              size="small"
              type="info"
              plain
              @click="selectAllFiles"
              :disabled="loadingFiles"
              style="margin-left: 10px;"
            >
              全选本文件夹
            </el-button>

            <div v-if="selectedFiles.length > 0" style="margin-top: 5px; color: green;">
              已选择 {{ selectedFiles.length }} 个文件。
            </div>
          </div>
        </el-card>

        <el-card shadow="never" class="box-card">
          <h3>第二步：配置模型学习参数</h3>
          <!-- 修改后的模型选择区域 -->
          <div style="margin-bottom: 15px;">
            <label>选择要训练的模型:</label>
            <el-checkbox-group 
              v-model="selectedModelNamesList"
              style="display: inline-block; margin-left: 10px;"
            >
              <el-checkbox 
                v-for="model in availableModels" 
                :key="model.name" 
                :value="model.name"
              >
                {{ model.displayName }}
              </el-checkbox>
            </el-checkbox-group>
            <el-button 
              size="small" 
              type="info" 
              plain 
              @click="selectAllModels"
              style="margin-left: 10px;"
            >
              全选
            </el-button>
            <el-button 
              size="small" 
              type="info" 
              plain 
              @click="clearAllModels"
              style="margin-left: 10px;"
            >
              清空
            </el-button>
          </div>
          <!-- 新增：标签列输入 -->
          <div class="slider-block" style="margin-bottom: 15px;">
            <label style="display: block; margin-bottom: 5px;">请输入标签列名:</label>
            <el-input
              v-model="labelColumn"
              placeholder="例如: label, category, target, outcome"
              style="width: 300px; margin-left: 10px;"
            />
            <p class="sub-text">请指定数据集中代表分类或回归目标的列名。</p>
          </div>

          <div class="slider-block">
            <span class="label">训练集分配比例 (用于学习): {{ (trainRatio * 100).toFixed(0) }}%</span>
            <el-slider v-model="trainRatio" :min="0.5" :max="0.9" :step="0.05" show-stops />
            <p class="sub-text">注：剩余比例将自动平分为验证集与测试集用于考核模型能力。</p>
          </div>

          <div id="gpu-info-container" style="margin-top: 20px; padding: 10px; border: 1px solid #ccc;">
            <h3>GPU Status</h3>
            <div id="gpu-info-content">
                <!-- GPU 信息将在这里动态加载 -->
            </div>
          </div>

          <el-button
            type="success"
            size="large"
            @click="startTraining"
            :loading="isTraining"
            :disabled="!selectedFolder || (selectedFiles.length === 0 && !selectAllChecked) || !labelColumn || selectedModelNamesList.length === 0"
            style="margin-top: 15px;"
          >
            {{ isTraining ? '模型后台全速训练中 (请勿刷新)...' : '一键并行训练 5 大基线模型' }}
          </el-button>
        </el-card>

       <el-card shadow="never" class="box-card" v-if="leaderboard.length > 0 || isTraining">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
    <h3 style="margin: 0;">第三步：模型性能排行榜 (Leaderboard)</h3>
    <!-- 独立的、不遮挡表格的 loading 提示 -->
    <div v-if="isTraining" style="display: flex; align-items: center; color: #909399; font-size: 12px;">
      <el-icon class="is-loading" style="margin-right: 6px;">
        <Loading />
      </el-icon>
      正在实时同步训练进度...
    </div>
  </div>

  <!-- 关键：移除了 v-loading="isTraining" -->
  <el-table :data="leaderboard" border style="width: 100%;">
    <el-table-column type="index" label="排名" width="80" align="center" />
    <el-table-column prop="model_name" label="算法模型名称" />
    <el-table-column prop="f1_score" label="F1 分数 (综合性能)" align="center">
      <template #default="scope">
        <el-tag :type="scope.row.f1_score > 0.8 ? 'success' : 'warning'">
          {{ scope.row.f1_score === null ? '--' : scope.row.f1_score }}
        </el-tag>
      </template>
    </el-table-column>
    <el-table-column prop="status" label="状态" align="center">
       <template #default="scope">
         <span :style="{ color: scope.row.status === 'Ready' ? 'green' : (scope.row.status === 'Failed' ? 'red' : 'orange') }">
           {{ scope.row.status || 'Pending' }}
         </span>
       </template>
    </el-table-column>
    <el-table-column label="操作" align="center" width="200">
      <template #default="scope">
        <el-button
          type="primary"
          size="small"
          @click="deployModel(scope.row)"
          :disabled="scope.row.status !== 'Ready'"
        >
          一键部署至 IoT
        </el-button>
      </template>
    </el-table-column>
  </el-table>
</el-card>
      </el-tab-pane>

      <!-- ================= 模块 2：模型集市与自定义 ================= -->
      <el-tab-pane label="高级模型集市" name="market">
        <el-card shadow="never" class="box-card">
          <h3>付费解锁专区 (企业级模型)</h3>
          <el-row :gutter="20">
            <el-col :span="12" v-for="(item, index) in premiumModels" :key="index">
              <el-card class="premium-card" shadow="hover">
                <h4> {{ item.name }}</h4>
                <p>{{ item.desc }}</p>
                <div class="price-bar">
                  <span class="price">{{ item.price }}</span>
                  <el-button type="warning" plain size="small">联系商务解锁</el-button>
                </div>
              </el-card>
            </el-col>
          </el-row>
        </el-card>

        <el-card shadow="never" class="box-card" style="margin-top: 20px;">
          <h3>托管自研模型</h3>
          <el-form :inline="true">
            <el-form-item label="模型名称">
              <el-input v-model="customModelName" placeholder="例如：自研 XGBoost V2" style="width: 250px;" />
            </el-form-item>
            <el-form-item>
              <el-upload
                action="http://localhost:8000/api/market/custom_upload"
                :data="{ model_name: customModelName }"
                :on-success="handleCustomUploadSuccess"
                :show-file-list="false"
                accept=".pkl, .h5, .pt"
              >
                <el-button type="primary" :disabled="!customModelName">上传模型文件 (.pkl, .h5)</el-button>
              </el-upload>
            </el-form-item>
          </el-form>
          <p v-if="customUploadStatus" class="sub-text" style="margin-top: 10px;">{{ customUploadStatus }}</p>
        </el-card>
      </el-tab-pane>

    </el-tabs>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue' // 移除了 computed
import axios from 'axios'
import { ElMessage, ElNotification } from 'element-plus'
import { Loading } from '@element-plus/icons-vue'
const activeTab = ref('train')

// 新增：模型选择相关变量
// availableModels 仅包含显示信息和值
const availableModels = ref([
  { name: 'TextCNN_Hyperopt_AutoFeat', displayName: 'TextCNN' },
  { name: 'Bi-LSTM_Hyperopt_AutoFeat', displayName: 'Bi-LSTM' },
  { name: 'Bi-GRU_Hyperopt_AutoFeat', displayName: 'Bi-GRU' },
  { name: 'Vanilla-RNN_Hyperopt_AutoFeat', displayName: 'Vanilla-RNN' },
  { name: 'FastText-Embed_Hyperopt_AutoFeat', displayName: 'FastText-Embed' },
  { name: 'RF_AutoFeat', displayName: 'Random Forest' },
  { name: 'NB_AutoFeat', displayName: 'Naive Bayes' }
]);

// 使用一个 ref 来存储选中的模型名称列表
const selectedModelNamesList = ref([]);

// 修改全选方法
const selectAllModels = () => {
  // 将所有模型的 name 添加到 selectedModelNamesList
  selectedModelNamesList.value = availableModels.value.map(m => m.name);
};

// 修改清空方法
const clearAllModels = () => {
  // 清空 selectedModelNamesList
  selectedModelNamesList.value = [];
};

// 新增：管理文件夹和文件的选择状态
const availableFolders = ref([])
const selectedFolder = ref('')
const availableFiles = ref([])
const selectedFiles = ref([])
const selectAllChecked = ref(false)
const loadingFiles = ref(false)

// 新增：存储用户输入的标签列名
const labelColumn = ref('')

// 修改默认值：防止数据量过少导致 F1=0，建议至少 0.7-0.8
const trainRatio = ref(0.8)

const isTraining = ref(false)
const currentTaskId = ref('')
const leaderboard = ref([])
let pollingTimer = null

const premiumModels = ref([])
const customModelName = ref('')
const customUploadStatus = ref('')

// 新增：加载数据集文件夹列表
const loadFolders = async () => {
  try {
    const response = await axios.get('http://localhost:8000/api/datasets/folders')
    if (response.data.code === 200) {
      availableFolders.value = response.data.folders
      console.log('数据集文件夹列表已加载:', response.data.folders)
      if (availableFolders.value.length === 0) {
        ElMessage.info('指定的数据集根目录中暂无子文件夹。')
      }
    } else {
      throw new Error(response.data.msg || '获取数据集文件夹列表失败')
    }
  } catch (error) {
    console.error('加载数据集文件夹列表失败:', error)
    ElMessage.error('加载数据集文件夹列表失败，请检查后端连接。')
  }
}

// 新增：当选择文件夹时，加载其下的文件
const onFolderChange = async (folderName) => {
  if (!folderName) {
    availableFiles.value = []
    selectedFiles.value = []
    selectAllChecked.value = false
    return
  }

  loadingFiles.value = true
  selectedFiles.value = []
  selectAllChecked.value = false

  try {
    const response = await axios.get(`http://localhost:8000/api/datasets/files/${encodeURIComponent(folderName)}`)
    if (response.data.code === 200) {
      availableFiles.value = response.data.files
      console.log(`文件夹 '${folderName}' 下的文件列表已加载:`, response.data.files)
      if (availableFiles.value.length === 0) {
        ElMessage.warning(`文件夹 '${folderName}' 中没有找到 .csv 或 .tsv 文件。`)
      }
    } else {
       // 特殊处理 404
       if (response.data.code === 404) {
         ElMessage.error(response.data.msg)
       } else {
         throw new Error(response.data.msg || '获取文件列表失败')
       }
    }
  } catch (error) {
    console.error('加载数据集文件列表失败:', error)
    ElMessage.error('加载数据集文件列表失败，请检查后端连接。')
    availableFiles.value = []
  } finally {
    loadingFiles.value = false
  }
}

// 新增：全选当前文件夹下的所有文件
const selectAllFiles = () => {
  selectedFiles.value = [...availableFiles.value]
  selectAllChecked.value = true
  console.log('已全选当前文件夹下的所有文件:', selectedFiles.value)
}

let gpuStatusIntervalId = null;

// 修正 GPU 状态获取函数，避免嵌套模板字符串问题
function fetchAndDisplayGpuStatus() {
    fetch('http://127.0.0.1:8000/api/gpu/status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const container = document.getElementById('gpu-info-content');
            if (data.gpus && Array.isArray(data.gpus)) {
                let html = '<ul>';
                data.gpus.forEach(gpu => {
                    if (gpu.error) {
                        html += `<li>Error: ${gpu.error}</li>`;
                        return;
                    }

                    // 提取条件部分，避免嵌套模板字符串
                    let tempStr = '';
                    let powerStr = '';
                    if (gpu.temperature !== null) {
                        tempStr = `- Temperature: ${gpu.temperature}°C<br>`;
                    }
                    if (gpu.power_usage_watts !== null) {
                        powerStr = `- Power: ${gpu.power_usage_watts.toFixed(2)} W<br>`;
                    }

                    html += `
                    <li>
                        <strong>GPU ${gpu.index} (${gpu.name}):</strong><br>
                        - Utilization: ${gpu.gpu_util_percent}%<br>
                        - Memory Used: ${(gpu.used_memory / (1024**3)).toFixed(2)} GB / ${(gpu.total_memory / (1024**3)).toFixed(2)} GB (${gpu.memory_util_percent}%)<br>
                        ${tempStr}
                        ${powerStr}
                    </li>`;
                });
                html += '</ul>';
                container.innerHTML = html;
            } else {
                container.innerHTML = '<p>No GPU information available.</p>';
            }
        })
        .catch(error => {
            console.error('Error fetching GPU status:', error);
            const container = document.getElementById('gpu-info-content');
            container.innerHTML = `<p>Error loading GPU status: ${error.message}</p>`;
        });
}

function startGpuStatusPolling() {
    if (gpuStatusIntervalId) {
        clearInterval(gpuStatusIntervalId);
    }
    gpuStatusIntervalId = setInterval(fetchAndDisplayGpuStatus, 2000);
    fetchAndDisplayGpuStatus();
}

// 修改：触发一键训练
// 修改：触发一键训练
const startTraining = async () => {
  if (!selectedFolder.value) {
    ElMessage.warning('请先选择一个数据集文件夹！')
    return
  }

  if (selectedFiles.value.length === 0 && !selectAllChecked.value) {
     ElMessage.warning('请选择至少一个数据文件，或者点击“全选本文件夹”。')
     return
  }

  if (selectedModelNamesList.value.length === 0) {
    ElMessage.warning('请至少选择一个模型进行训练！')
    return
  }

  if (!labelColumn.value.trim()) {
    ElMessage.warning('请输入数据集的标签列名！')
    return
  }

  isTraining.value = true
  // --- 修改 1: 初始化 leaderboard 为包含所有选中模型的占位符 ---
  // 这样前端就有了所有模型的初始状态
  leaderboard.value = selectedModelNamesList.value.map(name => ({
    model_name: name,
    f1_score: null, // 初始分数为 null
    accuracy: null, // 初始精度为 null
    status: 'Pending', // 初始状态为 Pending (注意大小写需与后端一致，这里是 'Pending'，后端是 'pending')
    model_path: null
  }))
  // ----------------------------

  try {
    const formData = new FormData()
    formData.append('folder_name', selectedFolder.value)
    let fileNamesToSend = selectAllChecked.value ? '' : selectedFiles.value.join(',')
    formData.append('file_names', fileNamesToSend)
    formData.append('train_ratio', trainRatio.value)
    formData.append('label_column', labelColumn.value.trim())
    formData.append('selected_models', JSON.stringify(selectedModelNamesList.value))

    const response = await axios.post('http://localhost:8000/api/train/start', formData)

    if (response.data.code === 200) {
      currentTaskId.value = response.data.task_id
      ElMessage.success('训练任务已提交，后台正在并行计算...')
      startPolling()
    } else {
      throw new Error(response.data.msg || '启动失败')
    }
  } catch (error) {
    console.error(error)
    isTraining.value = false
    // 清空排行榜
    leaderboard.value = []
    const errorMsg = error.response?.data?.detail || error.message
    ElMessage.error('训练启动失败：' + JSON.stringify(errorMsg))
  }
}

// 3. 轮询训练状态 (核心修复部分 - 增量显示)
const startPolling = () => {
  if (pollingTimer) clearInterval(pollingTimer)

  pollingTimer = setInterval(async () => {
    try {
      const res = await axios.get(`http://localhost:8000/api/train/status/${currentTaskId.value}`)

      const responseCode = res.data?.code
      const responseData = res.data?.data || res.data

      if (responseCode === 200) {
        const status = responseData.status
        const models = responseData.models || []

        if (models.length > 0) {
          // --- 修改 2: 遍历后端返回的 models，更新前端 leaderboard ---
          // 这次我们更细致地处理
          models.forEach(returnedModel => {
             const existingModelIndex = leaderboard.value.findIndex(item => item.model_name === returnedModel.model_name);
             if (existingModelIndex > -1) {
               // 找到前端 leaderboard 中对应的模型，直接更新其属性
               // 注意：这里直接赋值，Vue 2.x 需要用 $set 或者直接替换整个对象才能触发响应式更新
               // Vue 3.x 对象属性的更改是响应式的，所以直接赋值即可
               leaderboard.value[existingModelIndex].f1_score = returnedModel.f1_score;
               leaderboard.value[existingModelIndex].accuracy = returnedModel.accuracy;
               leaderboard.value[existingModelIndex].model_path = returnedModel.model_path;
               // 注意：后端状态是 'pending', 'completed', 'failed'，前端是 'Pending', 'Completed', 'Failed'
               // 需要映射一下，或者统一约定
               if (returnedModel.status === 'completed') {
                 leaderboard.value[existingModelIndex].status = 'Ready'; // 或者 'Completed'
               } else if (returnedModel.status === 'failed') {
                 leaderboard.value[existingModelIndex].status = 'Failed';
               } else if (returnedModel.status === 'pending') {
                 leaderboard.value[existingModelIndex].status = 'Pending';
               }
               // 如果后端还有 'running' 状态，也可以加上
               // else if (returnedModel.status === 'running') {
               //   leaderboard.value[existingModelIndex].status = 'Running';
               // }
               // 如果后端没有返回 status，可以根据 f1_score 是否有值来判断
               else if (returnedModel.f1_score !== null && returnedModel.f1_score !== undefined) {
                 leaderboard.value[existingModelIndex].status = 'Ready';
               }
             } else {
               // 理论上不应该发生，因为前端的 leaderboard 是根据 selectedModelNamesList 初始化的，
               // 而 selectedModelNamesList 也是发给后端的。但如果后端返回了别的模型，可以考虑添加。
               // 一般情况下可以忽略。
             }
          });

          // --- 修改 3: 检查是否所有模型都已完成，如果有新完成的，则排序 ---
          const completedModels = leaderboard.value.filter(m => m.status === 'Ready'); // 或 'Completed'
          // 如果有模型完成了，并且之前没有完成过，可以触发排序
          // 更简单的做法是，只要有模型的 f1_score 不为 null，就排序
          const hasAnyScore = leaderboard.value.some(m => m.f1_score !== null);

          if (hasAnyScore) {
              // 对整个 leaderboard 进行排序，将有分数的排前面，空的排后面
              // 确保 null 值被视为最小值
              leaderboard.value.sort((a, b) => {
                  if (a.f1_score === null && b.f1_score === null) return 0;
                  if (a.f1_score === null) return 1;
                  if (b.f1_score === null) return -1;
                  return b.f1_score - a.f1_score; // 降序排列
              });
          }
        }

        // --- 修改 4: 只有当后端任务状态变为 completed 或 failed 时才停止轮询 ---
        if (status === 'completed' || status === 'failed') {
          // 最终排序，确保最终顺序是正确的
          leaderboard.value.sort((a, b) => (b.f1_score || 0) - (a.f1_score || 0)); // (b.f1_score || 0) 确保 null 被视为 0

          isTraining.value = false
          clearInterval(pollingTimer)
          pollingTimer = null

          if (status === 'completed') {
            ElNotification({
              title: '训练完成',
              message: '所有选中模型训练完毕，请查看排行榜。',
              type: 'success',
              duration: 5000
            })
          } else {
            ElNotification({
              title: '训练异常',
              message: '训练过程中发生错误，请查看日志。',
              type: 'error',
            })
          }
        }
      } else {
          // 如果 API 返回 code 不是 200，可以考虑停止轮询或处理错误
          console.error('Polling received non-200 response:', res.data);
          // 可选：停止轮询
          // isTraining.value = false;
          // clearInterval(pollingTimer);
          // pollingTimer = null;
      }
    } catch (error) {
      console.error('Polling error:', error)
      // 可选：对轮询错误进行处理，例如记录或通知
    }
  }, 2000) // 轮询间隔 2 秒
}
// 4. 部署模型
const deployModel = async (row) => {
  const loadingMsg = ElMessage.loading({
    message: `正在将 ${row.model_name} 部署至边缘网关...`,
    duration: 0
  })

  try {
    const res = await axios.post('http://localhost:8000/api/deploy', {
      model_name: row.model_name,
      model_id: row.id
    })

    if (res.data.code === 200) {
      loadingMsg.close()
      ElNotification({
        title: '部署成功',
        message: `${row.model_name} 已成功下发至 IoT 设备，版本号为 v${new Date().getTime()}`,
        type: 'success',
        position: 'bottom-right'
      })

      row.status = 'Deployed'
    } else {
      throw new Error(res.data.msg)
    }
  } catch (error) {
    loadingMsg.close()
    ElMessage.error('部署失败：' + error.message)
  }
}

// 5. 自定义模型上传成功
const handleCustomUploadSuccess = (res) => {
  if (res.code === 200) {
    customUploadStatus.value = `模型 "${customModelName.value}" 上传成功！已存入私有仓库。`
    ElMessage.success('自研模型托管成功')
    customModelName.value = ''
  } else {
    customUploadStatus.value = '上传失败：' + (res.msg || '未知错误')
    ElMessage.error('上传失败')
  }
}

// 6. 生命周期：初始化时加载数据集文件夹列表
onMounted(() => {
  loadFolders();

  premiumModels.value = [
    { name: 'DeepForest Pro', desc: '适用于复杂森林火灾预测的深度集成模型，精度提升 15%。', price: '¥ 2,999/年' },
    { name: 'TimeSeries Transformer', desc: '基于 Transformer 架构的工业时序异常检测引擎。', price: '¥ 4,500/年' },
    { name: 'Edge-Lite V3', desc: '专为低功耗 MCU 优化的量化模型，推理速度 < 10ms。', price: '¥ 1,200/年' },
    { name: 'Multi-Modal Fusion', desc: '融合视觉与传感器数据的综合决策模型。', price: '联系报价' }
  ]

  // 启动 GPU 状态轮询
  startGpuStatusPolling();
})

// 7. 清理定时器
onBeforeUnmount(() => {
  if (pollingTimer) {
    clearInterval(pollingTimer)
    pollingTimer = null
  }
  // 清理 GPU 状态轮询
  if (gpuStatusIntervalId) {
    clearInterval(gpuStatusIntervalId);
    gpuStatusIntervalId = null;
  }
})
</script>
<!-- (保持原有的样式部分不变) -->
<style scoped>
.ai-platform {
  padding: 20px;
  background-color: #f5f7fa;
  min-height: 100vh;
}

.box-card {
  margin-bottom: 20px;
}

.box-card h3 {
  margin-top: 0;
  color: #303133;
  border-bottom: 1px solid #EBEEF5;
  padding-bottom: 10px;
}

.slider-block {
  margin-bottom: 20px;
}

.label {
  display: block;
  margin-bottom: 10px;
  font-weight: bold;
  color: #606266;
}

.sub-text {
  color: #909399;
  font-size: 13px;
  margin-top: 5px;
}

.success-text {
  color: #67C23A;
  margin-left: 10px;
}

.el-table {
  margin-top: 10px;
}

.premium-card {
  height: 150px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.premium-card h4 {
  margin: 0;
  color: #E6A23C;
}

.premium-card p {
  font-size: 13px;
  color: #909399;
  line-height: 1.5;
  margin: 5px 0;
}

.price-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 10px;
}

.price {
  font-weight: bold;
  color: #E74C3C;
}
</style>