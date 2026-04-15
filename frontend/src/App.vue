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
            :disabled="!selectedFolder || (selectedFiles.length === 0 && !selectAllChecked)"
            style="margin-top: 15px;"
          >
            {{ isTraining ? '模型后台全速训练中 (请勿刷新)...' : '一键并行训练 5 大基线模型' }}
          </el-button>
        </el-card>

        <el-card shadow="never" class="box-card" v-if="leaderboard.length > 0 || isTraining">
          <h3>第三步：模型性能排行榜 (Leaderboard)</h3>
          <el-table :data="leaderboard" v-loading="isTraining" border style="width: 100%">
            <el-table-column type="index" label="排名" width="80" align="center" />
            <el-table-column prop="model_name" label="算法模型名称" />
            <el-table-column prop="f1_score" label="F1 分数 (综合性能)" align="center">
              <template #default="scope">
                <el-tag :type="scope.row.f1_score > 0.8 ? 'success' : 'warning'">{{ scope.row.f1_score }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="status" label="状态" align="center">
               <template #default="scope">
                 <span :style="{ color: scope.row.status === 'Ready' ? 'green' : 'orange' }">
                   {{ scope.row.status || '训练完成' }}
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
          <div v-if="isTraining" style="margin-top: 10px; text-align: center; color: #909399; font-size: 12px;">
            正在实时同步训练进度...
          </div>
        </el-card>
      </el-tab-pane>

      <!-- ================= 模块 2：模型集市与自定义 ================= -->
      <el-tab-pane label="高级模型集市" name="market">
        <el-card shadow="never" class="box-card">
          <h3>付费解锁专区 (企业级模型)</h3>
          <el-row :gutter="20">
            <el-col :span="12" v-for="(item, index) in premiumModels" :key="index">
              <el-card class="premium-card" shadow="hover">
                <h4>       {{ item.name }}</h4>
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
import { ref, onMounted, onBeforeUnmount } from 'vue'
import axios from 'axios' // 确保已安装并引入 axios
import { ElMessage, ElNotification } from 'element-plus'

const activeTab = ref('train')

// 新增：管理文件夹和文件的选择状态
const availableFolders = ref([]) // 存储从后端获取的文件夹列表
const selectedFolder = ref('') // 用户选择的文件夹
const availableFiles = ref([]) // 存储选中文件夹下的文件列表
const selectedFiles = ref([]) // 用户勾选的文件列表
const selectAllChecked = ref(false) // 标记是否全选了当前文件夹
const loadingFiles = ref(false) // 标记是否正在加载文件列表


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
  selectedFiles.value = [] // 清空之前的选择
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
    availableFiles.value = [] // 加载失败也清空列表
  } finally {
    loadingFiles.value = false
  }
}

// 新增：全选当前文件夹下的所有文件
const selectAllFiles = () => {
  selectedFiles.value = [...availableFiles.value] // 创建副本以触发响应式更新
  selectAllChecked.value = true
  console.log('已全选当前文件夹下的所有文件:', selectedFiles.value)
}
let gpuStatusIntervalId = null;

// 修改 fetch 地址，确保与后端启动的 host:port 一致
// 如果后端是 127.0.0.1:8000，这里要么是相对路径（如果前端通过 8000 端口服务），要么是完整 URL
// 假设前后端同源或已配置 CORS，则使用相对路径是正确的
function fetchAndDisplayGpuStatus() {
    // 使用相对路径 /api/gpu/status
    // 如果前端是通过另一个服务器（如 Vite dev server）在不同端口（如 8080）运行，
    // 并且后端在 8000，你需要将地址改为 'http://127.0.0.1:8000/api/gpu/status'
    // 但鉴于你后端允许了 "*" 跨域，相对路径通常就够了。
    fetch('http://127.0.0.1:8000/api/gpu/status')  
        .then(response => {
            // 检查 HTTP 状态码是否为 OK
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json(); // 尝试解析 JSON
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
                    html += `
                    <li>
                        <strong>GPU ${gpu.index} (${gpu.name}):</strong><br>
                        - Utilization: ${gpu.gpu_util_percent}%<br>
                        - Memory Used: ${(gpu.used_memory / (1024**3)).toFixed(2)} GB / ${(gpu.total_memory / (1024**3)).toFixed(2)} GB (${gpu.memory_util_percent}%)<br>
                        ${gpu.temperature !== null ? `- Temperature: ${gpu.temperature}°C<br>` : ''}
                        ${gpu.power_usage_watts !== null ? `- Power: ${gpu.power_usage_watts.toFixed(2)} W<br>` : ''}
                    </li>`;
                });
                html += '</ul>';
                container.innerHTML = html;
            } else {
                container.innerHTML = '<p>No GPU information available.</p>';
            }
        })
        .catch(error => {
            // 这个 catch 现在可以捕获网络错误、非 2xx 状态码错误以及 JSON 解析错误
            console.error('Error fetching GPU status:', error);
            const container = document.getElementById('gpu-info-content');
            container.innerHTML = `<p>Error loading GPU status: ${error.message}</p>`;
        });
}

// 启动轮询
function startGpuStatusPolling() {
    if (gpuStatusIntervalId) {
        clearInterval(gpuStatusIntervalId);
    }
    gpuStatusIntervalId = setInterval(fetchAndDisplayGpuStatus, 2000);
    fetchAndDisplayGpuStatus(); // 立即获取一次
}

// 页面加载完成后启动轮询
document.addEventListener('DOMContentLoaded', function () {
    startGpuStatusPolling();
});
// 修改：触发一键训练
const startTraining = async () => {
  if (!selectedFolder.value) {
    ElMessage.warning('请先选择一个数据集文件夹！')
    return
  }

  // 检查是选择了特定文件还是全选
  if (selectedFiles.value.length === 0 && !selectAllChecked.value) {
     ElMessage.warning('请选择至少一个数据文件，或者点击“全选本文件夹”。')
     return
  }

  isTraining.value = true
  leaderboard.value = [] // 清空旧数据
  
  try {
    const formData = new FormData()
    formData.append('folder_name', selectedFolder.value)
    
    // 发送文件名列表，如果是全选，则发送空字符串
    let fileNamesToSend = selectAllChecked.value ? '' : selectedFiles.value.join(',')
    formData.append('file_names', fileNamesToSend)
    
    formData.append('train_ratio', trainRatio.value)

    // 修改请求地址
    const response = await axios.post('http://localhost:8000/api/train/start', formData)

    if (response.data.code === 200) {
      currentTaskId.value = response.data.task_id
      ElMessage.success('训练任务已提交，后台正在并行计算...')
      
      // 启动轮询
      startPolling()
    } else {
      throw new Error(response.data.msg || '启动失败')
    }
  } catch (error) {
    console.error(error)
    isTraining.value = false
    const errorMsg = error.response?.data?.detail || error.message
    ElMessage.error('训练启动失败：' + JSON.stringify(errorMsg))
  }
}

// 3. 轮询训练状态 (核心修复部分)
const startPolling = () => {
  if (pollingTimer) clearInterval(pollingTimer)
  
  pollingTimer = setInterval(async () => {
    try {
      const res = await axios.get(`http://localhost:8000/api/train/status/${currentTaskId.value}`)
      
      // 使用可选链 (?.) 防止层级不对导致报错
      const responseCode = res.data?.code
      const responseData = res.data?.data || res.data // 兼容两种返回结构
      
      if (responseCode === 200) {
        const status = responseData.status
        const models = responseData.models || []

        // 更新排行榜数据
        if (models.length > 0) {
          // 按 F1 分数排序
          leaderboard.value = models.sort((a, b) => b.f1_score - a.f1_score)
        }

        // 精确判断状态，一旦完成立即停止轮询
        if (status === 'completed' || status === 'failed') {
          isTraining.value = false
          clearInterval(pollingTimer)
          pollingTimer = null
          
          if (status === 'completed') {
            ElNotification({
              title: '训练完成',
              message: '所有基线模型训练完毕，请查看排行榜并部署最佳模型。',
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
      }
    } catch (error) {
      console.error('Polling error:', error)
      // 网络错误时不立即停止，允许重试
    }
  }, 2000) // 每 2 秒轮询一次
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
  loadFolders(); // 调用新函数加载文件夹列表

  premiumModels.value = [
    { name: 'DeepForest Pro', desc: '适用于复杂森林火灾预测的深度集成模型，精度提升 15%。', price: '¥ 2,999/年' },
    { name: 'TimeSeries Transformer', desc: '基于 Transformer 架构的工业时序异常检测引擎。', price: '¥ 4,500/年' },
    { name: 'Edge-Lite V3', desc: '专为低功耗 MCU 优化的量化模型，推理速度 < 10ms。', price: '¥ 1,200/年' },
    { name: 'Multi-Modal Fusion', desc: '融合视觉与传感器数据的综合决策模型。', price: '联系报价' }
  ]
})

// 7. 清理定时器
onBeforeUnmount(() => {
  if (pollingTimer) {
    clearInterval(pollingTimer)
    pollingTimer = null
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