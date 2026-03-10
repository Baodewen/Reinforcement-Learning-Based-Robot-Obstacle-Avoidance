const state = {
  size: 32,
  tool: 'wall',
  grid: [],
  start: [1, 1],
  goal: [30, 30],
};

const elements = {
  cudaStatus: document.getElementById('cudaStatus'),
  modelDir: document.getElementById('modelDir'),
  mapDir: document.getElementById('mapDir'),
  startTrainBtn: document.getElementById('startTrainBtn'),
  trainStatus: document.getElementById('trainStatus'),
  deviceResolved: document.getElementById('deviceResolved'),
  totalTimesteps: document.getElementById('totalTimesteps'),
  startedAt: document.getElementById('startedAt'),
  finishedAt: document.getElementById('finishedAt'),
  latestModelPath: document.getElementById('latestModelPath'),
  bestModelPath: document.getElementById('bestModelPath'),
  trainLogs: document.getElementById('trainLogs'),
  modelList: document.getElementById('modelList'),
  launchDemoBtn: document.getElementById('launchDemoBtn'),
  demoCommand: document.getElementById('demoCommand'),
  demoMapMode: document.getElementById('demoMapMode'),
  demoMapPath: document.getElementById('demoMapPath'),
  savedMaps: document.getElementById('savedMaps'),
  mapCanvas: document.getElementById('mapCanvas'),
  mapStatus: document.getElementById('mapStatus'),
  mapName: document.getElementById('mapName'),
};

function createGrid(size) {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => 0));
}

function resetEditor(size = state.size) {
  state.size = size;
  state.grid = createGrid(size);
  state.start = [1, 1];
  state.goal = [size - 2, size - 2];
  drawCanvas();
}

function drawCanvas() {
  const canvas = elements.mapCanvas;
  const ctx = canvas.getContext('2d');
  const cell = canvas.width / state.size;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#f8f2eb';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < state.size; y += 1) {
    for (let x = 0; x < state.size; x += 1) {
      if (state.grid[y][x] === 1) {
        ctx.fillStyle = '#2e2723';
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }
    }
  }

  ctx.fillStyle = '#88a17f';
  ctx.fillRect(state.start[0] * cell, state.start[1] * cell, cell, cell);
  ctx.fillStyle = '#c97945';
  ctx.fillRect(state.goal[0] * cell, state.goal[1] * cell, cell, cell);

  ctx.strokeStyle = 'rgba(20, 17, 15, 0.12)';
  for (let i = 0; i <= state.size; i += 1) {
    ctx.beginPath();
    ctx.moveTo(i * cell, 0);
    ctx.lineTo(i * cell, canvas.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * cell);
    ctx.lineTo(canvas.width, i * cell);
    ctx.stroke();
  }
}

function applyTool(x, y) {
  if (x < 0 || y < 0 || x >= state.size || y >= state.size) return;
  if (state.tool === 'wall') {
    if ((x === state.start[0] && y === state.start[1]) || (x === state.goal[0] && y === state.goal[1])) return;
    state.grid[y][x] = 1;
  }
  if (state.tool === 'erase') {
    state.grid[y][x] = 0;
  }
  if (state.tool === 'start') {
    state.grid[y][x] = 0;
    state.start = [x, y];
  }
  if (state.tool === 'goal') {
    state.grid[y][x] = 0;
    state.goal = [x, y];
  }
  drawCanvas();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || 'Request failed');
  }
  return data;
}

async function refreshSystem() {
  const data = await fetchJson('/api/system');
  elements.cudaStatus.textContent = data.cuda_available ? '可用' : '不可用';
  elements.modelDir.textContent = data.model_dir;
  elements.mapDir.textContent = data.map_dir;
}

async function refreshTrainStatus() {
  const data = await fetchJson('/api/train/status');
  elements.trainStatus.textContent = data.status;
  elements.deviceResolved.textContent = data.device_resolved || '-';
  elements.totalTimesteps.textContent = data.total_timesteps || 0;
  elements.startedAt.textContent = data.started_at || '-';
  elements.finishedAt.textContent = data.finished_at || '-';
  elements.latestModelPath.textContent = data.latest_model_path || '-';
  elements.bestModelPath.textContent = data.best_model_path || '-';
  elements.trainLogs.textContent = data.logs && data.logs.length ? data.logs.join('\n') : '等待训练日志...';
  if (data.error) {
    elements.trainLogs.textContent += `\n[error] ${data.error}`;
  }
}

async function refreshModels() {
  const data = await fetchJson('/api/models');
  elements.modelList.innerHTML = data.models.map((model) => `
    <article class="model-item">
      <strong>${model.name}</strong>
      <div>状态：${model.exists ? '已生成' : '未生成'}</div>
      <div>路径：${model.path}</div>
      <div>更新时间：${model.updated_at || '-'}</div>
    </article>
  `).join('');
}

async function refreshMaps() {
  const data = await fetchJson('/api/maps');
  const options = data.maps.map((item) => `<option value="${item.path}">${item.name}</option>`).join('');
  elements.savedMaps.innerHTML = options;
  elements.demoMapPath.innerHTML = `<option value="">随机地图</option>${options}`;
}

async function startTraining() {
  const payload = {
    timesteps: Number(document.getElementById('timesteps').value),
    seed: Number(document.getElementById('seed').value),
    map_size: Number(document.getElementById('mapSize').value),
    obstacle_density: Number(document.getElementById('obstacleDensity').value),
    lidar_rays: Number(document.getElementById('lidarRays').value),
    max_steps: Number(document.getElementById('maxSteps').value),
    device: document.getElementById('device').value,
  };
  try {
    await fetchJson('/api/train/start', { method: 'POST', body: JSON.stringify(payload) });
    elements.trainLogs.textContent = '训练任务已提交，等待日志...';
    await refreshTrainStatus();
  } catch (error) {
    elements.trainLogs.textContent = `[error] ${error.message}`;
  }
}

async function saveMap() {
  const payload = {
    name: elements.mapName.value.trim(),
    grid: state.grid,
    start: state.start,
    goal: state.goal,
  };
  try {
    const data = await fetchJson('/api/maps', { method: 'POST', body: JSON.stringify(payload) });
    elements.mapStatus.textContent = `地图已保存：${data.path}`;
    await refreshMaps();
  } catch (error) {
    elements.mapStatus.textContent = `[error] ${error.message}`;
  }
}

async function loadMapIntoEditor() {
  const path = elements.savedMaps.value;
  if (!path) return;
  const name = path.split(/[/\\]/).pop().replace('.json', '');
  try {
    const data = await fetchJson(`/api/maps/${name}`);
    state.size = data.grid.length;
    state.grid = data.grid;
    state.start = data.start;
    state.goal = data.goal;
    elements.mapName.value = data.name;
    drawCanvas();
    elements.mapStatus.textContent = `地图已读取：${data.name}`;
  } catch (error) {
    elements.mapStatus.textContent = `[error] ${error.message}`;
  }
}

async function launchDemo() {
  const payload = {
    model_path: document.getElementById('demoModelPath').value,
    map_path: elements.demoMapMode.value === 'saved' ? elements.demoMapPath.value : '',
    seed: Number(document.getElementById('demoSeed').value),
    fps: Number(document.getElementById('demoFps').value),
    max_episodes: Number(document.getElementById('demoEpisodes').value),
  };
  try {
    const data = await fetchJson('/api/demo/launch', { method: 'POST', body: JSON.stringify(payload) });
    elements.demoCommand.textContent = data.command.join(' ');
  } catch (error) {
    elements.demoCommand.textContent = `[error] ${error.message}`;
  }
}

function setupEditor() {
  resetEditor();
  drawCanvas();
  let drawing = false;
  elements.mapCanvas.addEventListener('mousedown', (event) => {
    drawing = true;
    handleCanvasEvent(event);
  });
  elements.mapCanvas.addEventListener('mousemove', (event) => {
    if (drawing && (state.tool === 'wall' || state.tool === 'erase')) {
      handleCanvasEvent(event);
    }
  });
  window.addEventListener('mouseup', () => {
    drawing = false;
  });

  document.querySelectorAll('.tool').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.tool').forEach((item) => item.classList.remove('active'));
      button.classList.add('active');
      state.tool = button.dataset.tool;
    });
  });
}

function handleCanvasEvent(event) {
  const rect = elements.mapCanvas.getBoundingClientRect();
  const scaleX = elements.mapCanvas.width / rect.width;
  const scaleY = elements.mapCanvas.height / rect.height;
  const x = Math.floor(((event.clientX - rect.left) * scaleX) / (elements.mapCanvas.width / state.size));
  const y = Math.floor(((event.clientY - rect.top) * scaleY) / (elements.mapCanvas.height / state.size));
  applyTool(x, y);
}

function bindEvents() {
  elements.startTrainBtn.addEventListener('click', startTraining);
  document.getElementById('saveMapBtn').addEventListener('click', saveMap);
  document.getElementById('loadMapBtn').addEventListener('click', loadMapIntoEditor);
  document.getElementById('clearMapBtn').addEventListener('click', () => resetEditor());
  elements.launchDemoBtn.addEventListener('click', launchDemo);
  elements.demoMapMode.addEventListener('change', () => {
    elements.demoMapPath.disabled = elements.demoMapMode.value !== 'saved';
  });
  document.getElementById('mapSize').addEventListener('change', (event) => {
    const size = Number(event.target.value);
    if (size >= 16 && size <= 128) {
      resetEditor(size);
    }
  });
}

async function boot() {
  setupEditor();
  bindEvents();
  await Promise.all([refreshSystem(), refreshTrainStatus(), refreshModels(), refreshMaps()]);
  setInterval(refreshTrainStatus, 2500);
  setInterval(refreshModels, 5000);
  setInterval(refreshMaps, 7000);
}

boot().catch((error) => {
  elements.trainLogs.textContent = `[error] ${error.message}`;
});

