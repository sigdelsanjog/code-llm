┌─────────────────────────────────────────────────────────────────────┐
│ FRONTEND (chat-ui) │
├─────────────────────────────────────────────────────────────────────┤
│ Training Config Page │
│ ├── Config Form (all TrainingConfig params) │
│ ├── Preset Selector (load saved configs) │
│ ├── Save Config Button │
│ └── Start Training Button │
│ │
│ Training Monitor Page │
│ ├── Progress Bar (step X/Y) │
│ ├── Live Metrics (loss, learning rate) │
│ ├── Logs Stream │
│ └── Stop Training Button │
└──────────────────────────┬──────────────────────────────────────────┘
│ REST API + WebSocket/SSE
▼
┌─────────────────────────────────────────────────────────────────────┐
│ BACKEND (FastAPI) │
├─────────────────────────────────────────────────────────────────────┤
│ Endpoints: │
│ ├── POST /api/training/config → Save configuration │
│ ├── GET /api/training/configs → List saved configs │
│ ├── GET /api/training/presets → Get default presets │
│ ├── POST /api/training/start → Start training job │
│ ├── GET /api/training/status/{id} → Get job status │
│ ├── WS /api/training/stream/{id} → Live progress stream │
│ └── POST /api/training/stop/{id} → Stop training │
│ │
│ Services: │
│ ├── TrainingJobService (manages async training) │
│ └── ConfigService (CRUD for configs) │
└──────────────────────────┬──────────────────────────────────────────┘
│ Direct Components
▼
┌─────────────────────────────────────────────────────────────────────┐
│ GPTMED PACKAGE │
│ (TrainingService, DeviceManager, Trainer, etc.) │
└─────────────────────────────────────────────────────────────────────┘
