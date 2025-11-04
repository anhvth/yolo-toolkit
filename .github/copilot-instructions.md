# Copilot Instructions: YOLO + Label Studio Integration Kit

## Project Architecture

This is a **human-in-the-loop ML pipeline** for iterative YOLO model training with Label Studio annotation platform. The workflow is:
1. Upload images → 2. Manual labeling → 3. Export YOLO format → 4. Train model → 5. Auto-predict → 6. Review/correct → repeat

**Key Components:**
- `scripts/1-6_*.py`: Sequential automation scripts (numbered workflow)
- `cli.py`: Advanced batch prediction tool with async support
- `docker-compose.yml`: Label Studio (8080) + ML backend (9090) services
- `data/`: Images (input), exports (YOLO format), predictions (model output)
- `models/`: YOLO .pt files (base pretrained, updated fine-tuned)

## Critical SDK Version Constraints

**MUST use Label Studio SDK v2.0.4** (not >=2.0.0) - pinned for compatibility with label-studio 1.21.0.

### Authentication Pattern (SDK v2.0+)
```python
from label_studio_sdk import LabelStudio  # NOT Client
client = LabelStudio(base_url=url, api_key=key)  # NOT Client(url=..., api_key=...)
```

### API Method Changes
- `client.projects.create()` not `client.start_project()`
- `client.tasks.create_many(project=id, request=tasks)` not `client.import_tasks()`
- `client.projects.exports.create(id=project_id, export_type="YOLO")` not `client.export_tasks()`
- `client.tasks.get(id=tid)` not `client.get_task(tid)`

See `test_api.py` for canonical authentication example.

## Apple Silicon (ARM64) Requirements

Docker images **must** include `platform: linux/amd64` for M1/M2 Macs (both label-studio and yolo-ml-backend services). Images lack native ARM64 builds.

## Environment Configuration

All scripts use `.env` with `python-dotenv`. Critical variables:
- `LABEL_STUDIO_API_KEY`: Personal Access Token from UI → Account & Settings → Access Token
- `PROJECT_ID`: Set after running script 2 (returned in output)
- Paths: `IMAGE_DIR`, `EXPORT_DIR`, `PREDICTIONS_DIR`, model paths

**Important:** API tokens are JWT format (not legacy tokens) - 401 errors mean token type mismatch or expiry.

## Script Execution Order

Scripts are **numbered 1-6 for sequential execution**:
1. Start server (Docker or local)
2. Create project → **save PROJECT_ID to .env** (checks for duplicates by default)
3. Upload images from `data/images/`
4. Export annotations (produces zip)
5. Train YOLO (requires extracted export, reads `data.yaml`)
6. Predict unlabeled images

**Project Management:**
- Script 2 prevents duplicate project names by default (use `--allow-duplicate` to override)
- Use `scripts/delete_project.py --id <id>` or `--title <name>` to delete projects
- `client.projects.list(title="name")` to search by name
- `client.projects.delete(id=id)` to delete via API

**All scripts** support `--help` and environment variable overrides via CLI args.

## Docker Workflow vs Local

**Docker (recommended):** `docker compose up -d` runs both services. Images served via `/data/images/` mount (absolute paths in tasks).

**Local:** Use `scripts/1_start_labelstudio.py` or `reset_label_studio_local.sh` (creates admin@example.com/admin user). Data in `~/.local/share/label-studio/`.

**ML Backend connection:** In project settings → Model → Connect to `http://localhost:9090` for auto-predictions in UI.

## Label Configuration Conventions

XML configs in `scripts/2_create_project.py`:
- `model_score_threshold="0.25"`: Confidence filter for ML predictions
- `toName="image"` links RectangleLabels to Image component
- Background colors: red/blue/green/yellow for visual distinction
- Image paths: `/data/images/{filename}` (Docker absolute path, not relative)

Modify `LABEL_CONFIG` variable to change classes before project creation.

## YOLO Integration Patterns

**Training:**
- Base models auto-download via ultralytics (e.g., `yolo11n.pt`)
- Export produces `data.yaml` with class mappings + train/val splits
- Results saved to `runs/train/yolo_labelstudio/` (ultralytics convention)
- Best weights copied to `models/updated_model.pt`

**Prediction:**
- CLI supports batch processing to avoid UI timeouts
- Predictions converted to Label Studio JSON format (percentage coordinates)
- `_convert_to_ls_format()` in `cli.py` handles YOLO → LS coordinate transform

## Common Debugging Steps

1. **401 Unauthorized**: Token expired or wrong type → regenerate in UI, update `.env`
2. **Export empty**: No completed annotations → check task list has labeled items
3. **Model not found**: Download pretrained: `python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"`
4. **Docker network issues**: ML backend can't reach label-studio → check `depends_on` and service names
5. **Platform errors**: Missing `platform: linux/amd64` on ARM64 Macs

## File Naming and Structure Conventions

- Scripts: `{number}_{action}.py` (self-documenting sequential workflow)
- Models: `{base|updated}_model.pt` (clear lifecycle stages)
- Exports: Always to `data/exports/export.zip` → extract before training
- All scripts executable (`#!/usr/bin/env python3`, `chmod +x`)

## Error Handling Pattern

All scripts follow:
```python
if not API_KEY:
    print("❌ Error: ...")
    print("\n📋 Steps to fix:")  # Actionable guidance
    sys.exit(1)
    
try:
    # operation
except Exception as e:
    print(f"❌ Error: {e}")
    if "401" in str(e):
        # Context-specific help
    sys.exit(1)
```

Use emoji indicators (🔗, 📝, ✅, ❌, 💡) for console output clarity.

## Testing and Validation

- `test_api.py`: Minimal auth verification (run before script 2)
- Check Label Studio version compatibility: API errors often indicate SDK/server version mismatch
- Verify ML backend health: `curl http://localhost:9090/health` → `{"status":"UP"}`

## When Making Changes

- **Scripts**: Preserve argument parsing pattern, environment variable fallbacks, help text
- **Dependencies**: Pin `label-studio-sdk==2.0.4` exactly (conflicts with >=2.0.5)
- **Docker**: Always test on ARM64 if adding services (platform constraint)
- **Documentation**: Update both README.md workflow section and script docstrings
