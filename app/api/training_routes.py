"""
Training API routes - endpoints cho web UI.
"""
import json
import logging
import threading
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["training"])

# Training state file
RESULTS_DIR = Path("outputs/training_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Global training thread reference
_training_thread = None


class TrainRequest(BaseModel):
    chexpert_dir: str = ""
    nih_dir: str = ""
    use_nih: bool = True
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    image_size: int = 224
    num_workers: int = 4
    patience: int = 7
    policy: str = "ones"
    resume: Optional[str] = None


def _run_training(config: dict):
    """Run training in background thread."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from app.training.pipeline import TrainingPipeline
        pipeline = TrainingPipeline(config)
        pipeline.run()
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        state_file = RESULTS_DIR / 'training_state.json'
        state = {'status': 'error', 'error': str(e)}
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)


@router.post("/start")
def start_training(req: TrainRequest):
    """Bat dau training (chay background)."""
    global _training_thread
    
    # Check if already training
    state_file = RESULTS_DIR / 'training_state.json'
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        if state.get('status') == 'training':
            return {"error": "Training is already running", "status": "training"}
    
    config = {
        'chexpert_dir': req.chexpert_dir,
        'nih_dir': req.nih_dir if req.use_nih else None,
        'no_nih': not req.use_nih,
        'epochs': req.epochs,
        'batch_size': req.batch_size,
        'lr': req.lr,
        'image_size': req.image_size,
        'num_workers': req.num_workers,
        'patience': req.patience,
        'policy': req.policy,
        'resume': req.resume,
        'output_dir': str(RESULTS_DIR),
        'weights_dir': 'weights',
    }
    
    _training_thread = threading.Thread(target=_run_training, args=(config,), daemon=True)
    _training_thread.start()
    
    return {"status": "started", "message": "Training started in background"}


@router.get("/status")
def get_training_status():
    """Lay trang thai training hien tai."""
    state_file = RESULTS_DIR / 'training_state.json'
    if not state_file.exists():
        return {"status": "idle", "message": "No training has been run yet"}
    
    with open(state_file) as f:
        return json.load(f)


@router.get("/history")
def get_training_history():
    """Lay lich su training (loss, auc qua cac epochs)."""
    history_file = RESULTS_DIR / 'training_history.json'
    if not history_file.exists():
        return {"history": [], "message": "No training history"}
    
    with open(history_file) as f:
        return {"history": json.load(f)}


@router.get("/evaluation")
def get_evaluation_results():
    """Lay ket qua evaluation."""
    eval_file = RESULTS_DIR / 'evaluation_results.json'
    if not eval_file.exists():
        return {"results": None, "message": "No evaluation results"}
    
    with open(eval_file) as f:
        return json.load(f)


@router.get("/hyperparameters")
def get_hyperparameters():
    """Lay hyperparameters cua lan train gan nhat."""
    hp_file = RESULTS_DIR / 'hyperparameters.json'
    if not hp_file.exists():
        return {"hyperparameters": None}
    
    with open(hp_file) as f:
        return json.load(f)


@router.post("/stop")
def stop_training():
    """Dung training (save current state)."""
    state_file = RESULTS_DIR / 'training_state.json'
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        state['status'] = 'stopped'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    return {"status": "stopped"}