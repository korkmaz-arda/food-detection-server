import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / "registry.json"
        self._ensure_structure()
        
    def _ensure_structure(self):
        (self.models_dir / "tray").mkdir(parents=True, exist_ok=True)
        (self.models_dir / "food").mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            self._save_registry({
                "active": {
                    "tray": "tray_detector.pt",
                    "food": "yolo11n.pt"
                }
            })

    def _load_registry(self) -> Dict:
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"active": {"tray": None, "food": None}}
    
    def _save_registry(self, registry: Dict):
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def get_active_models(self) -> Dict[str, Optional[str]]:
        registry = self._load_registry()
        active = registry.get("active", {})
        result = {}

        if active.get("tray"):
            tray_path = self.models_dir / "tray" / active["tray"]
            result["tray"] = str(tray_path) if tray_path.exists() else None
        else:
            result["tray"] = None

        if active.get("food"):
            food_path = self.models_dir / "food" / active["food"]
            result["food"] = str(food_path) if food_path.exists() else None
        else:
            result["food"] = None

        return result
    
    def get_model_config(self, model_type: str, model_name: str) -> Dict:
        config_path = self.models_dir / model_type / f"{Path(model_name).stem}_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)

        if model_type == "food":
            return {
                "food_classes": [
                    "banana", "apple", "sandwich", "orange", "broccoli",
                    "carrot", "hot dog", "pizza", "donut", "cake"
                ],
                "food_prices": {
                    "banana": 1.99, "apple": 1.49, "sandwich": 7.99,
                    "orange": 1.79, "broccoli": 2.99, "carrot": 1.29,
                    "hot dog": 4.99, "pizza": 3.99, "donut": 2.49, "cake": 5.99
                },
                "confidence": 0.2
            }
        else:
            return {"confidence": 0.65}
    
    def list_models(self, model_type: str) -> List[Dict]:
        models = []
        model_dir = self.models_dir / model_type
        registry = self._load_registry()
        active_model = registry.get("active", {}).get(model_type)

        for pt_file in model_dir.glob("*.pt"):
            config = self.get_model_config(model_type, pt_file.name)
            models.append({
                "name": pt_file.name,
                "type": model_type,
                "is_active": pt_file.name == active_model,
                "config": config,
                "uploaded_at": datetime.fromtimestamp(pt_file.stat().st_mtime).isoformat()
            })

        return sorted(models, key=lambda x: x["uploaded_at"], reverse=True)
    
    def upload_model(self, model_type: str, file_path: str, name: str = None, config: Dict = None) -> Dict:
        if name:
            target_name = name if name.endswith('.pt') else f"{name}.pt"
        else:
            target_name = Path(file_path).name

        target_path = self.models_dir / model_type / target_name
        import shutil
        shutil.copy(file_path, target_path)

        if config:
            config_path = self.models_dir / model_type / f"{Path(target_name).stem}_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

        return {
            "status": "success",
            "model": target_name,
            "type": model_type
        }
    
    def activate_model(self, model_type: str, model_name: str) -> bool:
        model_path = self.models_dir / model_type / model_name
        if not model_path.exists():
            return False

        registry = self._load_registry()
        if "active" not in registry:
            registry["active"] = {}
        registry["active"][model_type] = model_name
        self._save_registry(registry)

        return True
    
    def delete_model(self, model_type: str, model_name: str) -> bool:
        registry = self._load_registry()
        if registry.get("active", {}).get(model_type) == model_name:
            raise ValueError("Cannot delete active model")

        model_path = self.models_dir / model_type / model_name
        config_path = self.models_dir / model_type / f"{Path(model_name).stem}_config.json"

        if model_path.exists():
            model_path.unlink()
            if config_path.exists():
                config_path.unlink()
            return True

        return False
