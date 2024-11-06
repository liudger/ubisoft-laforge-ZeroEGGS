# class ModelManager:
#     def __init__(self):
#         self._models = {}
#         self._device = None
#         self._lock = threading.Lock()
        
#     def get_device(self):
#         if self._device is None:
#             self._device = "cuda" if torch.cuda.is_available() else "cpu"
#         return self._device
        
#     def load_model(self, model_path: Path, model_type: str):
#         with self._lock:
#             key = f"{model_type}_{str(model_path)}"
#             if key not in self._models:
#                 if not model_path.exists():
#                     raise FileNotFoundError(f"Model file not found: {model_path}")
#                 try:
#                     model = torch.load(model_path, map_location=self.get_device())
#                     model.to(self.get_device())
#                     model.eval()
#                     self._models[key] = model
#                 except Exception as e:
#                     raise RuntimeError(f"Error loading model {model_type}: {str(e)}")
#             return self._models[key]
            
#     def cleanup(self):
#         with self._lock:
#             self._models.clear()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()