from fastapi import FastAPI

from app import __version__
from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Trakheesi Compliance Detector",
        description=(
            "UAE real-estate listing compliance 2026: Trakheesi permit validation, "
            "YOLOv8 watermark/logo detection (ONNX Runtime), and pHash duplicate photos."
        ),
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()
