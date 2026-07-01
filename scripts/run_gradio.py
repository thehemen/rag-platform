from pathlib import Path
import os
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.gradio_app import GradioRAGApp, GradioSettings


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    settings = GradioSettings(
        api_url=os.getenv(
            "GRADIO_API_URL",
            "http://localhost:8000",
        ),
        request_timeout=float(
            os.getenv("GRADIO_REQUEST_TIMEOUT", "120"),
        ),
    )

    app = GradioRAGApp(settings=settings)
    demo = app.build()

    demo.launch(
        server_name=os.getenv("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower()
        in {"1", "true", "yes", "y"},
    )


if __name__ == "__main__":
    main()
