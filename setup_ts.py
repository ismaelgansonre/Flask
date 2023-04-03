from cx_Freeze import setup, Executable

executables = [Executable("YoloImg.py")]

setup(
    name="DetectionPoulets",
    version="1.0",
    description="Détection de poulets",
    executables=executables,
)
