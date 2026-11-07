"""Adapt the same frozen source notebook for an ordinary free Colab session."""
import ast
import json
from pathlib import Path

from build_kaggle_notebook import main as build_source


def main():
    build_source()
    notebook = json.loads(Path("notebooks/Free_Phase03_Kaggle.ipynb").read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        source = source.replace("/kaggle/temp/", "/content/sqlrl_temp/")
        source = source.replace("/kaggle/working/sqlrl_free_run", "/content/sqlrl_free_run")
        source = source.replace("Kaggle Settings", "Colab Runtime > Change runtime type")
        source = source.replace("'28000'", "'10800'")
        source = source.replace("# Attach your previous PRIVATE output and enter its sqlrl_free_run path.",
                                "# Upload/extract a prior private output and enter its sqlrl_free_run path.")
        source = source.replace("No DB/archive symlink is placed under /kaggle/working.",
                                "Only results and processed records are included in the output archive.")
        source = source.replace("from kaggle_secrets import UserSecretsClient",
                                "from google.colab import userdata")
        source = source.replace("UserSecretsClient().get_secret('GEMINI_API_KEY')",
                                "userdata.get('GEMINI_API_KEY')")
        if cell["cell_type"] == "markdown":
            source = (
                "# Free Qwen text-to-SQL: Phases 0–3\n"
                "Use the free T4 runtime and keep this notebook private. "
                "The experiment is the same frozen NF4/FP16 profile as the Kaggle notebook. "
                "Runtime files are temporary. The final export cell downloads a private "
                "checkpoint archive; run it before leaving the session. "
                "No Google Drive mount, paid runtime or API billing is enabled. "
                "This notebook has not yet produced a model benchmark.\n"
            )
        cell["source"] = source.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            ast.parse(source)
    notebook["cells"].append({
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": [
            "# Run before leaving this temporary session, including after a paused stage.\n",
            "from google.colab import files\n",
            "archive_path = shutil.make_archive('/content/sqlrl_free_run_checkpoint', 'zip', root_dir=PERSIST.parent, base_dir=PERSIST.name)\n",
            "print('Private results/checkpoints archive:', archive_path)\n",
            "files.download(archive_path)\n",
        ],
    })
    notebook["metadata"]["colab"] = {"name": "Free_Phase03_Colab.ipynb", "provenance": []}
    notebook["metadata"]["accelerator"] = "GPU"
    output = Path("notebooks/Free_Phase03_Colab.ipynb")
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(json.dumps({"notebook": str(output), "notebook_bytes": output.stat().st_size,
                      "source_snapshot_sha256": notebook["metadata"]["source_snapshot_sha256"]}))


if __name__ == "__main__":
    main()
