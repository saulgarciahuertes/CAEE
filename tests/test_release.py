"""Offline release checks; no research data, model loads, or external requests."""

import ast
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def notebook_code(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def python_source(source):
    return "\n".join(re.sub(r"^(\s*)!", r"\1# shell: ", line) for line in source.splitlines())


def isolated_functions(source, names, namespace=None):
    nodes = [node for node in ast.parse(python_source(source)).body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Expected source functions were not found")
    scope = {"re": re}
    scope.update(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<isolated-release-check>", "exec"), scope)
    return scope


class ReleaseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
        cls.gpt_source = notebook_code(ROOT / "notebooks/evaluation/Evaluation_GEval_GPT_256.ipynb")
        cls.inference_source = notebook_code(ROOT / "notebooks/inference/Inference_Comparison_1024.ipynb")

    def test_expected_notebook_scope(self):
        self.assertEqual(len(self.notebooks), 7)
        self.assertEqual(len(list((ROOT / "notebooks/training").glob("*.ipynb"))), 4)
        self.assertFalse((ROOT / "notebooks/experimental").exists())

    def test_notebooks_have_no_outputs_or_private_metadata(self):
        for path in self.notebooks:
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                self.assertNotIn("widgets", notebook.get("metadata", {}))
                self.assertNotIn("colab", notebook.get("metadata", {}))
                for cell in notebook["cells"]:
                    self.assertFalse(cell.get("attachments"))
                    if cell["cell_type"] == "code":
                        self.assertEqual(cell.get("outputs"), [])
                        self.assertIsNone(cell.get("execution_count"))

    def test_code_cell_syntax(self):
        for path in self.notebooks:
            ast.parse(python_source(notebook_code(path)))

    def test_no_data_weights_or_archives(self):
        prohibited = {".csv", ".tsv", ".xlsx", ".xls", ".pdf", ".arrow", ".parquet", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".h5", ".zip"}
        for path in ROOT.rglob("*"):
            if ".git" in path.relative_to(ROOT).parts or "__pycache__" in path.parts:
                continue
            self.assertFalse(path.is_symlink())
            if path.is_file():
                self.assertNotIn(path.suffix.lower(), prohibited)
                self.assertNotEqual(path.name, ".env")
                self.assertLess(path.stat().st_size, 1024 * 1024)

    def test_no_hardcoded_private_notebook_locations_or_uploads(self):
        for path in self.notebooks:
            source = notebook_code(path)
            self.assertNotIn("/Users/", source)
            self.assertNotIn("/content/drive/MyDrive/", source)
            self.assertNotIn("userdata.get", source)
            self.assertNotIn("push_to_hub(", source)
            self.assertNotIn("display(comparison_df", source)
            self.assertNotIn("pprint.pprint(", source)

    def test_no_credential_patterns(self):
        pattern = re.compile(r"hf_[A-Za-z0-9]{10,}|sk-(?:proj-)?[A-Za-z0-9_-]{15,}|AIza[A-Za-z0-9_-]{20,}")
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.relative_to(ROOT).parts or "__pycache__" in path.parts:
                continue
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), path.name)

    def test_training_sizes_and_internal_split(self):
        for rows in [4096, 5120, 6144, 7168]:
            source = notebook_code(ROOT / f"notebooks/training/ORPO_Training_{rows}.ipynb")
            self.assertIn(f".select(range({rows}))", source)
            self.assertIn("train_test_split(test_size=0.01)", source)
            self.assertIn('base_model = "meta-llama/Meta-Llama-3-8B-Instruct"', source)
            self.assertIn('report_to="none"', source)

    def test_training_configurations(self):
        configurations = json.loads((ROOT / "configurations/training.json").read_text())
        self.assertEqual([entry["selected_rows_before_internal_split"] for entry in configurations], [4096, 5120, 6144, 7168])
        for entry in configurations:
            self.assertEqual(entry["training_base_model"], "meta-llama/Meta-Llama-3-8B-Instruct")
            self.assertEqual(entry["orpo"]["learning_rate"], 8e-6)
            self.assertEqual(entry["orpo"]["num_train_epochs"], 1)
            self.assertEqual(entry["lora"]["r"], 16)
            self.assertEqual(entry["internal_validation_fraction"], 0.01)

    def test_inference_prefix_and_baseline(self):
        self.assertIn('split="test"', self.inference_source)
        self.assertIn("num_rows_selected = 1024", self.inference_source)
        self.assertIn("test_dataset.select(range(num_rows_selected))", self.inference_source)
        self.assertIn('perform_inference("meta-llama/Meta-Llama-3-8B-Instruct", num_rows_selected, suffix="base")', self.inference_source)
        self.assertIn('os.environ.get("CAEE_FINETUNED_MODELS")', self.inference_source)

    def test_scorers_retain_same_sampling(self):
        for filename in ["Evaluation_GEval_GPT_256.ipynb", "Evaluation_SemScore_256.ipynb"]:
            source = notebook_code(ROOT / "notebooks/evaluation" / filename)
            self.assertIn("num_rows_selected = 256", source)
            self.assertIn(".sample(num_rows_selected, random_state=42)", source)

    def test_external_scoring_requires_opt_in_before_client_creation(self):
        gate = self.gpt_source.index('os.environ.get("CAEE_ALLOW_EXTERNAL_EVALUATION") != "1"')
        client = self.gpt_source.index('client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])')
        self.assertLess(gate, client)
        self.assertIn("raise RuntimeError", self.gpt_source[gate:client])
        example = (ROOT / ".env.example").read_text()
        self.assertIn("CAEE_ALLOW_EXTERNAL_EVALUATION=0", example)

    def test_original_rating_parser_behavior_is_explicit(self):
        extract = isolated_functions(self.gpt_source, ["extract_ratings"])["extract_ratings"]
        self.assertEqual(extract("[coherence: 5, consistency: 4, fluency: 5, relevance: 3]"), (5, 4, 5, 3))
        self.assertEqual(extract("unparseable synthetic response"), (None, None, None, None))
        self.assertEqual(extract("[coherence: 6, consistency: 4, fluency: 5, relevance: 3]")[0], 6)

    def test_original_parse_retries_with_stubbed_responses(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="synthetic unparseable response"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        scope = isolated_functions(self.gpt_source, ["extract_ratings", "g_eval_gpt"], {"client": client})
        result = scope["g_eval_gpt"]("synthetic rubric", "synthetic question", "synthetic reference", "synthetic answer")
        self.assertEqual(result, ("unknown",) * 4)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call["model"] == "gpt-3.5-turbo-0125" for call in calls))

    def test_inference_parsing_uses_only_synthetic_text(self):
        scope = isolated_functions(self.inference_source, ["format_messages", "extract_text_within_curly_braces"])
        self.assertEqual(scope["extract_text_within_curly_braces"]("before {synthetic answer} after"), "synthetic answer")
        self.assertEqual(scope["extract_text_within_curly_braces"]("synthetic plain answer"), "synthetic plain answer")
        self.assertIn("synthetic user", scope["format_messages"]("synthetic system", "synthetic user"))

    def test_provenance_has_only_reported_workflow_notebooks(self):
        provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
        self.assertEqual(len(provenance["notebooks"]), 7)
        self.assertTrue(all(re.fullmatch(r"[a-f0-9]{64}", entry["source_sha256"]) for entry in provenance["notebooks"]))
        self.assertTrue(all((ROOT / entry["release_notebook"]).is_file() for entry in provenance["notebooks"]))

    def test_manifest_matches_files(self):
        manifest = json.loads((ROOT / "MANIFEST.json").read_text())
        actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()
                  and ".git" not in path.relative_to(ROOT).parts and "__pycache__" not in path.parts
                  and path.name != "MANIFEST.json"}
        self.assertEqual(set(manifest["files"]), actual)
        for name, digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)


if __name__ == "__main__":
    unittest.main()
