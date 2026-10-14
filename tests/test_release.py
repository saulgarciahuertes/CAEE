"""Offline release checks; no research data, model loads, or external requests."""

import ast
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unicodedata
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
        cls.gpt_source = notebook_code(ROOT / "notebooks/evaluation/Evaluate_Answer_Quality_with_GPT_256_Samples.ipynb")
        cls.inference_source = notebook_code(ROOT / "notebooks/inference/Generate_Base_and_Fine_Tuned_Llama_3_Answers_1024_Test_Rows.ipynb")
        preparation = json.loads((ROOT / "notebooks/preparation/Redact_CVs_Augment_Survey_Data_and_Build_ORPO_Dataset.ipynb").read_text())
        cls.preparation_cells = ["".join(cell["source"]) for cell in preparation["cells"] if cell["cell_type"] == "code"]

    def test_expected_notebook_scope(self):
        self.assertEqual(len(self.notebooks), 8)
        self.assertEqual(len(list((ROOT / "notebooks/training").glob("*.ipynb"))), 4)
        self.assertFalse((ROOT / "notebooks/experimental").exists())

    def test_notebook_names_and_guide(self):
        guide = (ROOT / "docs/NOTEBOOK_GUIDE.md").read_text()
        for path in self.notebooks:
            self.assertTrue(path.name.startswith(("Redact_CVs_", "Fine_Tune_Llama_3_8B_",
                                                 "Generate_Base_and_Fine_Tuned_", "Evaluate_")))
            self.assertIn(path.name, guide)

    def test_provenance_release_paths_match_notebooks(self):
        provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
        recorded = {entry["release_notebook"] for entry in provenance["notebooks"]}
        actual = {str(path.relative_to(ROOT)) for path in self.notebooks}
        self.assertEqual(recorded, actual)

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
            source = notebook_code(ROOT / f"notebooks/training/Fine_Tune_Llama_3_8B_with_ORPO_{rows}_Selected_Rows.ipynb")
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
        for filename in ["Evaluate_Answer_Quality_with_GPT_256_Samples.ipynb", "Evaluate_Semantic_Similarity_with_SemScore_256_Samples.ipynb"]:
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
        self.assertEqual(len(provenance["notebooks"]), 8)
        self.assertTrue(all(re.fullmatch(r"[a-f0-9]{64}", entry["source_sha256"]) for entry in provenance["notebooks"]))
        self.assertTrue(all((ROOT / entry["release_notebook"]).is_file() for entry in provenance["notebooks"]))

    def test_preparation_api_guards_with_stub_client(self):
        calls = []

        def create_client(**kwargs):
            calls.append(kwargs)
            return "synthetic client"

        environments = [{}, {"CAEE_ALLOW_EXTERNAL_PREPARATION": "1"},
                        {"CAEE_ALLOW_EXTERNAL_PREPARATION": "1", "CAEE_PREPARATION_INPUTS_REVIEWED": "1"}]
        for environment in environments:
            factory = isolated_functions(self.preparation_cells[0], ["preparation_client"],
                                         {"os": SimpleNamespace(environ=environment), "OpenAI": create_client})["preparation_client"]
            with self.assertRaises(RuntimeError):
                factory()
        self.assertEqual(calls, [])
        environment = {"CAEE_ALLOW_EXTERNAL_PREPARATION": "1", "CAEE_PREPARATION_INPUTS_REVIEWED": "1", "OPENAI_API_KEY": "synthetic-test-value"}
        factory = isolated_functions(self.preparation_cells[0], ["preparation_client"],
                                     {"os": SimpleNamespace(environ=environment), "OpenAI": create_client})["preparation_client"]
        self.assertEqual(factory(), "synthetic client")
        self.assertEqual(len(calls), 1)
        example = (ROOT / ".env.example").read_text()
        self.assertIn("CAEE_ALLOW_EXTERNAL_PREPARATION=0", example)
        self.assertIn("CAEE_PREPARATION_INPUTS_REVIEWED=0", example)
        for cell in [self.preparation_cells[2], self.preparation_cells[4]]:
            self.assertLess(cell.index("client = preparation_client()"), cell.index("client.chat.completions.create("))

    def test_preparation_positional_schema_guard(self):
        validate = isolated_functions(self.preparation_cells[0], ["validate_master_schema"])["validate_master_schema"]
        columns = [f"synthetic field {index}" for index in range(113)]
        configuration = json.loads((ROOT / "configurations/preparation.json").read_text())
        for name, position in configuration["expected_master_identifier_positions_zero_based"].items():
            columns[position] = name
        validate(SimpleNamespace(columns=columns))
        with self.assertRaises(ValueError):
            validate(SimpleNamespace(columns=columns[1:]))
        columns[112] = "synthetic unexpected field"
        with self.assertRaises(ValueError):
            validate(SimpleNamespace(columns=columns))

    def test_preparation_presidio_configuration_with_stub_engines(self):
        requests = []

        def analyze(**kwargs):
            requests.append(kwargs)
            return ["synthetic entity"]

        def anonymize(**kwargs):
            self.assertEqual(kwargs["analyzer_results"], ["synthetic entity"])
            return SimpleNamespace(text="synthetic redacted text")

        scope = isolated_functions(self.preparation_cells[1], ["anonymize"], {
            "AnalyzerEngine": lambda: SimpleNamespace(analyze=analyze),
            "AnonymizerEngine": lambda: SimpleNamespace(anonymize=anonymize)})
        self.assertEqual(scope["anonymize"]("synthetic\ntext"), "synthetic redacted text")
        self.assertEqual(requests[0]["text"], "synthetic text")
        self.assertEqual(requests[0]["entities"], ["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "URL"])
        self.assertEqual(requests[0]["language"], "en")

    def test_preparation_literal_name_behavior_retained(self):
        replace = isolated_functions(self.preparation_cells[1], ["replace_names"])["replace_names"]
        self.assertEqual(replace("Alice Example", "Alice Example"), "<NAME> <NAME>")
        self.assertEqual(replace("ALICE EXAMPLE", "Alice Example"), "ALICE EXAMPLE")

    def test_preparation_narrative_requests_match_prompt_fixture(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="synthetic narrative"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        generate = isolated_functions(self.preparation_cells[2], ["generate_descriptions"],
                                      {"pd": SimpleNamespace(notna=lambda value: value is not None)})["generate_descriptions"]
        self.assertEqual(generate(["Synthetic University", None], ["Institution", "Omitted field"], "Synthetic task", client),
                         ["synthetic narrative"] * 3)
        prompts = json.loads((ROOT / "prompts/preparation_prompts.json").read_text())
        self.assertEqual(len(calls), 3)
        for call, suffix in zip(calls, prompts["style_suffixes"]):
            self.assertEqual(call["model"], "gpt-3.5-turbo-0125")
            self.assertEqual(call["max_tokens"], 500)
            self.assertEqual(call["messages"][0]["content"], prompts["narrative_system_message"])
            self.assertEqual(call["messages"][1]["content"], "Synthetic task" + suffix + ": Institution: Synthetic University")

    def test_preparation_contrast_requests_and_unused_helper(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="synthetic contrast"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        scope = isolated_functions(self.preparation_cells[4], ["generate_wrong_answer", "replace_yes_with_no"])
        self.assertEqual(scope["generate_wrong_answer"]("synthetic preferred answer", client), "synthetic contrast")
        prompts = json.loads((ROOT / "prompts/preparation_prompts.json").read_text())
        self.assertEqual(calls[0]["messages"][1]["content"], prompts["rejected_answer_prompt_prefix"] + "synthetic preferred answer")
        self.assertEqual(calls[0]["messages"][0]["content"], prompts["rejected_answer_system_message"])
        self.assertEqual(calls[0]["model"], "gpt-3.5-turbo")
        self.assertEqual(calls[0]["temperature"], 0.8)
        self.assertEqual(calls[0]["max_tokens"], 500)
        self.assertEqual(scope["replace_yes_with_no"]("Yes"), "No")
        self.assertEqual(scope["replace_yes_with_no"]("No"), "")
        tree = ast.parse(self.preparation_cells[4])
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                             and node.func.id == "replace_yes_with_no" for node in ast.walk(tree)))

    def test_preparation_instruction_combinations_and_cleanup_alternatives(self):
        construct = isolated_functions(self.preparation_cells[3], ["create_instruction"])["create_instruction"]
        self.assertEqual(construct({"cv": "synthetic CV", "context": "synthetic context", "answer": "synthetic answer"},
                                   "cv", "context", "answer", "synthetic question"),
                         ("synthetic CV", "synthetic context synthetic question", "synthetic answer"))
        config = json.loads((ROOT / "configurations/preparation.json").read_text())
        combinations = config["instruction_construction"]
        self.assertEqual(combinations["cv_variants"] * combinations["context_variants"] * combinations["answer_variants"] * combinations["tasks"], 162)
        self.assertIn('file_path = output_dir / "orpo_data.csv"', self.preparation_cells[5])
        self.assertIn('orpo_data_path = output_dir / "orpo_data.csv"', self.preparation_cells[6])
        self.assertNotIn("train_test_split(", "\n".join(self.preparation_cells))

    def test_preparation_placeholder_mismatch_is_retained_not_hidden(self):
        scope = isolated_functions(self.preparation_cells[6], ["remove_accents", "clean_text"], {
            "unicodedata": unicodedata, "pd": SimpleNamespace(isna=lambda value: False),
            "phone_remove_counter": 0, "email_remove_counter": 0, "url_remove_counter": 0})
        self.assertEqual(scope["clean_text"]("synthetic <PHONE_NUMBER>"), "synthetic <PHONE_NUMBER>")
        self.assertEqual(scope["clean_text"]("synthetic <EMAIL_ADDRESS>"), "synthetic ")

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
