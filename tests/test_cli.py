"""Tests for the command line: one exit code per error family, and stage selection."""
import contextlib
import io
import logging
import unittest
from pathlib import Path
from unittest import mock

from data_factory import credit_pipeline, olist_pipeline, pipeline
from data_factory.cli import main
from data_factory.exceptions import (
    ExitCode,
    InvalidResponseError,
    NetworkError,
    QuarantineThresholdError,
    SourceSchemaChangedError,
)

PROJECT_CONFIG = str(Path(__file__).resolve().parents[1] / "factory.toml")


def fake_stages(**behaviour):
    """Stages of both datasets that record their calls ('olist:' prefix for Olist);
    a stage given an exception raises it."""
    calls = []

    def make(name):
        def stage(context):
            calls.append(name)
            if name in behaviour:
                raise behaviour[name]
        return stage

    nyc311 = {name: make(name) for name in pipeline.STAGE_NAMES}
    olist = {name: make(f"olist:{name}") for name in olist_pipeline.STAGE_NAMES}
    credit = {name: make(f"credit:{name}") for name in credit_pipeline.STAGE_NAMES}
    return nyc311, olist, credit, calls


class CliTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)   # keep test output readable

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def run_cli(self, *args, **behaviour):
        stages, olist_stages, credit_stages, calls = fake_stages(**behaviour)
        with mock.patch.dict(pipeline.STAGES, stages), \
             mock.patch.dict(olist_pipeline.STAGES, olist_stages), \
             mock.patch.object(olist_pipeline, "record_run"), \
             mock.patch.dict(credit_pipeline.STAGES, credit_stages), \
             mock.patch.object(credit_pipeline, "record_run"), \
             mock.patch("data_factory.cli.ensure_database"), \
             mock.patch.object(pipeline, "record_run") as self.record_run:
            code = main(["--config", PROJECT_CONFIG, *args])
        return code, calls

    def test_full_run_calls_every_stage_in_order(self):
        code, calls = self.run_cli("run")
        self.assertEqual(code, ExitCode.OK)
        self.assertEqual(calls, list(pipeline.STAGE_NAMES)
                         + [f"olist:{name}" for name in olist_pipeline.STAGE_NAMES]
                         + [f"credit:{name}" for name in credit_pipeline.STAGE_NAMES])

    def test_dataset_option_runs_only_that_dataset(self):
        code, calls = self.run_cli("run", "--dataset", "olist", "--stages", "load,fetch")
        self.assertEqual(code, ExitCode.OK)
        self.assertEqual(calls, ["olist:fetch", "olist:load"])

    def test_stage_selection_keeps_factory_order(self):
        code, calls = self.run_cli("run", "--dataset", "nyc311", "--stages", "clean,fetch")
        self.assertEqual(code, ExitCode.OK)
        self.assertEqual(calls, ["fetch", "clean"])

    def test_unknown_stage_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--config", PROJECT_CONFIG, "run", "--stages", "bake"])
        self.assertEqual(caught.exception.code, 2)

    def test_check_config_runs_no_stage(self):
        code, calls = self.run_cli("check-config")
        self.assertEqual((code, calls), (ExitCode.OK, []))

    def test_each_error_family_has_its_own_exit_code(self):
        cases = {
            NetworkError("down"): ExitCode.NETWORK,
            InvalidResponseError("html"): ExitCode.INVALID_RESPONSE,
            QuarantineThresholdError("too many"): ExitCode.QUARANTINE_THRESHOLD,
            SourceSchemaChangedError("column gone"): ExitCode.SOURCE_SCHEMA_CHANGED,
            RuntimeError("bug"): ExitCode.UNEXPECTED,
        }
        for error, expected in cases.items():
            with self.subTest(error=type(error).__name__):
                code, _ = self.run_cli("run", fetch=error)
                self.assertEqual(code, expected)

    def test_failure_stops_later_stages(self):
        _, calls = self.run_cli("run", validate=QuarantineThresholdError("too many"))
        self.assertEqual(calls, ["fetch", "validate"])

    def test_every_run_is_recorded_successful_or_not(self):
        self.run_cli("run")
        self.assertEqual(self.record_run.call_args.kwargs["status"], "succeeded")
        self.run_cli("run", clean=NetworkError("down"))
        self.assertEqual(self.record_run.call_args.kwargs["status"], "failed")
        self.assertIn("NetworkError", self.record_run.call_args.kwargs["error"])

    def test_missing_config_file_gives_config_exit_code(self):
        self.assertEqual(main(["--config", "no/such/file.toml", "check-config"]), ExitCode.CONFIG)

    def test_exit_codes_are_distinct(self):
        values = [code.value for code in ExitCode]
        self.assertEqual(len(values), len(set(values)))


if __name__ == "__main__":
    unittest.main()
