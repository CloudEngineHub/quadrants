"""Tests for the xdist worker retirement hooks in conftest.py.

Verifies that when a worker is killed via os._exit(1) after a test failure:
1. Failures are not double-counted (no synthetic "worker crashed" report)
2. The session completes even with many failures (--max-worker-restart cap
   does not trigger premature shutdown)

These tests use pytester to run pytest-xdist in a subprocess, so they do
not require GPU hardware.
"""

import pytest

pytest_plugins = ["pytester"]

SUBPROCESS_ARGS = [
    "-p",
    "no:retry",
    "-p",
    "no:rerunfailures",
    "-p",
    "no:nbmake",
    "-p",
    "no:timeout",
    "-p",
    "no:cacheprovider",
    "-o",
    "addopts=",
]


@pytest.fixture
def xdist_project(pytester):
    """Write a minimal conftest that reproduces our worker-retirement hooks."""
    pytester.makeconftest(
        """
        import os
        import pytest

        @pytest.hookimpl(trylast=True)
        def pytest_runtest_logreport(report):
            if not os.environ.get("PYTEST_XDIST_WORKER"):
                return
            if report.outcome not in ("error", "failed"):
                return
            os._exit(1)

        def pytest_handlecrashitem(crashitem, report, sched):
            report.outcome = "passed"
            report.when = "teardown"
            report.longrepr = None
        """
    )
    return pytester


class TestNoDuplicateFailures:
    def test_single_failure_counted_once(self, xdist_project):
        """A single failing test should appear exactly once in the summary."""
        xdist_project.makepyfile(
            """
            def test_pass():
                pass

            def test_fail():
                assert False, "intentional failure"
            """
        )
        result = xdist_project.runpytest_subprocess(
            "-n", "2", "--dist=worksteal", *SUBPROCESS_ARGS, "-v"
        )
        result.assert_outcomes(passed=1, failed=1)

    def test_multiple_failures_counted_correctly(self, xdist_project):
        """Each failing test should be counted exactly once."""
        xdist_project.makepyfile(
            """
            import pytest

            @pytest.mark.parametrize("i", range(4))
            def test_fail(i):
                assert False, f"failure {i}"

            def test_pass():
                pass
            """
        )
        result = xdist_project.runpytest_subprocess(
            "-n",
            "2",
            "--dist=worksteal",
            "--max-worker-restart=999999",
            *SUBPROCESS_ARGS,
            "-v",
        )
        result.assert_outcomes(passed=1, failed=4)


class TestSessionCompletesWithManyFailures:
    def test_no_premature_shutdown(self, xdist_project):
        """With a high --max-worker-restart, all tests should run even if many fail."""
        xdist_project.makepyfile(
            """
            import pytest

            @pytest.mark.parametrize("i", range(20))
            def test_fail(i):
                assert False, f"failure {i}"

            @pytest.mark.parametrize("i", range(5))
            def test_pass(i):
                pass
            """
        )
        result = xdist_project.runpytest_subprocess(
            "-n",
            "2",
            "--dist=worksteal",
            "--max-worker-restart=999999",
            *SUBPROCESS_ARGS,
            "-v",
        )
        result.assert_outcomes(passed=5, failed=20)
        assert "maximum crashed workers reached" not in result.stdout.str()
