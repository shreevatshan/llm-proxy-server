import io
import time
import unittest
from unittest.mock import patch

from app import diagnostics


def _reset_module_state() -> None:
    """Return the diagnostics singleton to its pre-init state between tests."""
    diagnostics.shutdown()
    thread = diagnostics._thread
    if thread is not None:
        thread.join(timeout=2)
    with diagnostics._cond:
        diagnostics._deadlines.clear()
        diagnostics._fp = None
        diagnostics._max_bytes = 0
        diagnostics._thread = None
        diagnostics._started = False
        diagnostics._shutdown = False


class StallWatchdogTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_module_state()
        self.addCleanup(_reset_module_state)
        self.fp = io.StringIO()

    def _init(self) -> list:
        """Init the watchdog with dump_traceback stubbed; return a calls list."""
        calls = []

        def _fake_dump(*, file, all_threads):
            calls.append((file, all_threads))
            file.write("<stack-dump>\n")

        patcher = patch.object(diagnostics.faulthandler, "dump_traceback", _fake_dump)
        patcher.start()
        self.addCleanup(patcher.stop)
        diagnostics.init_watchdog(self.fp, max_bytes=10 * 1024)
        return calls

    def test_fires_after_deadline_when_never_reset(self):
        calls = self._init()
        diagnostics.arm("req-1", 0.05)
        time.sleep(0.3)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1])  # all_threads=True
        self.assertIn("key='req-1'", self.fp.getvalue())

    def test_no_fire_when_disarmed_before_deadline(self):
        calls = self._init()
        diagnostics.arm("req-2", 0.2)
        time.sleep(0.05)
        diagnostics.disarm("req-2")
        time.sleep(0.3)
        self.assertEqual(calls, [])

    def test_reset_extends_deadline(self):
        calls = self._init()
        diagnostics.arm("req-3", 0.15)
        # Keep resetting before the deadline elapses — should never fire.
        for _ in range(4):
            time.sleep(0.05)
            diagnostics.reset("req-3", 0.15)
        self.assertEqual(calls, [])
        # Stop resetting; now it should fire.
        time.sleep(0.4)
        self.assertEqual(len(calls), 1)

    def test_one_shot_does_not_spam(self):
        calls = self._init()
        diagnostics.arm("req-4", 0.05)
        time.sleep(0.4)
        # A single armed deadline produces exactly one dump, not a stream.
        self.assertEqual(len(calls), 1)

    def test_functions_are_noops_before_init(self):
        # No init_watchdog call: arm/reset/disarm must not raise.
        diagnostics.arm("x", 0.01)
        diagnostics.reset("x", 0.01)
        diagnostics.disarm("x")
        time.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
