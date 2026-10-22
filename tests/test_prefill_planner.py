"""[bdw-prefill] Pure-numpy tests of the grouped-prefill split planner (cpu_tier.pf_plan): no GPU, no native module."""
import ast
import os
import unittest

import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), "..", "glm53", "cpu_tier.py")


def _load_pf_plan():
    """Compile only pf_plan from cpu_tier.py (importing the module needs torch and the native extension)."""
    tree = ast.parse(open(_SRC).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pf_plan")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), _SRC, "exec"), ns)
    return ns["pf_plan"]


pf_plan = _load_pf_plan()


class PlannerTest(unittest.TestCase):
    def test_least_routed_first_and_balance(self):
        counts = np.zeros(288, dtype=np.int64)
        counts[:250] = np.random.default_rng(1).poisson(9.7, 250) + 1
        take, t_cpu = pf_plan(counts, 0.055, 0.047, 0.86, 64)
        cpu = counts[take == 1]
        gpu = counts[(take == 0) & (counts > 0)]
        self.assertTrue(cpu.size and gpu.size)
        self.assertLessEqual(cpu.max(), gpu.min())                  # CPU takes the least-routed experts
        t_gpu = gpu.size * 0.86
        self.assertLessEqual(t_cpu, t_gpu)                          # the CPU finishes before the GPU's PCIe share
        self.assertAlmostEqual(t_cpu, float((0.055 + 0.047 * cpu).sum()), places=6)

    def test_rowcap_and_empty(self):
        counts = np.zeros(288, dtype=np.int64)
        counts[3] = 65
        counts[4] = 2
        take, _ = pf_plan(counts, 0.055, 0.047, 0.86, 64)
        self.assertEqual(take[3], 0)                                # over the native per-expert row cap: GPU
        take, t = pf_plan(np.zeros(288, dtype=np.int64), 0.055, 0.047, 0.86, 64)
        self.assertEqual(int(take.sum()), 0)
        self.assertEqual(t, 0.0)

    def test_single_cold_expert_goes_to_a_faster_cpu(self):
        counts = np.zeros(288, dtype=np.int64)
        counts[7] = 9                                               # CPU 0.478 ms < PCIe 0.86 ms (review 2026-10-22)
        take, t = pf_plan(counts, 0.055, 0.047, 0.86, 64)
        self.assertEqual(int(take[7]), 1)
        self.assertAlmostEqual(t, 0.055 + 0.047 * 9, places=9)

    def test_expensive_cpu_takes_nothing(self):
        counts = np.full(288, 10, dtype=np.int64)
        take, _ = pf_plan(counts, 500.0, 5.0, 0.86, 64)              # one CPU expert > all PCIe time: all stay on the GPU
        self.assertEqual(int(take.sum()), 0)


if __name__ == "__main__":
    unittest.main()
