import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.g1_dodgeball.style_balance import validate_log


class StrictTrainingLogTest(unittest.TestCase):
    def check_log(self, body):
        output = pathlib.Path(__file__).resolve().parents[1] / "output"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as folder:
            path = pathlib.Path(folder) / "log.txt"
            path.write_text(body, encoding="utf-8")
            return validate_log(path, 131137536, 2001)

    def test_complete_budget(self):
        result = self.check_log("Iteration Samples Smp_Reward_Mean Sds_Loss_Mean Exp_Prob\n"
                                "0 65536 0.1 1.5 1\n2000 131137536 0.4 0.8 1\n")
        self.assertEqual(result["rows"], 2)

    def test_joined_header_and_extra_last_value_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "column count"):
            self.check_log("Iteration Samples Smp_Reward_Mean Sds_Loss_Mean LongMetricExp_Prob\n"
                           "2000 131137536 0.4 0.8 0.2 nan\n")

    def test_nonfinite_final_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.check_log("Iteration Samples Smp_Reward_Mean Sds_Loss_Mean Exp_Prob\n"
                           "2000 131137536 0.4 0.8 nan\n")

    def test_incomplete_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact sample/update"):
            self.check_log("Iteration Samples Smp_Reward_Mean Sds_Loss_Mean\n"
                           "1999 131072000 0.4 0.8\n")


if __name__ == "__main__":
    unittest.main()
