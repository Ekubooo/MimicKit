"""Long diagnostic names must remain separate from adjacent numeric columns."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'mimickit'))
from util.logger import Logger


class LoggerColumns(unittest.TestCase):
    def test_long_names_and_values_keep_column_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'log.txt'
            log = Logger()
            try:
                log.configure_output_file(str(path))
                names = ['x'*25, 'Getup_Rollout_Nonstanding_Positive_Adv', 'last']
                values = [10**30, .125, 7]
                for key,value in zip(names,values):
                    log.log(key,value)
                log.write_log()
            finally:
                log.close()
            lines = path.read_text().splitlines()
            self.assertEqual(lines[0].split(),names)
            self.assertEqual(lines[1].split(),list(map(str,values)))


if __name__ == '__main__':
    unittest.main()
