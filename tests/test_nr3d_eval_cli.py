import unittest

from eval_helpers import _build_nr3d_eval_arg_parser


class Nr3dEvalCliTest(unittest.TestCase):
    def test_parser_defaults_to_unfiltered_queries(self) -> None:
        parser = _build_nr3d_eval_arg_parser()

        args = parser.parse_args(["--scene_dir", "/tmp/scene0011_00"])

        self.assertFalse(args.use_filtered_queries)

    def test_parser_accepts_use_filtered_queries_flag(self) -> None:
        parser = _build_nr3d_eval_arg_parser()

        args = parser.parse_args(
            ["--scene_dir", "/tmp/scene0011_00", "--use_filtered_queries"]
        )

        self.assertTrue(args.use_filtered_queries)


if __name__ == "__main__":
    unittest.main()
