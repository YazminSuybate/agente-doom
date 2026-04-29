from __future__ import annotations

import argparse

from doom_agent.config import PROFILE_NAMES, SCENARIO_NAMES
from doom_agent.services.evaluator import evaluate


def add_evaluate_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Nombre del checkpoint sin extension .zip, o ruta al archivo.",
    )
    parser.add_argument(
        "--config",
        choices=PROFILE_NAMES,
        default="default",
        help="Perfil base para derivar el checkpoint cuando no se pasa '--checkpoint'.",
    )
    parser.add_argument(
        "--select",
        choices=("best", "last", "exact"),
        default="best",
        help="Politica para resolver el checkpoint: mejor modelo, ultimo estado o nombre exacto.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Cantidad maxima de pasos de evaluacion.",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_NAMES,
        default=None,
        help="Sobrescribe el escenario de evaluacion. Util para checkpoints legacy o pruebas cruzadas.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = add_evaluate_arguments(
        argparse.ArgumentParser(description="Evalua un checkpoint entrenado.")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        checkpoint_name=args.checkpoint,
        steps=args.steps,
        profile_name=args.config,
        checkpoint_selection=args.select,
        scenario_name=args.scenario,
    )


if __name__ == "__main__":
    main()
