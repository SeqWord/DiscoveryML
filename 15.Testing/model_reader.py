#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read informtion from PKL model files."""

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR / ".." / "lib"
if not LIB_DIR.is_dir():
    raise SystemExit(
        f"ERROR: Required classifier library directory does not exist: {LIB_DIR}"
    )
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

try:
    from model_bundle import ModelBundle
except Exception as exc:
    raise SystemExit(
        f"ERROR: Cannot import model_bundle.py and blast.py from {LIB_DIR}: {exc}"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read informtion from PKL model files."
        )
    )
    parser.add_argument(
        "project_folder",
        help="Folder under input_folder containing FASTA/GenBank genome files."
    )
    parser.add_argument("-i", "--input_folder", default="input")
    parser.add_argument("-o", "--output_folder", default="output")
    parser.add_argument("-d", "--delimiter", help="Symbol used to separate hierarchical cluster names", default="|")
    parser.add_argument(
        "--model_file", "-m", default="model.pkl",
        help="Model PKL path relative to classifier.py (default: model/model.pkl)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    in_path = SCRIPT_DIR / args.input_folder / args.project_folder
    if not in_path.exists():
        sys.exit(f"ERROR: Input path {in_path} does not exist!")
    
    model_path = in_path / args.model_file
    if not model_path.exists():
        sys.exit(f"ERROR: Model file {model_path} does not exist!")
    
    out_path = SCRIPT_DIR / args.output_folder / args.project_folder
    os.makedirs(str(out_path), exist_ok=True)
    
    try:
        model = ModelBundle.load(model_path)
    except Exception as exc:
        sys.exit(f"ERROR: Cannot load model file {model_path}: {exc}")

    def print_info():
        info = model.get_interface_info()
        print()
        for key, value in info.items():
            print(f"{key}: {value}")
        print()

    def show_graph():
        try:
            model.show_cluster_graph(
                output_folder=out_path,
                delimiter=args.delimiter,
            )
        except Exception as exc:
            print(f"\nERROR: Cannot display cluster graph: {exc}\n")

    commands = [
        ["1. Print info", print_info],
        ["2. Show cluster graph", show_graph],
    ]
    
    while True:
        print("") 
        print("Model reader")
        print("Select the option to perform or enter Q to quit...")
        for command in commands:
            print(command[0])
        print()
        response = input("Enter option number...")
        if response.upper() == "Q":
            sys.exit("\nProgram terminated!")
            
        try:
            option = int(response.strip()) - 1
        except:
            print(f"\n\tERROR: inappropriate option number {response.strip()}. Enter option number in range form 1 to {len(commands)}\n")
            continue
        if option < 0 or option >= len(commands):
            print(f"\n\tERROR: inappropriate option number {response.strip()}. Enter option number in range form 1 to {len(commands)}\n")
            continue
            
        commands[option][1]()
            
    
if __name__ == "__main__":
    main()
