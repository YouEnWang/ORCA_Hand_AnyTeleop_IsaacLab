#!/usr/bin/env python3

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import yaml


FINGERS = {
    "index": {
        "mesh": "I-FingerTipAssembly.stl",
        "parent": "I-FingerTipAssembly_ec49c16c",
    },

    "middle": {
        "mesh": "M-FingerTipAssembly.stl",
        "parent": "M-FingerTipAssembly_34afb748",
    },

    "ring": {
        "mesh": "M-FingerTipAssembly.stl",
        "parent": "M-FingerTipAssembly_424a8e75",
    },

    "pinky": {
        "mesh": "P-FingerTipAssembly.stl",
        "parent": "P-FingerTipAssembly_cd219176",
    },

    "thumb": {
        "mesh": "T-DP.stl",
        "parent": "T-DP_b7429e50",
    },
}


def load_stl_vertices(path: Path):

    data = path.read_bytes()

    vertices = None


    # ----------------------------------------------------------
    # Try binary STL
    # ----------------------------------------------------------

    if len(data) >= 84:

        triangle_count = struct.unpack(
            "<I",
            data[80:84],
        )[0]

        expected_size = (
            84
            +
            triangle_count * 50
        )

        if expected_size == len(data):

            result = []

            offset = 84

            for _ in range(
                triangle_count
            ):

                # skip normal
                offset += 12

                for _ in range(3):

                    vertex = struct.unpack(
                        "<fff",
                        data[
                            offset:
                            offset + 12
                        ],
                    )

                    result.append(
                        vertex
                    )

                    offset += 12

                # attribute byte count
                offset += 2


            vertices = np.asarray(
                result,
                dtype=np.float64,
            )


    # ----------------------------------------------------------
    # ASCII fallback
    # ----------------------------------------------------------

    if vertices is None:

        text = data.decode(
            "utf-8",
            errors="ignore",
        )

        result = []

        for line in text.splitlines():

            line = line.strip()

            if not line.startswith(
                "vertex "
            ):
                continue

            parts = line.split()

            result.append(
                [
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                ]
            )


        vertices = np.asarray(
            result,
            dtype=np.float64,
        )


    if vertices.size == 0:

        raise RuntimeError(
            f"No STL vertices: {path}"
        )


    # ORCA URDF mesh scale = 0.001.
    vertices *= 0.001


    # Remove repeated triangle vertices.
    vertices = np.unique(
        np.round(
            vertices,
            decimals=7,
        ),
        axis=0,
    )


    return vertices


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--assets",
        required=True,
        help=(
            "ORCA right-hand mesh directory, "
            "e.g. .../v2/models/assets/right"
        ),
    )

    parser.add_argument(
        "--slab-mm",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()


    assets = Path(
        args.assets
    ).resolve()

    slab = (
        args.slab_mm
        / 1000.0
    )


    output = {
        "method": (
            "distal_positive_z_mesh_cap"
        ),

        "slab_mm": float(
            args.slab_mm
        ),

        "frames": {},
    }


    print("=" * 80)
    print("ORCA fingertip mesh diagnostics")
    print("=" * 80)


    for finger, info in FINGERS.items():

        path = (
            assets
            /
            info["mesh"]
        )

        vertices = load_stl_vertices(
            path
        )


        minimum = np.min(
            vertices,
            axis=0,
        )

        maximum = np.max(
            vertices,
            axis=0,
        )

        extent = (
            maximum
            -
            minimum
        )


        # PIP/DIP axes in this URDF are approximately local -Y,
        # while the distal finger geometry proceeds mainly
        # along local +Z.
        z_max = float(
            maximum[2]
        )


        distal = vertices[
            vertices[:, 2]
            >= z_max - slab
        ]


        tip = np.mean(
            distal,
            axis=0,
        )


        print()
        print(
            f"[{finger.upper()}]"
        )

        print(
            "mesh   :",
            path,
        )

        print(
            "min[m] :",
            minimum,
        )

        print(
            "max[m] :",
            maximum,
        )

        print(
            "extent :",
            extent,
        )

        print(
            "candidate tip:",
            tip,
        )

        print(
            "distal vertices:",
            len(distal),
        )


        frame_name = (
            f"retarget_{finger}_tip"
        )


        output["frames"][
            frame_name
        ] = {

            "parent": info[
                "parent"
            ],

            "xyz": [
                float(x)
                for x in tip
            ],

            "rpy": [
                0.0,
                0.0,
                0.0,
            ],
        }


    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            output,
            f,
            sort_keys=False,
        )


    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
