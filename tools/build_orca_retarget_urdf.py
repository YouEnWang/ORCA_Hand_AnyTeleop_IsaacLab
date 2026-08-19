#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--frames",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()


    source = Path(
        args.source
    ).resolve()

    frame_config = Path(
        args.frames
    ).resolve()

    output = Path(
        args.output
    ).resolve()


    tree = ET.parse(
        source
    )

    robot = tree.getroot()


    with open(
        frame_config,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(
            f
        )


    existing_links = {
        link.attrib["name"]
        for link
        in robot.findall("link")
    }


    for frame_name, info in (
        config["frames"].items()
    ):

        parent = info["parent"]

        if parent not in existing_links:

            raise RuntimeError(
                f"Unknown parent link: "
                f"{parent}"
            )


        if frame_name in existing_links:

            raise RuntimeError(
                f"Frame already exists: "
                f"{frame_name}"
            )


        xyz = " ".join(
            str(float(x))
            for x in info["xyz"]
        )

        rpy = " ".join(
            str(float(x))
            for x in info.get(
                "rpy",
                [0, 0, 0],
            )
        )


        # Empty massless semantic link.
        ET.SubElement(
            robot,
            "link",
            {
                "name": frame_name,
            },
        )


        joint = ET.SubElement(
            robot,
            "joint",
            {
                "name":
                    f"{frame_name}_fixed_joint",

                "type":
                    "fixed",
            },
        )


        ET.SubElement(
            joint,
            "parent",
            {
                "link": parent,
            },
        )


        ET.SubElement(
            joint,
            "child",
            {
                "link": frame_name,
            },
        )


        ET.SubElement(
            joint,
            "origin",
            {
                "xyz": xyz,
                "rpy": rpy,
            },
        )


        print(
            f"[ADD] "
            f"{parent} "
            f"-> {frame_name} "
            f"xyz={xyz}"
        )


    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    try:
        ET.indent(
            tree,
            space="  ",
        )
    except AttributeError:
        pass


    tree.write(
        output,
        encoding="utf-8",
        xml_declaration=True,
    )


    print()
    print(
        "Saved retarget URDF:",
        output,
    )


if __name__ == "__main__":
    main()
