"""
This module converts .schem files to CSV format for GDPC structure placement.
"""

import re
import shutil
from pathlib import Path

import nbtlib
import pandas as pd

from consts import BUILD_INPUT_DIR, BUILD_OUTPUT_DIR


def decode_varint_array(data: nbtlib.ByteArray) -> list[int]:
    """
    The Sponge schematic format stores block data as an array of variable-width
    integers. As the nbtlib library is relatively low-level, we need to take
    care of this ourselves when interpreting the bytes.

    :param data: a variable-width integer array
    :return: a list of integers
    """
    result = []

    value = 0
    shift = 0
    for b in data:
        b &= 0xFF
        value |= (b & 0x7F) << shift

        if (b & 0x80) == 0:  # Leading 1 indicates that another byte follows
            result.append(value)
            value = 0
            shift = 0
        else:
            shift += 7

    return result


def parse_block_state(block_name: str) -> tuple[str, dict[str, str]]:
    """
    Split a palette entry into its block ID and block states.

    Example:
        minecraft:oak_sign[rotation=9,waterlogged=false]

    becomes:
        (
            "minecraft:oak_sign",
            {
                "rotation": "9",
                "waterlogged": "false"
            }
        )

    :param block_name: a palette entry
    :return: (block_id, states)
    """
    if "[" not in block_name:
        return block_name, {}

    block_id, state_str = block_name[:-1].split("[", 1)

    states = {}
    for entry in state_str.split(","):
        key, value = entry.split("=", 1)
        states[key] = value

    return block_id, states


def main():
    """Convert .schem files to CSV format for GDPC structure placement."""
    schematics_dir = Path(BUILD_INPUT_DIR)
    output_dir = Path(BUILD_OUTPUT_DIR)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schem_paths = list(schematics_dir.rglob("*.schem"))

    chain_cmd_re = re.compile(r"^minecraft:chain_command_block(\[.*])?$")

    for schem_path in schem_paths:
        try:
            nbt = nbtlib.load(schem_path)
        except Exception as e:
            print(f"Error loading {schem_path}: {e}")
            continue

        try:
            schem = nbt["Schematic"]

            width = int(schem["Width"])
            height = int(schem["Height"])
            length = int(schem["Length"])

            blocks_compound = schem["Blocks"]
            block_palette = blocks_compound["Palette"]
            block_data = decode_varint_array(blocks_compound["Data"])

            index_to_block = [None] * len(block_palette)
            for block_name, tag in block_palette.items():
                index_to_block[int(tag)] = block_name

            block_entity_list = blocks_compound.get("BlockEntities", [])
            # Keys that describe position/identity rather than block-entity state.
            # These are implied by placement position and must be stripped before
            # passing the SNBT to GDPC.
            _META_KEYS = frozenset({"Pos", "Id", "x", "y", "z"})
            block_entities = {}
            for be in block_entity_list:
                pos = tuple(int(v) for v in be["Pos"])
                if "Data" in be:
                    data_nbt = be["Data"]
                else:
                    data_nbt = nbtlib.Compound(
                        {k: v for k, v in be.items() if k not in _META_KEYS}
                    )
                block_entities[pos] = data_nbt

            block_entries = []
            for y in range(height):
                for z in range(length):
                    for x in range(width):
                        index = y * length * width + z * width + x
                        block_index = block_data[index]
                        block_name = index_to_block[block_index]

                        if block_name == "minecraft:air" or chain_cmd_re.match(
                            block_name
                        ):
                            continue

                        block_id, states = parse_block_state(block_name)

                        # GDPC expects block entity data as SNBT.
                        # Remove schematic-specific fields that are implied by
                        # the placement position and block type.
                        data = None
                        pos = (x, y, z)
                        if pos in block_entities:
                            data = block_entities[pos].snbt()

                        block_entries.append(
                            {
                                "Position": pos,
                                "BlockID": block_id,
                                "States": states,
                                "Data": data,
                            }
                        )

            df = pd.DataFrame(
                block_entries,
                columns=[
                    "Position",
                    "BlockID",
                    "States",
                    "Data",
                ],
            )

            output_path = output_dir / schem_path.relative_to(
                schematics_dir
            ).with_suffix(".csv")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

            print(f"Converted {schem_path} ({len(block_entries)} blocks)")

        except Exception as e:
            print(f"Error processing {schem_path}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
