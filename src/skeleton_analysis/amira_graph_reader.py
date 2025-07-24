from typing import List

import numpy as np


class AmiraGraphReader:
    """Class to read an Amira ASCII spatial graph file efficiently."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._num_vertices = 0
        self._num_edges = 0
        self._num_points = 0
        self._file_lines = self._read_file()  # Read file once
        self.file_intro, self._vertex_data, self._edge_data, self._point_data = (
            self._parse_file()
        )
        self._structured_data = self.get_structured_data()


    @property
    def vertex_data(self):
        """Returns the structured vertex data dictionary."""
        return self.structured_data.get("VERTEX", {})

    @property
    def edge_data(self):
        """Returns the structured edge data dictionary."""
        return self.structured_data.get("EDGE", {})

    @property
    def point_data(self):
        """Returns the structured point data dictionary."""
        return self.structured_data.get("POINT", {})

    @property
    def structured_data(self):
        return self._structured_data


    @property
    def num_vertices(self):
        return self._num_vertices

    @property
    def num_edges(self):
        return self._num_edges

    @property
    def num_points(self):
        return self._num_points

    def _read_file(self):
        """
        Reads the file once and returns a list of its lines.

        Returns:
            list of str: The file content split into lines.
        """
        with open(self.filepath, "r", encoding="latin1") as file:
            return file.readlines()

    def write_file(self, output_file: str):
        """
        Writes the modified data back to an Amira ASCII file.

        Parameters:
            output_file (str): The path where the modified file should be saved.
        """
        with open(output_file, "w") as file:
            # 1. Write the file intro (metadata and headers)
            for line in self.file_intro:
                file.write(line + "\n")

            file.write("\n")

            # 2. Write vertex, edge, and point sections with correct @ indices
            data_counter = 1  # Keeps track of @X indices

            for entity_type, entity_list in [
                ("VERTEX", self._vertex_data),
                ("EDGE", self._edge_data),
                ("POINT", self._point_data),
            ]:
                for entity in entity_list:
                    # Assign incremental @X value without relying on previous keys
                    file.write(
                        f"{entity_type} {{ {entity['data_type']} {entity['attribute_name']} }} @{data_counter}\n"
                    )
                    entity["data_key"] = (
                        f"@{data_counter}"  # Update entity with new @X reference
                    )
                    data_counter += 1  # Increment for the next section

            # 3. Write the data section marker
            file.write("\n# Data section follows\n")

            # 4. Write numerical data using incremental @X indices
            data_counter = 1  # Reset counter for writing data blocks
            for entity_list in [self._vertex_data, self._edge_data, self._point_data]:
                for entity in entity_list:
                    if data_counter != 1:
                        file.write("\n")
                    file.write(
                        f"@{data_counter}\n"
                    )  # Write key identifier (e.g., @1, @2, etc.)
                    for row in entity["data"]:
                        file.write(
                            " ".join(map(str, row)) + "\n"
                        )  # Write numerical data
                    data_counter += 1  # Increment for the next block

        print(f"File successfully written to: {output_file}")

    def _parse_file(self):
        """
        Parses the file to extract definitions and data for vertices, edges, and points.

        - Extracts metadata (`define` sections) separately.
        - Stores all lines in `file_intro` up to the first VERTEX header (excluding it).
        - Organizes data into structured dictionaries.
        """
        file_intro = []  # Stores metadata up to the first VERTEX line (excluding it)
        entity_data = {"VERTEX": [], "EDGE": [], "POINT": []}  # Main storage

        for line in self._file_lines:
            if line.startswith("VERTEX"):
                break
            # Store metadata up to (but not including) the first VERTEX header
            file_intro.append(line.rstrip("\n"))

        for line in self._file_lines:
            stripped_line = line.strip()

            # Stop processing when reaching the data section
            if "# Data section follows" in stripped_line:
                break  # Exit early since we only process headers

            # Process 'define' lines to extract counts
            if stripped_line.startswith("define VERTEX"):
                self._num_vertices = int(stripped_line.split()[-1])
            elif stripped_line.startswith("define EDGE"):
                self._num_edges = int(stripped_line.split()[-1])
            elif stripped_line.startswith("define POINT"):
                self._num_points = int(stripped_line.split()[-1])

            # Process actual entity attribute definitions (VERTEX, EDGE, POINT)
            parts = stripped_line.split()
            if not len(parts) >= 5:
                continue
            if parts[0] in ["VERTEX", "EDGE", "POINT"]:
                entity_type = parts[0]  # 'VERTEX', 'EDGE', or 'POINT'
                data_type = parts[2]  # 'float' or 'int'
                attribute_name = " ".join(
                    parts[3:-2]
                )  # Handle multi-word attribute names
                data_key = parts[-1]  # e.g., '@1'

                entity_data[entity_type].append(
                    {
                        "data_type": data_type,
                        "attribute_name": attribute_name,
                        "data_key": data_key,
                        "data": self._extract_data_from_key(
                            data_key
                        ),  # Extract actual data
                    }
                )

        return (
            file_intro,
            entity_data["VERTEX"],
            entity_data["EDGE"],
            entity_data["POINT"],
        )

    def _extract_data_from_key(self, key: str) -> List[np.ndarray]:
        """
        Extracts numerical data associated with a given key from the data section.

        Parameters:
            key (str): The identifier for the data block (e.g., '@1').

        Returns:
            List[np.ndarray]: A list of NumPy arrays where each array represents a row of numerical data.
        """
        data: List[np.ndarray] = []
        collecting = False
        for line in self._file_lines:
            stripped_line = line.strip()
            if stripped_line == key:
                collecting = True
                continue
            if collecting:
                # Stop collecting if we hit an empty line or a new header/marker.
                if (
                    not stripped_line
                    or stripped_line.startswith("#")
                    or stripped_line.startswith("@")
                    or stripped_line.split()[0] in ["VERTEX", "EDGE", "POINT"]
                ):
                    break
                try:
                    values = np.array(list(map(float, stripped_line.split())))
                    data.append(values)
                except ValueError:
                    break
        return data

    def add_EDGE_data(
        self,
        new_key: str,
        data_type: str,
        new_values: List[List[int]],
    ):
        """
        Adds new edge data to the `_edge_data` dictionary.

        Parameters:
            new_key (str): The label for the new EDGE data (e.g., 'NewConnectivity').
            new_values (list of lists): New edge connectivity values to be added.
        """

        new_values = [np.atleast_1d(n) for n in new_values]

        # Warn if the key already exists
        if any(edge["attribute_name"] == new_key for edge in self._edge_data):
            print(f"Warning: EDGE key '{new_key}' already exists. Overwriting.")

        # Append new edge data
        self._edge_data.append(
            {
                "data_type": data_type,  # EDGE data is typically integer
                "attribute_name": new_key,
                "data_key": f"@{len(self._edge_data) + 1}",  # Assigning a unique @X identifier
                "data": new_values,
            }
        )

        print(f"Added new EDGE data '{new_key}' successfully.")


    def remove_EDGE_data(self, key: str):
        """
        Removes an EDGE data field by its attribute name.
        """
        self._edge_data = [e for e in self._edge_data if e["attribute_name"] != key]
        print(f"Removed EDGE data '{key}' (if it existed).")


    def get_structured_data(self):
        """
        Returns all parsed data (vertex, edge, point) as a structured dictionary.
        Format:
        {
            "VERTEX": {attribute_name: data},
            "EDGE": {attribute_name: data},
            "POINT": {attribute_name: data}
        }
        """
        structured = {}

        for category, data_list in [("VERTEX", self._vertex_data),
                                    ("EDGE", self._edge_data),
                                    ("POINT", self._point_data)]:
            structured[category] = {}
            for entry in data_list:
                name = entry["attribute_name"]
                structured[category][name] = entry["data"]
        
        return structured
