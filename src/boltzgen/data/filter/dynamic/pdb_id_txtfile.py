from boltzgen.data.data import Record
from boltzgen.data.filter.dynamic.filter import DynamicFilter
from typing import List


class FilterIDFromTXT(DynamicFilter):
    """A filter that filters complexes based on a list of IDs from text files."""

    def __init__(self, paths: List[str], reverse: bool = False):
        """
        Initializes the filter by loading IDs from one or more text files.

        Parameters
        ----------
        paths : List[str]
            A list of file paths to the text files containing IDs (one per line).
        reverse : bool, optional
            If False (default), records with IDs in the files are filtered out.
            If True, only records with IDs in the files are kept.
        """
        self.paths = paths
        self.reverse = reverse

        all_ids = []
        for path in paths:
            with open(path) as f:
                ids_from_file = [
                    s_line.lower() for line in f if (s_line := line.strip())
                ]
                all_ids.extend(ids_from_file)

        self.id_set = set(all_ids)

    def filter(self, record: Record) -> bool:
        """
        Filters a record based on whether its ID is in the loaded set.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            True if the record should be kept, False if it should be discarded.
        """
        is_present = record.id.lower() in self.id_set

        return is_present if self.reverse else not is_present
