import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from typing import Union


def get_tracker_hits_schema() -> pa.Schema:
    """
    Returns the expected PyArrow schema for tracker_hits files.
    """
    return pa.schema(
        [
            pa.field("event_id", pa.uint32()),
            pa.field("x", pa.list_(pa.float32())),
            pa.field("y", pa.list_(pa.float32())),
            pa.field("z", pa.list_(pa.float32())),
            pa.field("true_x", pa.list_(pa.float32())),
            pa.field("true_y", pa.list_(pa.float32())),
            pa.field("true_z", pa.list_(pa.float32())),
            pa.field("time", pa.list_(pa.float32())),
            pa.field("particle_id", pa.list_(pa.uint64())),
            pa.field("detector", pa.list_(pa.uint8())),
            pa.field("volume_id", pa.list_(pa.uint8())),
            pa.field("layer_id", pa.list_(pa.uint16())),
            pa.field("surface_id", pa.list_(pa.uint32())),
        ]
    )


def get_particles_schema() -> pa.Schema:
    """
    Returns the expected PyArrow schema for particles files.
    """
    return pa.schema(
        [
            pa.field("event_id", pa.uint32()),
            pa.field("particle_id", pa.list_(pa.uint64())),
            pa.field("pdg_id", pa.list_(pa.int64())),
            pa.field("mass", pa.list_(pa.float32())),
            pa.field("energy", pa.list_(pa.float32())),
            pa.field("charge", pa.list_(pa.float32())),
            pa.field("vx", pa.list_(pa.float32())),
            pa.field("vy", pa.list_(pa.float32())),
            pa.field("vz", pa.list_(pa.float32())),
            pa.field("time", pa.list_(pa.float32())),
            pa.field("px", pa.list_(pa.float32())),
            pa.field("py", pa.list_(pa.float32())),
            pa.field("pz", pa.list_(pa.float32())),
            pa.field("perigee_d0", pa.list_(pa.float32())),
            pa.field("perigee_z0", pa.list_(pa.float32())),
            pa.field("vertex_primary", pa.list_(pa.uint16())),
            pa.field("parent_id", pa.list_(pa.int64())),
            pa.field("primary", pa.list_(pa.bool_())),
        ]
    )


def validate_file_schema(
    file_path: str, expected_schema: pa.Schema, event_id: Union[int, str] = None
) -> None:
    """
    Validate schema of a single file, raising SchemaValidationError on failure.

    Args:
        file_path: Path to the Parquet file
        expected_schema: Expected PyArrow schema
        event_id: Optional event ID to include in error message

    Raises:
        SchemaValidationError: If validation fails
    """
    actual_schema = pq.read_schema(file_path)

    if expected_schema != actual_schema:
        event_info = f" (event_id={event_id})" if event_id is not None else ""
        raise SchemaValidationError(
            f"Schema mismatch in {file_path}{event_info}.\n"
            f"Expected: {expected_schema}\n"
            f"Got: {actual_schema}"
        )


class SchemaValidationError(Exception):
    pass


def load_explode_parquet(parquet_path, event_id):
    """
    Load and explode parquet data for a specific event.

    Uses lazy evaluation with predicate pushdown for efficiency.
    Auto-detects list columns and explodes them.

    Args:
        parquet_path: Path to Parquet file
        event_id: Event ID to load (required, never None)

    Returns:
        pandas DataFrame with flat format (one row per hit/particle)
    """
    df = pl.scan_parquet(parquet_path).filter(pl.col("event_id") == event_id).collect()

    # Auto-detect list columns and explode
    list_cols = [c for c in df.columns if c != "event_id" and df[c].dtype == pl.List]
    if list_cols:
        df = df.explode(list_cols)

    return df.to_pandas()
