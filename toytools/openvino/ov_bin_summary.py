import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from openvino.runtime import Core

BYTES_GENERATORS = {
    "ones": lambda length: np.ones((length,), dtype=np.uint8).tobytes(),
    "zeros": lambda length: np.zeros((length,), dtype=np.uint8).tobytes(),
    "random": lambda length: np.random.randint(low=10, high=100, size=(length,), dtype=np.uint8).tobytes(),
}


@dataclass
class _OVBinSummaryPart:
    start: int
    length: int = 0
    op_name: Optional[str] = None
    value: Optional[np.ndarray] = None
    index: int = -1

    @classmethod
    def from_dict(cls, content: dict):
        value = content.pop("value")
        value = np.array(value, dtype=np.uint8)
        return cls(value=value, **content)

    def to_dict(self):
        result = asdict(self)
        if result["value"] is not None:
            result["value"] = result["value"].tolist()
        return result


class OVBinSummary:
    ie = None

    def __init__(self, parts: List[_OVBinSummaryPart]):
        self.parts = parts

    @classmethod
    def from_xml(cls, xml_path: Union[str, Path]):
        xml_path = Path(xml_path).resolve()
        ovmodel = cls._read_ov_model(xml_path)
        reference_array = cls._read_from_bin(xml_path.with_suffix(".bin"))

        parts: List[_OVBinSummaryPart] = []
        cur = 0
        for op in ovmodel.get_ordered_ops():
            if "constant" in str(op.get_type_info()).lower():
                vector = op.get_vector()
                if vector.size > 10:
                    converted_uint8 = np.frombuffer(vector.tobytes(), dtype=np.uint8)
                    starting_index = cls._find_index(reference_array, cur, converted_uint8)
                    assert starting_index >= cur
                    if starting_index > cur:
                        parts.append(
                            _OVBinSummaryPart(
                                start=cur,
                                length=starting_index - cur,
                                value=reference_array[cur:starting_index],
                            )
                        )
                    parts.append(
                        _OVBinSummaryPart(
                            start=starting_index,
                            length=converted_uint8.size,
                            op_name=op.get_name(),
                        )
                    )
                    cur = starting_index + converted_uint8.size

        if cur < reference_array.size:
            parts.append(
                _OVBinSummaryPart(
                    start=cur,
                    value=reference_array[cur:],
                    length=reference_array.size - cur,
                )
            )
        return cls(parts=parts)

    @classmethod
    def from_summary(cls, summary_path: Union[str, Path]):
        summary_path = Path(summary_path)
        with open(summary_path, "rb") as f:
            parts = pickle.load(f)
            return cls(parts=parts)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]):
        summary_path = Path(json_path)
        with open(summary_path, "rb") as f:
            parts = json.load(f)
            return cls(parts=[_OVBinSummaryPart.from_dict(part) for part in parts])

    def to_summary(self, save_path: Union[str, Path]):
        save_path = Path(save_path)
        with open(save_path, "wb") as f:
            pickle.dump(self.parts, f)

    def to_json(self, save_path: Union[str, Path]):
        with open(save_path, "w") as f:
            json.dump([part.to_dict() for part in self.parts], f, indent=2)

    def to_bin(self, save_path: Union[str, Path], bytes_generator: Literal["ones", "zeros", "random"] = "ones"):
        with open(save_path, "wb") as f:
            for part in self.parts:
                if part.value is not None:
                    f.write(part.value.tobytes())
                else:
                    f.write(BYTES_GENERATORS[bytes_generator](part.length))
        return save_path

    @staticmethod
    def _read_from_bin(bin_path: Union[str, Path]):
        with open(bin_path, "rb") as f:
            return np.frombuffer(f.read(), dtype=np.uint8)

    @classmethod
    def _read_ov_model(cls, xml_path: Union[str, Path]):
        if cls.ie is None:
            cls.ie = Core()
        return cls.ie.read_model(model=Path(xml_path).resolve())

    @staticmethod
    def _find_index(vector, starting_index, sub_vector):
        length = sub_vector.size
        for i in range(starting_index, vector.size - length + 1):
            if (vector[i : i + length] == sub_vector).all():
                return i
        raise ValueError("Model analysis failed.")


if __name__ == "__main__":
    import os
    import shutil
    import tempfile

    from huggingface_hub import hf_hub_download

    model_id = "yujiepan/llama-2-tiny-random"
    ir_xml_path = hf_hub_download(repo_id=model_id, filename="openvino_model.xml")
    ir_bin_path = hf_hub_download(repo_id=model_id, filename="openvino_model.bin")

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(ir_xml_path, tmpdir)
        shutil.copy(ir_bin_path, tmpdir)
        summary = OVBinSummary.from_xml(Path(tmpdir, "openvino_model.xml"))
        summary.to_json(Path(tmpdir, "random.json"))
        summary.to_summary(Path(tmpdir, "random.summary"))
        summary = OVBinSummary.from_summary(Path(tmpdir, "random.summary"))
        summary.to_bin(Path(tmpdir, "random_weight.bin"))
        os.system(f"ls -alh {tmpdir}")
