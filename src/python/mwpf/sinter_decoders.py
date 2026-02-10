import sinter
import os
import glob
import tempfile
import math
import pathlib
from typing import Tuple, Any, Optional, Iterable, List
import mwpf
from mwpf import (  # type: ignore
    SyndromePattern,
    HyperEdge,
    SolverInitializer,
    Solver,
    BP,
    BenchmarkSuite,
    WeightRange,
)
from dataclasses import dataclass, field
import pickle
import json
import traceback
from enum import Enum
import random
import numpy as np
import stim
from io import BufferedReader, BufferedWriter
import time
import struct
from contextlib import nullcontext
# Assuming these are local modules you have; kept as is
from .ref_circuit import *
from .heralded_dem import *
import pandas as pd

# --- DECODER CLASSES ---

available_decoders = [
    "Solver",
    "SolverSerialJointSingleHair",
    "SolverSerialSingleHair",
    "SolverSerialUnionFind",
]

default_cluster_node_limit: int = 50


@dataclass
class DecoderPanic:
    initializer: SolverInitializer
    config: dict
    syndrome: SyndromePattern
    panic_message: str


class PanicAction(Enum):
    RAISE = 1
    CATCH = 2


@dataclass
class SinterMWPFDecoder:
    """
    Use MWPF to predict observables from detection events.
    """
    decoder_type: str = "SolverSerialJointSingleHair"
    cluster_node_limit: Optional[int] = None
    c: Optional[int] = None
    timeout: Optional[float] = None
    with_progress: bool = False
    circuit: Optional[stim.Circuit] = None
    pass_circuit: bool = False
    panic_action: PanicAction = PanicAction.CATCH
    panic_cases: list[DecoderPanic] = field(default_factory=list)
    benchmark_suite_filename: Optional[str] = None
    trace_filename: Optional[str] = None
    bp: bool = False
    max_iter: int = 0
    bp_method: str = "ms"
    ms_scaling_factor: float = 0.625
    schedule: str = "parallel"
    omp_thread_count: int = 1
    random_schedule_seed: int = 0
    serial_schedule_order: Optional[list[int]] = None
    bp_weight_mix_ratio: float = 1.0
    floor_weight: Optional[float] = None
    bp_converge: bool = True

    @staticmethod
    def parse_mwpf_trace(base_filename: str) -> pd.DataFrame:
        """Parses and merges all binary trace files (base + worker extensions)."""
        results = []
        record_size = struct.calcsize("ffff")
        # Use glob to find all worker files (e.g., trace.bin.1234, trace.bin.5678)
        all_files = glob.glob(f"{base_filename}*")
        
        for filename in all_files:
            try:
                # Check if file is empty to avoid errors
                if os.path.getsize(filename) == 0:
                    continue
                    
                with open(filename, "rb") as f:
                    while chunk := f.read(record_size):
                        if len(chunk) != record_size:
                            break
                        elapsed, lower, upper, _ = struct.unpack("ffff", chunk)
                        results.append({
                            "cpu_time": elapsed,
                            "objective_value": upper,
                            "lower_bound": lower
                        })
            except Exception as e:
                print(f"Warning: Could not read trace file {filename}: {e}")

        return pd.DataFrame(results, columns=["cpu_time", "objective_value", "lower_bound"])

    @property
    def _cluster_node_limit(self) -> int:
        if self.cluster_node_limit is not None:
            assert self.c is None, "Cannot set both `cluster_node_limit` and `c`."
            return self.cluster_node_limit
        elif self.c is not None:
            assert self.cluster_node_limit is None, "Cannot set both `cluster_node_limit` and `c`."
            return self.c
        return default_cluster_node_limit

    @property
    def config(self) -> dict[str, Any]:
        return dict(cluster_node_limit=self._cluster_node_limit)

    def with_circuit(self, circuit: stim.Circuit | None) -> "SinterMWPFDecoder":
        if circuit is None:
            self.circuit = None
            return self
        assert isinstance(circuit, stim.Circuit)
        self.circuit = circuit.copy()
        return self

    def common_prepare(self, dem: "stim.DetectorErrorModel") -> tuple[Any, Predictor, Any]:
        if self.pass_circuit:
            assert self.circuit is not None, "The circuit is not loaded but the flag `pass_circuit` is True"

        solver, predictor = construct_decoder_and_predictor(
            dem,
            decoder_type=self.decoder_type,
            config=self.config,
            ref_circuit=(RefCircuit.of(self.circuit) if self.circuit is not None else None),
        )
        assert dem.num_detectors == predictor.num_detectors()
        assert dem.num_observables == predictor.num_observables()

        bp_decoder: Optional[Any] = None
        if self.bp:
            if self.circuit is not None:
                assert not predictor.is_dynamic, "BP is not supported for dynamic predictors."
            from ldpc import BpDecoder
            from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices

            bp_matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
            bp_decoder = BpDecoder(
                pcm=bp_matrices.check_matrix,
                error_channel=list(bp_matrices.priors),
                max_iter=self.max_iter,
                bp_method=self.bp_method,
                ms_scaling_factor=self.ms_scaling_factor,
                schedule=self.schedule,
                omp_thread_count=self.omp_thread_count,
                serial_schedule_order=self.serial_schedule_order,
                input_vector_type="syndrome",
            )

        return solver, predictor, bp_decoder

    def compile_decoder_for_dem(self, *, dem: "stim.DetectorErrorModel") -> "MwpfCompiledDecoder":
        solver, predictor, bp_decoder = self.common_prepare(dem)

        benchmark_suite: Optional[BenchmarkSuite] = None
        if self.benchmark_suite_filename is not None:
            benchmark_suite = BenchmarkSuite(solver.get_initializer())
            solver = None

        # Pass the base filename directly. 
        # The PID will be appended inside the worker process later.
        worker_trace_filename = self.trace_filename

        return MwpfCompiledDecoder(
            solver,
            predictor,
            dem.num_detectors,
            dem.num_observables,
            panic_action=self.panic_action,
            panic_cases=self.panic_cases,
            benchmark_suite=benchmark_suite,
            benchmark_suite_filename=self.benchmark_suite_filename,
            trace_filename=worker_trace_filename,
            bp_decoder=bp_decoder,
            bp_weight_mix_ratio=self.bp_weight_mix_ratio,
            floor_weight=self.floor_weight,
            bp_converge=self.bp_converge,
        )

    def decode_via_files(self, *, num_shots: int, num_dets: int, num_obs: int, dem_path: pathlib.Path,
                         dets_b8_in_path: pathlib.Path, obs_predictions_b8_out_path: pathlib.Path,
                         tmp_dir: pathlib.Path) -> None:
        dem = stim.DetectorErrorModel.from_file(dem_path)
        solver, predictor, bp_decoder = self.common_prepare(dem)
        
        benchmark_suite: Optional[BenchmarkSuite] = None
        if self.benchmark_suite_filename is not None:
            benchmark_suite = BenchmarkSuite(solver.get_initializer())
            solver = None

        num_det_bytes = math.ceil(num_dets / 8)
        
        # When decoding via files directly (usually single process/main thread),
        # we can use the filename as-is or append PID if desired.
        # Since this isn't the Sinter worker loop, we use it directly or with a main PID.
        actual_trace_filename = self.trace_filename
        if actual_trace_filename is not None and not os.path.exists(actual_trace_filename):
             # Ensure we don't conflict if called multiple times in parallel manually
             actual_trace_filename = f"{self.trace_filename}.{os.getpid()}"
        
        mode = "ab" if actual_trace_filename is not None else "wb"
        
        with (open(actual_trace_filename, mode) if actual_trace_filename is not None else nullcontext()) as trace_f:
            with open(dets_b8_in_path, "rb") as dets_in_f:
                with open(obs_predictions_b8_out_path, "wb") as obs_out_f:
                    for dets_bit_packed in iter_det(dets_in_f, num_shots, num_det_bytes, self.with_progress):
                        prediction = decode_common(
                            dets_bit_packed=dets_bit_packed,
                            predictor=predictor,
                            solver=solver,
                            num_dets=num_dets,
                            num_obs=num_obs,
                            panic_action=self.panic_action,
                            panic_cases=self.panic_cases,
                            benchmark_suite=benchmark_suite,
                            trace_f=trace_f,
                            bp_decoder=bp_decoder,
                            bp_weight_mix_ratio=self.bp_weight_mix_ratio,
                            floor_weight=self.floor_weight,
                            bp_converge=self.bp_converge,
                        )
                        obs_out_f.write(int(prediction).to_bytes((num_obs + 7) // 8, byteorder="little"))

        if benchmark_suite is not None:
            benchmark_suite.save_cbor(self.benchmark_suite_filename)


def iter_det(f: BufferedReader, num_shots: int, num_det_bytes: int, with_progress: bool = False) -> Iterable[np.ndarray]:
    if with_progress:
        from tqdm import tqdm
        pbar = tqdm(total=num_shots, desc="shots")
    for _ in range(num_shots):
        if with_progress:
            pbar.update(1)
        dets_bit_packed = np.fromfile(f, dtype=np.uint8, count=num_det_bytes)
        if dets_bit_packed.shape != (num_det_bytes,):
            raise IOError("Missing dets data.")
        yield dets_bit_packed


def construct_decoder_and_predictor(model: "stim.DetectorErrorModel", decoder_type: Any, config: dict[str, Any],
                                    ref_circuit: Optional[RefCircuit] = None) -> Tuple[Any, Predictor]:
    if ref_circuit is not None:
        heralded_dem = HeraldedDetectorErrorModel(ref_circuit=ref_circuit)
        initializer = heralded_dem.initializer
        predictor: Predictor = heralded_dem.predictor
    else:
        ref_dem = RefDetectorErrorModel.of(dem=model)
        initializer = ref_dem.initializer
        predictor = ref_dem.predictor

    if decoder_type is None:
        decoder_cls = Solver
    elif isinstance(decoder_type, str):
        decoder_cls = getattr(mwpf, decoder_type)
    else:
        decoder_cls = decoder_cls
    return (decoder_cls(initializer, config=config), predictor)


def panic_text_of(solver, syndrome) -> str:
    initializer = solver.get_initializer()
    config = solver.config
    panic_text = f"""
######## MWPF Sinter Decoder Panic ######## 
solver_initializer: dict = json.loads('{initializer.to_json()}')
config: dict = json.loads('{json.dumps(config)}')
syndrome: dict = json.loads('{syndrome.to_json()}')
######## PICKLE DATA ######## 
solver_initializer: SolverInitializer = pickle.loads({pickle.dumps(initializer)!r})
config: dict = pickle.loads({pickle.dumps(config)!r})
syndrome: SyndromePattern = pickle.loads({pickle.dumps(syndrome)!r})
######## End Panic Information ######## 
"""
    return panic_text

@dataclass
class SinterHUFDecoder(SinterMWPFDecoder):
    decoder_type: str = "SolverSerialUnionFind"
    cluster_node_limit: int = 0


@dataclass
class SinterSingleHairDecoder(SinterMWPFDecoder):
    decoder_type: str = "SolverSerialSingleHair"
    cluster_node_limit: int = 0


@dataclass
class MwpfCompiledDecoder:
    solver: Any
    predictor: Predictor
    num_dets: int
    num_obs: int
    panic_action: PanicAction
    panic_cases: list[DecoderPanic]
    benchmark_suite: Optional[BenchmarkSuite]
    trace_filename: Optional[str]
    benchmark_suite_filename: Optional[str]
    bp_decoder: Any
    bp_weight_mix_ratio: float
    floor_weight: Optional[float]
    bp_converge: bool = True

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: "np.ndarray") -> "np.ndarray":
        num_shots = bit_packed_detection_event_data.shape[0]
        predictions = np.zeros(shape=(num_shots, (self.num_obs + 7) // 8), dtype=np.uint8)
        
        # Calculate the actual filename using the current process ID (Worker PID)
        # This prevents file locking/corruption when multiple Sinter workers run
        actual_trace_filename = None
        if self.trace_filename is not None:
            actual_trace_filename = f"{self.trace_filename}.{os.getpid()}"

        # Open in 'ab' (append binary) mode to preserve data from previous batches
        ctx = open(actual_trace_filename, "ab") if actual_trace_filename else nullcontext()
        
        with ctx as trace_f:
            for shot in range(num_shots):
                dets_bit_packed = bit_packed_detection_event_data[shot]
                prediction = decode_common(
                    dets_bit_packed=dets_bit_packed,
                    predictor=self.predictor,
                    solver=self.solver,
                    num_dets=self.num_dets,
                    num_obs=self.num_obs,
                    panic_action=self.panic_action,
                    panic_cases=self.panic_cases,
                    benchmark_suite=self.benchmark_suite,
                    trace_f=trace_f,
                    bp_decoder=self.bp_decoder,
                    bp_weight_mix_ratio=self.bp_weight_mix_ratio,
                    floor_weight=self.floor_weight,
                    bp_converge=self.bp_converge,
                )
                predictions[shot] = np.packbits(
                    np.array(list(np.binary_repr(prediction, width=self.num_obs))[::-1], dtype=np.uint8),
                    bitorder="little",
                )

        if self.benchmark_suite is not None:
            self.benchmark_suite.save_cbor(self.benchmark_suite_filename)

        return predictions


def decode_common(dets_bit_packed: np.ndarray, predictor: Predictor, solver: Any, num_dets: int, num_obs: int,
                  panic_action: PanicAction, panic_cases: list[DecoderPanic], benchmark_suite: Optional[BenchmarkSuite],
                  trace_f: Optional[BufferedWriter], bp_decoder: Any, bp_weight_mix_ratio: float,
                  floor_weight: Optional[float], bp_converge: bool = True) -> int:
    syndrome = predictor.syndrome_of(dets_bit_packed)
    if solver is None:
        if benchmark_suite is not None:
            benchmark_suite.append(syndrome)
        prediction = 0
    else:
        try:
            start = time.perf_counter()
            if bp_decoder is not None:
                dets_bits = np.unpackbits(dets_bit_packed, count=num_dets, bitorder="little")
                bp_solution = bp_decoder.decode(dets_bits)
                if bp_decoder.converge and bp_converge:
                    prediction = predictor.prediction_of(syndrome, np.flatnonzero(bp_solution))
                else:
                    syndrome = SyndromePattern(
                        defect_vertices=syndrome.defect_vertices,
                        override_weights=list(bp_decoder.log_prob_ratios),
                        override_ratio=bp_weight_mix_ratio,
                        floor_weight=floor_weight,
                    )
                    solver.solve(syndrome)
                    if trace_f is None:
                        subgraph = solver.subgraph()
                    else:
                        subgraph, bound = solver.subgraph_range()
                        record_trace(trace_f, time.perf_counter() - start, bound)
                    prediction = predictor.prediction_of(syndrome, subgraph)
            else:
                solver.solve(syndrome)
                if trace_f is None:
                    subgraph = solver.subgraph()
                else:
                    subgraph, bound = solver.subgraph_range()
                    record_trace(trace_f, time.perf_counter() - start, bound)
                prediction = predictor.prediction_of(syndrome, subgraph)
        except BaseException as e:
            panic_cases.append(DecoderPanic(
                initializer=solver.get_initializer(),
                config=solver.config,
                syndrome=syndrome,
                panic_message=traceback.format_exc(),
            ))
            if "<class 'KeyboardInterrupt'>" in str(e):
                raise e
            elif panic_action == PanicAction.RAISE:
                raise e
            elif panic_action == PanicAction.CATCH:
                prediction = random.getrandbits(num_obs)
    return prediction


def record_trace(trace_f: BufferedWriter, elapsed: float, bound: WeightRange):
    trace_f.write(struct.pack("f", elapsed))
    trace_f.write(struct.pack("f", bound.lower.float()))
    trace_f.write(struct.pack("f", bound.upper.float()))
    trace_f.write(struct.pack("f", 0))
