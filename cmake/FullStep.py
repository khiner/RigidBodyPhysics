"""Generate complete-step shader source from shared solver routines."""

import argparse
from pathlib import Path
import re
import subprocess


ENTRIES = {
    "BuildBodyBounds", "CollectContacts", "PrepareSmallWorld", "Finalize",
    "Restitution", "ApplyRestitution", "FinishPoses", "FinishWaking", "CaptureStep",
}


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one occurrence: {before}")
    return source.replace(before, after)


def helpers(source, keep):
    result, position, found = [], 0, set()
    for match in re.finditer(r"(?m)^[ \t]*kernel void (\w+)\(", source):
        if match.start() < position:
            continue
        start = source.index("{", match.end())
        depth = 0
        for token in re.finditer(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[{}]''', source[start:]):
            depth += (token[0] == "{") - (token[0] == "}")
            if depth == 0:
                end = start + token.end()
                break
        else:
            raise ValueError(f"Unterminated kernel: {match[1]}")
        result.append(source[position:match.start()])
        if match[1] in keep:
            found.add(match[1])
            signature = re.sub(r"\s*\[\[[^]]+\]\]", "", source[match.start():start])
            signature = signature.replace("kernel void", "static void")
            body = source[start:end]
            if match[1] == "CollectContacts":
                for declaration in ["threadgroup GeometryManifold geometries[32];", "threadgroup ContactHistory history;", "threadgroup bool measure_cached;"]:
                    body = replace_once(body, declaration, "")
                signature = signature[:signature.rindex(")")] + ", threadgroup GeometryManifold *geometries, threadgroup ContactHistory &history, threadgroup bool &measure_cached) "
                # Store one query result per SIMD group.
                query = "QueryGeometry(query, candidate, own_triangle, previous, hull_vertices, hull_faces, mesh_triangles, cooperate ? lane % 32 : NoIndex);"
                body = replace_once(body, "geometries[lane] = " + query, "const GeometryManifold geometry = " + query + "\nif (lane == 0) geometries[0] = geometry;")
                body = replace_once(body, "geometries[query_at]", "geometries[0]")
            if match[1] == "PrepareSmallWorld":
                # Full-world execution requires no island schedule.
                body = replace_once(body, "    if (budgets[0]) FindSmallIslandsBody(masses, contacts, p, joints, incoming, slots, joint_incidence, quiet, islands, body);", "")
            result.append(signature + body)
        position = end
    if found != keep:
        raise ValueError(f"Missing complete-step helpers: {keep - found}")
    result.append(source[position:])
    source = "".join(result)
    source = source.replace("constant StepParams &", "thread const StepParams &")
    source = source.replace("constant StepOutputFlags &", "thread const StepOutputFlags &")
    return source.replace("threadgroup_barrier(", "simdgroup_barrier(")


def generate(source, compiler):
    def preprocess(name, **defines):
        return subprocess.check_output(
            [compiler, "-E", "-P", "-x", "c++", *[f"-D{k}={v}" for k, v in defines.items()], str(source / name)], text=True)

    shared, variants = [], []
    for name, sensor, stabilize in [("solid", 0, 0), ("sensor", 1, 0), ("stabilize", 0, 1)]:
        text = helpers(preprocess("Solve.metal", COLLECT_LANES=32, BOUNDS_LANES=1,
                                 BROAD_PHASE_MODE=1, SENSOR_PASS=sensor, STABILIZE=stabilize,
                                 FUSED_SMALL_SOLVE=0, MESH_SHAPES=0, MESH_PAIRS=0,
                                 BOUNDED_PLANES=0, INTEGRATE_BOUNDS=int(not sensor)), ENTRIES)
        for index, type_name in enumerate(["GeometryManifold", "ContactHistory"]):
            match = re.search(r"struct " + type_name + r" \{.*?\n\};", text, re.S)
            if not match:
                raise ValueError(f"Missing shared scratch type: {type_name}")
            if name == "solid":
                shared.append(match[0])
            elif match[0] != shared[index]:
                raise ValueError(f"Scratch layout differs in {name}: {type_name}")
            text = replace_once(text, match[0], "")
        variants.append("namespace " + name + " {\n" + text + "\n}\n")
    variants.append("namespace broad {\n" + helpers(preprocess("BroadPhase.metal"), set()) + "\n}\n")
    return (source / "FullStepData.h").read_text() + "\n" + "\n".join(shared) + "\n" + "".join(variants) + (source / "FullStep.metal").read_text()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = generate(args.source, args.compiler)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != text:
        args.output.write_text(text)
