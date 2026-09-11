"""Generate the solver's offline Metal sources and pipeline lookup table."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from FullStep import function_ranges, generate


def write(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def kernel_source(source, kernels):
    parts, position, found = [], 0, set()
    for match, start, end in function_ranges(source, r"(?m)^kernel void (\w+)\("):
        parts.append(source[position:match.start()])
        if match[1] in kernels:
            parts.append(source[match.start():end].replace("kernel void " + match[1] + "(", "kernel void " + kernels[match[1]] + "("))
            found.add(match[1])
        position = end
    if found != kernels.keys():
        raise ValueError(f"Missing kernels: {kernels.keys() - found}")
    parts.append(source[position:])
    return "".join(parts) + "\n"


def prune_helpers(source):
    declaration = r"(?m)^(?:template<[^\n]+>\n)?(?:__attribute__\(\([^\n]*\)\)\s*)?static [^(\n]+\b(\w+)\("
    functions = list(function_ranges(source, declaration))
    parts, position, dependencies = [], 0, {}
    for match, start, end in functions:
        parts.append(source[position:match.start()])
        dependencies.setdefault(match[1], set()).update(re.findall(r"\b\w+\b", source[match.start():end]))
        position = end
    parts.append(source[position:])
    # Names in declarations and retained kernels conservatively retain matching overloads and namespaces.
    pending = set(re.findall(r"\b\w+\b", "".join(parts))) & dependencies.keys()
    needed = set()
    while pending:
        name = pending.pop()
        needed.add(name)
        pending.update((dependencies[name] & dependencies.keys()) - needed)
    parts, position = [], 0
    for match, start, end in functions:
        parts.append(source[position:match.start()])
        if match[1] in needed:
            parts.append(source[match.start():end])
        position = end
    parts.append(source[position:])
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--safe-math", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entries = json.loads((args.source / "Pipelines.json").read_text())
    sources = {name: (args.source / (name + ".metal")).read_text() for name in ["Solve", "BroadPhase", "SolveCommands"]}
    sources["FullStep"] = generate(args.source, args.compiler)
    sources["SolveCommands"] = (args.source / "SolveCommandData.h").read_text() + sources["SolveCommands"]
    shared = (args.source / "Shared.h").read_text()
    pipelines, compile_commands, groups = [], [], {}
    indices = [[[65535] * 4 for _ in entries] for _ in range(3)]
    extras = []

    def add(name, kernel, source, defines=None, safe=False, indirect=False):
        defines = dict(defines or {})
        mode = "safe" if safe or args.safe_math else "fast"
        key = (source, tuple(sorted(defines.items())), mode)
        groups.setdefault(key, {})[kernel] = name
        pipelines.append({"compute_function": "alias:rbp#" + name, "support_indirect_command_buffers": indirect})
        return len(pipelines) - 1

    for p, entry in enumerate(entries):
        kind = entry.get("variants")
        modes = range(3) if kind == "collect" else [0]
        for mode in modes:
            variants = {"collect": [0] if mode == 2 else [2], "native": [0, 1, 3],
                        "bounds": range(2), "solve": range(2)}.get(kind, [0])
            for variant in variants:
                defines = dict(entry.get("defines", {}))
                if mode:
                    defines[["", "PREPARE_QUERIES", "QUEUED_QUERIES"][mode]] = 1
                if kind in ["collect", "native"]:
                    defines["BROAD_PHASE_MODE"] = 0 if kind == "collect" else 2 if variant else 1
                    defines["COLLECT_LANES"] = {0: 64, 1: 1, 3: 32}[variant] if kind == "native" else 64 if variant else 1
                if kind == "bounds":
                    defines["BOUNDS_LANES"] = 32 if variant else 1
                if kind == "solve":
                    defines["SOLVE_BODIES_PER_GROUP"] = 5 if variant else 1
                name = f'{entry["pass_name"]}_{mode}_{variant}'
                indirect = entry["pass_name"] in ["SolvePass", "PublishPass", "DualPass", "JointDualPass"]
                indices[mode][p][variant] = add(name, entry["kernel"], entry["source"], defines, entry.get("safe_math", False), indirect)

    for name in ["FullStep", "EncodeSolveCommands"]:
        source = "SolveCommands" if name == "EncodeSolveCommands" else name
        extras.append(f"inline constexpr uint32_t {name} = {add(name, name, source)};")

    module_names = []
    for (source, defines, mode), kernels in groups.items():
        name = next(iter(kernels.values()))
        body = sources[source]
        if source in ["Solve", "BroadPhase"]:
            body = subprocess.check_output([args.compiler, "-E", "-P", "-x", "c++",
                                            *[f"-D{k}={v}" for k, v in defines], "-"], input=body, text=True)
        write(args.output / (name + ".metal"), shared + "\n" + prune_helpers(kernel_source(body, kernels)))
        module_names.append(name)
        compile_commands.append(f'set(ShaderMath_{name} {mode})')

    names = [p["compute_function"].split("#")[1] for p in pipelines]
    header = '#pragma once\n#include <cstdint>\n\nnamespace rbp::shaders {\n'
    header += 'enum Pass : uint32_t {\n' + ''.join('    ' + e['pass_name'] + ',\n' for e in entries) + '    PassCount\n};\n'
    header += 'inline constexpr struct { const char *Name; bool Indirect; } Pipelines[]{\n'
    header += ''.join(f'    {{"{name}", {str(p["support_indirect_command_buffers"]).lower()}}},\n' for name, p in zip(names, pipelines)) + '};\n'
    header += 'inline constexpr uint16_t PipelineIndices[3][PassCount][4]{\n'
    header += ''.join('    {\n' + ''.join('        {' + ', '.join(map(str, row)) + '},\n' for row in mode) + '    },\n' for mode in indices) + '};\n'
    header += '\n'.join(extras) + '\n} // namespace rbp::shaders\n'
    write(args.output / 'Pipelines.h', header)
    write(args.output / 'Shaders.cmake', 'set(ShaderNames ' + ' '.join(module_names) + ')\n' + '\n'.join(compile_commands) + '\n')
    config = {"libraries": {"paths": [{"label": "rbp", "path": "rbp.metallib"}]}, "pipelines": {"compute_pipelines": pipelines}}
    write(args.output / 'rbp.mtlp-json', json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    main()
