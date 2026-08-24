# Third-party notices

`src/gpu/Solve.metal` is ported from the two AVBD reference implementations below, both MIT.

Its convex narrowphase — the hull path — is ported from nothing: neither of the two, nor jure/webphysics, has a hull or a general convex query, so it follows the published algorithms directly.
GJK uses the signed-volume distance subalgorithm of Montanari, Petrinic and Barbieri (ACM TOG 36(3), 2017), EPA is bounded in the manner of van den Bergen, and manifold reduction follows Gregorius, "Robust Contact Creation for Physics Simulations" (GDC 2015).
Written from the papers and the talk: openGJK, the GJK authors' reference implementation, was deliberately not read, and no code derives from it.

## avbd-demo2d — the algorithm, its parameters and its sign conventions

https://github.com/savant117/avbd-demo2d

```
MIT License

Copyright (c) 2025 Chris Giles

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MetalAVBD — the three-dimensional contact basis, friction cone and GPU pass structure

https://github.com/tatsuya-ogawa/MetalAVBD

```
MIT License

Copyright (c) 2026 Tatsuya Ogawa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
