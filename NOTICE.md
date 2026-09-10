# Third-party notices

`src/gpu/Solve.metal` is ported from the two AVBD reference implementations below, both MIT.

The convex narrowphase follows these published algorithms:

- GJK distance follows Montanari, Petrinic and Barbieri's signed-volume method (ACM TOG 36(3), 2017).
- Bounded EPA follows van den Bergen.
- Manifold reduction follows Gregorius, "Robust Contact Creation for Physics Simulations" (GDC 2015).

The cylinder support mapping and penetration initialization reference [Jolt's cylinder][jolt-cylinder] and EPA implementations.
The cylinder query fixture uses the same revision with authored radii and zero additional convex radius.

[jolt-cylinder]: https://github.com/jrouwe/JoltPhysics/blob/187da15da976652567738cc7603ade72bd0ae702/Jolt/Physics/Collision/Shape/CylinderShape.cpp

## avbd-demo2d

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

## MetalAVBD

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
