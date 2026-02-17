# FE-Kin

A kinematic library intended to provide kinematics (position, velocity, and acceleration) with
a reasonably high level of performance for "tree-like" systems. There are always trade-offs
for different implementations, this particular experiment is aimed at being able to compute
all velocities and accelerations. It expresses the rigid-body transforms using "twist" math,
that is a rewriting of Lie algebra solutions into 3D vector and matrix operations.

# Learnings thus far

Somewhat surprising is that 4x4 matrices actually seem to operate faster than scalar and
vector operations. This is likely due to what is easily optimized through naive implementation
of various operations. I think most surprising is that the SIMD operations don't naturally
kick-in for quaternion and vector formed operations. See the "benchmarks" branch to see
some of these tests.
