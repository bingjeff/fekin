use glam as ga;
use nalgebra as na;
use std::hint::black_box;
use std::time::Instant;

type Vector3d = na::Vector3<f64>;
type Matrix3d = na::Matrix3<f64>;
type UnitQuaterniond = na::UnitQuaternion<f64>;
type GVector3d = ga::DVec3;
type GMatrix3d = ga::DMat3;
type GQuaterniond = ga::DQuat;

const NUM_ROTATIONS: usize = 1024;
const NUM_WARMUP_LOOPS: usize = 64;
const NUM_TIMED_LOOPS: usize = 4096;
const BATCHES_PER_SAMPLE: usize = 8;

#[derive(Clone, Copy)]
struct PackedQuatd {
    w: f64,
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy)]
struct AxisAngled {
    axis: Vector3d,
    angle: f64,
    sin_angle: f64,
    cos_angle: f64,
}

struct BenchData {
    vectors: Vec<Vector3d>,
    quat_fast: Vec<PackedQuatd>,
    quat_na: Vec<UnitQuaterniond>,
    matrix3: Vec<Matrix3d>,
    vectors_glam: Vec<GVector3d>,
    quat_glam: Vec<GQuaterniond>,
    matrix3_glam: Vec<GMatrix3d>,
    axis_angle: Vec<AxisAngled>,
}

struct Stats {
    mean_ns: f64,
    stddev_ns: f64,
    max_ns: f64,
}

impl PackedQuatd {
    #[inline(always)]
    fn from_unit_quat(q_rot: &UnitQuaterniond) -> Self {
        let q_wxyz = q_rot.as_ref();
        Self {
            w: q_wxyz.w,
            x: q_wxyz.i,
            y: q_wxyz.j,
            z: q_wxyz.k,
        }
    }

    // Fast unit-quaternion vector rotation:
    // t = 2 * (q_xyz x v)
    // v' = v + w * t + (q_xyz x t)
    #[inline(always)]
    fn rotate_vec3(&self, v: &Vector3d) -> Vector3d {
        let tx = 2.0 * (self.y * v.z - self.z * v.y);
        let ty = 2.0 * (self.z * v.x - self.x * v.z);
        let tz = 2.0 * (self.x * v.y - self.y * v.x);

        let cx = self.y * tz - self.z * ty;
        let cy = self.z * tx - self.x * tz;
        let cz = self.x * ty - self.y * tx;

        Vector3d::new(
            v.x + self.w.mul_add(tx, cx),
            v.y + self.w.mul_add(ty, cy),
            v.z + self.w.mul_add(tz, cz),
        )
    }
}

impl AxisAngled {
    #[inline(always)]
    fn rotate_vec3_runtime_trig(&self, v: &Vector3d) -> Vector3d {
        let (sin_angle, cos_angle) = self.angle.sin_cos();
        rotate_axis_angle_with_sin_cos(&self.axis, sin_angle, cos_angle, v)
    }

    #[inline(always)]
    fn rotate_vec3_cached_trig(&self, v: &Vector3d) -> Vector3d {
        rotate_axis_angle_with_sin_cos(&self.axis, self.sin_angle, self.cos_angle, v)
    }
}

#[inline(always)]
fn rotate_axis_angle_with_sin_cos(
    axis: &Vector3d,
    sin_angle: f64,
    cos_angle: f64,
    v: &Vector3d,
) -> Vector3d {
    let one_minus_cos = 1.0 - cos_angle;
    let axis_dot_v = axis.dot(v);
    let axis_cross_v = axis.cross(v);

    v * cos_angle + axis_cross_v * sin_angle + axis * (axis_dot_v * one_minus_cos)
}

fn build_data() -> BenchData {
    let mut vectors = Vec::with_capacity(NUM_ROTATIONS);
    let mut quat_fast = Vec::with_capacity(NUM_ROTATIONS);
    let mut quat_na = Vec::with_capacity(NUM_ROTATIONS);
    let mut matrix3 = Vec::with_capacity(NUM_ROTATIONS);
    let mut vectors_glam = Vec::with_capacity(NUM_ROTATIONS);
    let mut quat_glam = Vec::with_capacity(NUM_ROTATIONS);
    let mut matrix3_glam = Vec::with_capacity(NUM_ROTATIONS);
    let mut axis_angle = Vec::with_capacity(NUM_ROTATIONS);

    for i in 0..NUM_ROTATIONS {
        let x = i as f64;
        let axis_raw = Vector3d::new(
            0.8 + 0.3 * (0.13 * x).sin(),
            -0.4 + 0.2 * (0.29 * x).cos(),
            0.5 + 0.4 * (0.17 * x).sin(),
        );
        let unit_axis = na::Unit::new_normalize(axis_raw);
        let angle = 0.9 * (0.011 * x + 0.4).sin();
        let (sin_angle, cos_angle) = angle.sin_cos();

        let q_rot = UnitQuaterniond::from_axis_angle(&unit_axis, angle);
        let m_rot = q_rot.to_rotation_matrix().into_inner();
        let v = Vector3d::new(
            0.7 * (0.07 * x).sin(),
            -0.2 + 0.8 * (0.03 * x + 0.2).cos(),
            0.4 * (0.05 * x + 0.4).sin(),
        );

        vectors.push(v);
        quat_fast.push(PackedQuatd::from_unit_quat(&q_rot));
        quat_na.push(q_rot);
        matrix3.push(m_rot);
        vectors_glam.push(GVector3d::new(v.x, v.y, v.z));
        quat_glam.push(GQuaterniond::from_xyzw(q_rot.i, q_rot.j, q_rot.k, q_rot.w));
        matrix3_glam.push(GMatrix3d::from_cols(
            GVector3d::new(m_rot[(0, 0)], m_rot[(1, 0)], m_rot[(2, 0)]),
            GVector3d::new(m_rot[(0, 1)], m_rot[(1, 1)], m_rot[(2, 1)]),
            GVector3d::new(m_rot[(0, 2)], m_rot[(1, 2)], m_rot[(2, 2)]),
        ));
        axis_angle.push(AxisAngled {
            axis: unit_axis.into_inner(),
            angle,
            sin_angle,
            cos_angle,
        });
    }

    BenchData {
        vectors,
        quat_fast,
        quat_na,
        matrix3,
        vectors_glam,
        quat_glam,
        matrix3_glam,
        axis_angle,
    }
}

fn compute_stats(samples_ns: &[f64]) -> Stats {
    let n = samples_ns.len() as f64;
    let sum_ns: f64 = samples_ns.iter().sum();
    let mean_ns = sum_ns / n;
    let variance_ns = samples_ns
        .iter()
        .map(|sample_ns| {
            let delta = sample_ns - mean_ns;
            delta * delta
        })
        .sum::<f64>()
        / n;
    let max_ns = samples_ns.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    Stats {
        mean_ns,
        stddev_ns: variance_ns.sqrt(),
        max_ns,
    }
}

fn bench_quat_fast(data: &BenchData) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut accum = Vector3d::zeros();
        for i in 0..NUM_ROTATIONS {
            accum += data.quat_fast[i].rotate_vec3(&data.vectors[i]);
        }
        black_box(accum);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        let mut accum = Vector3d::zeros();
        for _ in 0..BATCHES_PER_SAMPLE {
            for i in 0..NUM_ROTATIONS {
                accum += data.quat_fast[i].rotate_vec3(&data.vectors[i]);
            }
        }
        let elapsed_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(elapsed_ns / (NUM_ROTATIONS * BATCHES_PER_SAMPLE) as f64);
        black_box(accum);
    }

    compute_stats(&samples_ns)
}

fn bench_quat_nalgebra(data: &BenchData) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut accum = Vector3d::zeros();
        for i in 0..NUM_ROTATIONS {
            accum += data.quat_na[i].transform_vector(&data.vectors[i]);
        }
        black_box(accum);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        let mut accum = Vector3d::zeros();
        for _ in 0..BATCHES_PER_SAMPLE {
            for i in 0..NUM_ROTATIONS {
                accum += data.quat_na[i].transform_vector(&data.vectors[i]);
            }
        }
        let elapsed_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(elapsed_ns / (NUM_ROTATIONS * BATCHES_PER_SAMPLE) as f64);
        black_box(accum);
    }

    compute_stats(&samples_ns)
}

fn bench_quat_glam(data: &BenchData) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut accum = GVector3d::ZERO;
        for i in 0..NUM_ROTATIONS {
            accum += data.quat_glam[i].mul_vec3(data.vectors_glam[i]);
        }
        black_box(accum);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        let mut accum = GVector3d::ZERO;
        for _ in 0..BATCHES_PER_SAMPLE {
            for i in 0..NUM_ROTATIONS {
                accum += data.quat_glam[i].mul_vec3(data.vectors_glam[i]);
            }
        }
        let elapsed_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(elapsed_ns / (NUM_ROTATIONS * BATCHES_PER_SAMPLE) as f64);
        black_box(accum);
    }

    compute_stats(&samples_ns)
}

fn bench_matrix3(data: &BenchData) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut accum = Vector3d::zeros();
        for i in 0..NUM_ROTATIONS {
            accum += data.matrix3[i] * data.vectors[i];
        }
        black_box(accum);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        let mut accum = Vector3d::zeros();
        for _ in 0..BATCHES_PER_SAMPLE {
            for i in 0..NUM_ROTATIONS {
                accum += data.matrix3[i] * data.vectors[i];
            }
        }
        let elapsed_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(elapsed_ns / (NUM_ROTATIONS * BATCHES_PER_SAMPLE) as f64);
        black_box(accum);
    }

    compute_stats(&samples_ns)
}

fn bench_matrix3_glam(data: &BenchData) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut accum = GVector3d::ZERO;
        for i in 0..NUM_ROTATIONS {
            accum += data.matrix3_glam[i] * data.vectors_glam[i];
        }
        black_box(accum);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        let mut accum = GVector3d::ZERO;
        for _ in 0..BATCHES_PER_SAMPLE {
            for i in 0..NUM_ROTATIONS {
                accum += data.matrix3_glam[i] * data.vectors_glam[i];
            }
        }
        let elapsed_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(elapsed_ns / (NUM_ROTATIONS * BATCHES_PER_SAMPLE) as f64);
        black_box(accum);
    }

    compute_stats(&samples_ns)
}

fn bench_axis_angle_runtime(data: &BenchData) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut accum = Vector3d::zeros();
        for i in 0..NUM_ROTATIONS {
            accum += data.axis_angle[i].rotate_vec3_runtime_trig(&data.vectors[i]);
        }
        black_box(accum);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        let mut accum = Vector3d::zeros();
        for _ in 0..BATCHES_PER_SAMPLE {
            for i in 0..NUM_ROTATIONS {
                accum += data.axis_angle[i].rotate_vec3_runtime_trig(&data.vectors[i]);
            }
        }
        let elapsed_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(elapsed_ns / (NUM_ROTATIONS * BATCHES_PER_SAMPLE) as f64);
        black_box(accum);
    }

    compute_stats(&samples_ns)
}

fn bench_axis_angle_cached(data: &BenchData) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut accum = Vector3d::zeros();
        for i in 0..NUM_ROTATIONS {
            accum += data.axis_angle[i].rotate_vec3_cached_trig(&data.vectors[i]);
        }
        black_box(accum);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        let mut accum = Vector3d::zeros();
        for _ in 0..BATCHES_PER_SAMPLE {
            for i in 0..NUM_ROTATIONS {
                accum += data.axis_angle[i].rotate_vec3_cached_trig(&data.vectors[i]);
            }
        }
        let elapsed_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(elapsed_ns / (NUM_ROTATIONS * BATCHES_PER_SAMPLE) as f64);
        black_box(accum);
    }

    compute_stats(&samples_ns)
}

fn print_stats(label: &str, stats: &Stats) {
    println!(
        "{label} vec3 rotate (ns): mean={:.3}, stddev={:.3}, max={:.3}",
        stats.mean_ns, stats.stddev_ns, stats.max_ns
    );
}

fn main() {
    let data = build_data();
    let total_rotations = NUM_TIMED_LOOPS * BATCHES_PER_SAMPLE * NUM_ROTATIONS;

    println!("Vec3 rotation benchmark");
    println!(
        "config: rotations={NUM_ROTATIONS}, warmup_loops={NUM_WARMUP_LOOPS}, timed_loops={NUM_TIMED_LOOPS}, batches_per_sample={BATCHES_PER_SAMPLE}"
    );
    println!("rotation ops per method: {total_rotations}");

    let quat_fast_stats = bench_quat_fast(&data);
    let quat_na_stats = bench_quat_nalgebra(&data);
    let quat_glam_stats = bench_quat_glam(&data);
    let matrix3_stats = bench_matrix3(&data);
    let matrix3_glam_stats = bench_matrix3_glam(&data);
    let axis_angle_runtime_stats = bench_axis_angle_runtime(&data);
    let axis_angle_cached_stats = bench_axis_angle_cached(&data);

    print_stats("quaternion fast", &quat_fast_stats);
    print_stats("quaternion nalgebra", &quat_na_stats);
    print_stats("quaternion glam", &quat_glam_stats);
    print_stats("matrix3 nalgebra", &matrix3_stats);
    print_stats("matrix3 glam", &matrix3_glam_stats);
    print_stats("axis-angle runtime trig", &axis_angle_runtime_stats);
    print_stats("axis-angle cached trig", &axis_angle_cached_stats);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gvec_to_na(v: GVector3d) -> Vector3d {
        Vector3d::new(v.x, v.y, v.z)
    }

    fn assert_vec_close(a: &Vector3d, b: &Vector3d, tol: f64) {
        let diff = (a - b).norm();
        assert!(diff <= tol, "vector mismatch: diff={diff}, a={a:?}, b={b:?}");
    }

    #[test]
    fn fast_quaternion_matches_nalgebra_quaternion() {
        let data = build_data();
        for i in 0..NUM_ROTATIONS {
            let fast = data.quat_fast[i].rotate_vec3(&data.vectors[i]);
            let na = data.quat_na[i].transform_vector(&data.vectors[i]);
            let glam = gvec_to_na(data.quat_glam[i].mul_vec3(data.vectors_glam[i]));
            assert_vec_close(&fast, &na, 1.0e-12);
            assert_vec_close(&glam, &na, 1.0e-12);
        }
    }

    #[test]
    fn matrix_and_axis_angle_match_quaternion_rotation() {
        let data = build_data();
        for i in 0..NUM_ROTATIONS {
            let q = data.quat_na[i].transform_vector(&data.vectors[i]);
            let m = data.matrix3[i] * data.vectors[i];
            let m_glam = gvec_to_na(data.matrix3_glam[i] * data.vectors_glam[i]);
            let aa_runtime = data.axis_angle[i].rotate_vec3_runtime_trig(&data.vectors[i]);
            let aa_cached = data.axis_angle[i].rotate_vec3_cached_trig(&data.vectors[i]);

            assert_vec_close(&m, &q, 1.0e-12);
            assert_vec_close(&m_glam, &q, 1.0e-12);
            assert_vec_close(&aa_runtime, &q, 1.0e-12);
            assert_vec_close(&aa_cached, &q, 1.0e-12);
        }
    }
}
