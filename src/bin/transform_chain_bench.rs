use glam as ga;
use nalgebra as na;
use std::hint::black_box;
use std::time::Instant;

type NaVector3d = na::Vector3<f64>;
type NaMatrix3d = na::Matrix3<f64>;
type NaMatrix4d = na::Matrix4<f64>;
type NaQuaterniond = na::UnitQuaternion<f64>;
type NaIsometry3d = na::Isometry3<f64>;

type GVector3d = ga::DVec3;
type GMatrix3d = ga::DMat3;
type GMatrix4d = ga::DMat4;
type GQuaterniond = ga::DQuat;
type GAffine3d = ga::DAffine3;

const NUM_TRANSFORMS: usize = 100;
const NUM_WARMUP_LOOPS: usize = 64;
const NUM_TIMED_LOOPS: usize = 4096;
const CHAINS_PER_SAMPLE: usize = 8;

#[derive(Clone, Copy)]
struct NaMat3Vec3Tfm {
    rot: NaMatrix3d,
    xyz: NaVector3d,
}

#[derive(Clone, Copy)]
struct NaQuatVec3Tfm {
    rot: NaQuaterniond,
    xyz: NaVector3d,
}

#[derive(Clone, Copy)]
struct NaSo3Vec3Tfm {
    rot_so3: NaVector3d,
    xyz: NaVector3d,
}

#[derive(Clone, Copy)]
struct GMat3Vec3Tfm {
    rot: GMatrix3d,
    xyz: GVector3d,
}

#[derive(Clone, Copy)]
struct GQuatVec3Tfm {
    rot: GQuaterniond,
    xyz: GVector3d,
}

#[derive(Clone, Copy)]
struct GSo3Vec3Tfm {
    rot_so3: GVector3d,
    xyz: GVector3d,
}

struct BenchData {
    na_isometry_chain: Vec<NaIsometry3d>,
    na_matrix4_chain: Vec<NaMatrix4d>,
    na_matrix3_vec3_chain: Vec<NaMat3Vec3Tfm>,
    na_quat_vec3_chain: Vec<NaQuatVec3Tfm>,
    na_so3_vec3_chain: Vec<NaSo3Vec3Tfm>,

    g_affine_chain: Vec<GAffine3d>,
    g_matrix4_chain: Vec<GMatrix4d>,
    g_matrix3_vec3_chain: Vec<GMat3Vec3Tfm>,
    g_quat_vec3_chain: Vec<GQuatVec3Tfm>,
    g_so3_vec3_chain: Vec<GSo3Vec3Tfm>,
}

struct Stats {
    mean_ns: f64,
    stddev_ns: f64,
    max_ns: f64,
}

impl NaMat3Vec3Tfm {
    fn identity() -> Self {
        Self {
            rot: NaMatrix3d::identity(),
            xyz: NaVector3d::zeros(),
        }
    }

    fn compose_assign(&mut self, rhs_tfm: &Self) {
        self.xyz = self.rot * rhs_tfm.xyz + self.xyz;
        self.rot *= rhs_tfm.rot;
    }
}

impl NaQuatVec3Tfm {
    fn identity() -> Self {
        Self {
            rot: NaQuaterniond::identity(),
            xyz: NaVector3d::zeros(),
        }
    }

    fn compose_assign(&mut self, rhs_tfm: &Self) {
        self.xyz = self.rot.transform_vector(&rhs_tfm.xyz) + self.xyz;
        self.rot *= rhs_tfm.rot;
    }
}

impl NaSo3Vec3Tfm {
    fn identity() -> Self {
        Self {
            rot_so3: NaVector3d::zeros(),
            xyz: NaVector3d::zeros(),
        }
    }

    fn compose_assign(&mut self, rhs_tfm: &Self) {
        let self_q_rot = NaQuaterniond::from_scaled_axis(self.rot_so3);
        let rhs_q_rot = NaQuaterniond::from_scaled_axis(rhs_tfm.rot_so3);
        let out_q_rot = self_q_rot * rhs_q_rot;
        self.xyz = self_q_rot.transform_vector(&rhs_tfm.xyz) + self.xyz;
        self.rot_so3 = out_q_rot.scaled_axis();
    }
}

impl GMat3Vec3Tfm {
    fn identity() -> Self {
        Self {
            rot: GMatrix3d::IDENTITY,
            xyz: GVector3d::ZERO,
        }
    }

    fn compose_assign(&mut self, rhs_tfm: &Self) {
        self.xyz = self.rot * rhs_tfm.xyz + self.xyz;
        self.rot *= rhs_tfm.rot;
    }
}

impl GQuatVec3Tfm {
    fn identity() -> Self {
        Self {
            rot: GQuaterniond::IDENTITY,
            xyz: GVector3d::ZERO,
        }
    }

    fn compose_assign(&mut self, rhs_tfm: &Self) {
        self.xyz = self.rot.mul_vec3(rhs_tfm.xyz) + self.xyz;
        self.rot *= rhs_tfm.rot;
    }
}

impl GSo3Vec3Tfm {
    fn identity() -> Self {
        Self {
            rot_so3: GVector3d::ZERO,
            xyz: GVector3d::ZERO,
        }
    }

    fn compose_assign(&mut self, rhs_tfm: &Self) {
        let self_q_rot = GQuaterniond::from_scaled_axis(self.rot_so3);
        let rhs_q_rot = GQuaterniond::from_scaled_axis(rhs_tfm.rot_so3);
        let out_q_rot = self_q_rot * rhs_q_rot;
        self.xyz = self_q_rot.mul_vec3(rhs_tfm.xyz) + self.xyz;
        self.rot_so3 = out_q_rot.to_scaled_axis();
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

fn build_transform_chains() -> BenchData {
    let mut na_isometry_chain = Vec::with_capacity(NUM_TRANSFORMS);
    let mut na_matrix4_chain = Vec::with_capacity(NUM_TRANSFORMS);
    let mut na_matrix3_vec3_chain = Vec::with_capacity(NUM_TRANSFORMS);
    let mut na_quat_vec3_chain = Vec::with_capacity(NUM_TRANSFORMS);
    let mut na_so3_vec3_chain = Vec::with_capacity(NUM_TRANSFORMS);

    let mut g_affine_chain = Vec::with_capacity(NUM_TRANSFORMS);
    let mut g_matrix4_chain = Vec::with_capacity(NUM_TRANSFORMS);
    let mut g_matrix3_vec3_chain = Vec::with_capacity(NUM_TRANSFORMS);
    let mut g_quat_vec3_chain = Vec::with_capacity(NUM_TRANSFORMS);
    let mut g_so3_vec3_chain = Vec::with_capacity(NUM_TRANSFORMS);

    for i in 0..NUM_TRANSFORMS {
        let x = i as f64;
        let tx = 0.05 * (0.31 * x).sin();
        let ty = 0.04 * (0.47 * x).cos();
        let tz = 0.03 * (0.19 * x).sin();
        let na_axis_xyz = NaVector3d::new(
            1.0 + 0.11 * x.sin(),
            -0.5 + 0.07 * x.cos(),
            0.9 + 0.05 * (0.17 * x).sin(),
        );
        let na_axis_unit = na::Unit::new_normalize(na_axis_xyz);
        let angle_rad = 0.03 * (0.13 * x + 0.2).sin();
        let na_xyz = NaVector3d::new(tx, ty, tz);

        let na_q_rot = NaQuaterniond::from_axis_angle(&na_axis_unit, angle_rad);
        let na_r_mat = na_q_rot.to_rotation_matrix().into_inner();
        let na_iso_tfm = NaIsometry3d::from_parts(na::Translation3::from(na_xyz), na_q_rot);
        let na_so3 = na_axis_unit.into_inner() * angle_rad;

        na_isometry_chain.push(na_iso_tfm);
        na_matrix4_chain.push(na_iso_tfm.to_homogeneous());
        na_matrix3_vec3_chain.push(NaMat3Vec3Tfm {
            rot: na_r_mat,
            xyz: na_xyz,
        });
        na_quat_vec3_chain.push(NaQuatVec3Tfm {
            rot: na_q_rot,
            xyz: na_xyz,
        });
        na_so3_vec3_chain.push(NaSo3Vec3Tfm {
            rot_so3: na_so3,
            xyz: na_xyz,
        });

        let g_axis_xyz = GVector3d::new(na_axis_xyz.x, na_axis_xyz.y, na_axis_xyz.z).normalize();
        let g_xyz = GVector3d::new(tx, ty, tz);
        let g_q_rot = GQuaterniond::from_axis_angle(g_axis_xyz, angle_rad);
        let g_r_mat = GMatrix3d::from_quat(g_q_rot);
        let g_aff_tfm = GAffine3d::from_rotation_translation(g_q_rot, g_xyz);
        let g_so3 = g_axis_xyz * angle_rad;

        g_affine_chain.push(g_aff_tfm);
        g_matrix4_chain.push(GMatrix4d::from_scale_rotation_translation(
            GVector3d::ONE,
            g_q_rot,
            g_xyz,
        ));
        g_matrix3_vec3_chain.push(GMat3Vec3Tfm {
            rot: g_r_mat,
            xyz: g_xyz,
        });
        g_quat_vec3_chain.push(GQuatVec3Tfm {
            rot: g_q_rot,
            xyz: g_xyz,
        });
        g_so3_vec3_chain.push(GSo3Vec3Tfm {
            rot_so3: g_so3,
            xyz: g_xyz,
        });
    }

    BenchData {
        na_isometry_chain,
        na_matrix4_chain,
        na_matrix3_vec3_chain,
        na_quat_vec3_chain,
        na_so3_vec3_chain,
        g_affine_chain,
        g_matrix4_chain,
        g_matrix3_vec3_chain,
        g_quat_vec3_chain,
        g_so3_vec3_chain,
    }
}

fn benchmark_na_isometry(chain: &[NaIsometry3d]) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut total_tfm = NaIsometry3d::identity();
        for tfm in chain {
            total_tfm *= tfm;
        }
        black_box(total_tfm);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        for _ in 0..CHAINS_PER_SAMPLE {
            let mut total_tfm = NaIsometry3d::identity();
            for tfm in chain {
                total_tfm *= tfm;
            }
            black_box(total_tfm);
        }
        let chain_time_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(chain_time_ns / (chain.len() * CHAINS_PER_SAMPLE) as f64);
    }

    compute_stats(&samples_ns)
}

fn benchmark_na_matrix4(chain: &[NaMatrix4d]) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut total_tfm = NaMatrix4d::identity();
        for tfm in chain {
            total_tfm *= tfm;
        }
        black_box(total_tfm);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        for _ in 0..CHAINS_PER_SAMPLE {
            let mut total_tfm = NaMatrix4d::identity();
            for tfm in chain {
                total_tfm *= tfm;
            }
            black_box(total_tfm);
        }
        let chain_time_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(chain_time_ns / (chain.len() * CHAINS_PER_SAMPLE) as f64);
    }

    compute_stats(&samples_ns)
}

fn benchmark_na_matrix3_vec3(chain: &[NaMat3Vec3Tfm]) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut total_tfm = NaMat3Vec3Tfm::identity();
        for tfm in chain {
            total_tfm.compose_assign(tfm);
        }
        black_box(total_tfm);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        for _ in 0..CHAINS_PER_SAMPLE {
            let mut total_tfm = NaMat3Vec3Tfm::identity();
            for tfm in chain {
                total_tfm.compose_assign(tfm);
            }
            black_box(total_tfm);
        }
        let chain_time_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(chain_time_ns / (chain.len() * CHAINS_PER_SAMPLE) as f64);
    }

    compute_stats(&samples_ns)
}

fn benchmark_na_quat_vec3(chain: &[NaQuatVec3Tfm]) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut total_tfm = NaQuatVec3Tfm::identity();
        for tfm in chain {
            total_tfm.compose_assign(tfm);
        }
        black_box(total_tfm);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        for _ in 0..CHAINS_PER_SAMPLE {
            let mut total_tfm = NaQuatVec3Tfm::identity();
            for tfm in chain {
                total_tfm.compose_assign(tfm);
            }
            black_box(total_tfm);
        }
        let chain_time_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(chain_time_ns / (chain.len() * CHAINS_PER_SAMPLE) as f64);
    }

    compute_stats(&samples_ns)
}

fn benchmark_na_so3_vec3(chain: &[NaSo3Vec3Tfm]) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut total_tfm = NaSo3Vec3Tfm::identity();
        for tfm in chain {
            total_tfm.compose_assign(tfm);
        }
        black_box(total_tfm);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        for _ in 0..CHAINS_PER_SAMPLE {
            let mut total_tfm = NaSo3Vec3Tfm::identity();
            for tfm in chain {
                total_tfm.compose_assign(tfm);
            }
            black_box(total_tfm);
        }
        let chain_time_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(chain_time_ns / (chain.len() * CHAINS_PER_SAMPLE) as f64);
    }

    compute_stats(&samples_ns)
}

fn benchmark_g_affine(chain: &[GAffine3d]) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut total_tfm = GAffine3d::IDENTITY;
        for tfm in chain {
            total_tfm *= *tfm;
        }
        black_box(total_tfm);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        for _ in 0..CHAINS_PER_SAMPLE {
            let mut total_tfm = GAffine3d::IDENTITY;
            for tfm in chain {
                total_tfm *= *tfm;
            }
            black_box(total_tfm);
        }
        let chain_time_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(chain_time_ns / (chain.len() * CHAINS_PER_SAMPLE) as f64);
    }

    compute_stats(&samples_ns)
}

fn benchmark_g_matrix4(chain: &[GMatrix4d]) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut total_tfm = GMatrix4d::IDENTITY;
        for tfm in chain {
            total_tfm *= *tfm;
        }
        black_box(total_tfm);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        for _ in 0..CHAINS_PER_SAMPLE {
            let mut total_tfm = GMatrix4d::IDENTITY;
            for tfm in chain {
                total_tfm *= *tfm;
            }
            black_box(total_tfm);
        }
        let chain_time_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(chain_time_ns / (chain.len() * CHAINS_PER_SAMPLE) as f64);
    }

    compute_stats(&samples_ns)
}

fn benchmark_g_matrix3_vec3(chain: &[GMat3Vec3Tfm]) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut total_tfm = GMat3Vec3Tfm::identity();
        for tfm in chain {
            total_tfm.compose_assign(tfm);
        }
        black_box(total_tfm);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        for _ in 0..CHAINS_PER_SAMPLE {
            let mut total_tfm = GMat3Vec3Tfm::identity();
            for tfm in chain {
                total_tfm.compose_assign(tfm);
            }
            black_box(total_tfm);
        }
        let chain_time_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(chain_time_ns / (chain.len() * CHAINS_PER_SAMPLE) as f64);
    }

    compute_stats(&samples_ns)
}

fn benchmark_g_quat_vec3(chain: &[GQuatVec3Tfm]) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut total_tfm = GQuatVec3Tfm::identity();
        for tfm in chain {
            total_tfm.compose_assign(tfm);
        }
        black_box(total_tfm);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        for _ in 0..CHAINS_PER_SAMPLE {
            let mut total_tfm = GQuatVec3Tfm::identity();
            for tfm in chain {
                total_tfm.compose_assign(tfm);
            }
            black_box(total_tfm);
        }
        let chain_time_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(chain_time_ns / (chain.len() * CHAINS_PER_SAMPLE) as f64);
    }

    compute_stats(&samples_ns)
}

fn benchmark_g_so3_vec3(chain: &[GSo3Vec3Tfm]) -> Stats {
    let mut samples_ns = Vec::with_capacity(NUM_TIMED_LOOPS);

    for _ in 0..NUM_WARMUP_LOOPS {
        let mut total_tfm = GSo3Vec3Tfm::identity();
        for tfm in chain {
            total_tfm.compose_assign(tfm);
        }
        black_box(total_tfm);
    }

    for _ in 0..NUM_TIMED_LOOPS {
        let start = Instant::now();
        for _ in 0..CHAINS_PER_SAMPLE {
            let mut total_tfm = GSo3Vec3Tfm::identity();
            for tfm in chain {
                total_tfm.compose_assign(tfm);
            }
            black_box(total_tfm);
        }
        let chain_time_ns = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        samples_ns.push(chain_time_ns / (chain.len() * CHAINS_PER_SAMPLE) as f64);
    }

    compute_stats(&samples_ns)
}

fn print_stats(label: &str, stats: &Stats) {
    println!(
        "{label} per multiply (ns): mean={:.3}, stddev={:.3}, max={:.3}",
        stats.mean_ns, stats.stddev_ns, stats.max_ns
    );
}

fn main() {
    let data = build_transform_chains();
    let samples_per_repr = NUM_TIMED_LOOPS * NUM_TRANSFORMS * CHAINS_PER_SAMPLE;

    let na_isometry_stats = benchmark_na_isometry(&data.na_isometry_chain);
    let na_matrix4_stats = benchmark_na_matrix4(&data.na_matrix4_chain);
    let na_matrix3_vec3_stats = benchmark_na_matrix3_vec3(&data.na_matrix3_vec3_chain);
    let na_quat_vec3_stats = benchmark_na_quat_vec3(&data.na_quat_vec3_chain);
    let na_so3_vec3_stats = benchmark_na_so3_vec3(&data.na_so3_vec3_chain);

    let g_affine_stats = benchmark_g_affine(&data.g_affine_chain);
    let g_matrix4_stats = benchmark_g_matrix4(&data.g_matrix4_chain);
    let g_matrix3_vec3_stats = benchmark_g_matrix3_vec3(&data.g_matrix3_vec3_chain);
    let g_quat_vec3_stats = benchmark_g_quat_vec3(&data.g_quat_vec3_chain);
    let g_so3_vec3_stats = benchmark_g_so3_vec3(&data.g_so3_vec3_chain);

    println!("Transform chain benchmark");
    println!(
        "config: transforms={NUM_TRANSFORMS}, warmup_loops={NUM_WARMUP_LOOPS}, timed_loops={NUM_TIMED_LOOPS}, chains_per_sample={CHAINS_PER_SAMPLE}"
    );
    println!("multiply operations per representation: {samples_per_repr}");
    print_stats("nalgebra isometry3", &na_isometry_stats);
    print_stats("nalgebra matrix4", &na_matrix4_stats);
    print_stats("nalgebra matrix3+vec3", &na_matrix3_vec3_stats);
    print_stats("nalgebra quaternion+vec3", &na_quat_vec3_stats);
    print_stats("nalgebra so3(vec3)+vec3", &na_so3_vec3_stats);
    print_stats("glam affine3", &g_affine_stats);
    print_stats("glam matrix4", &g_matrix4_stats);
    print_stats("glam matrix3+vec3", &g_matrix3_vec3_stats);
    print_stats("glam quaternion+vec3", &g_quat_vec3_stats);
    print_stats("glam so3(vec3)+vec3", &g_so3_vec3_stats);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_scalar_close(a: f64, b: f64, tol: f64) {
        let diff = (a - b).abs();
        assert!(diff <= tol, "scalar mismatch: left={a}, right={b}, diff={diff}");
    }

    fn assert_na_vec3_close(a_xyz: &NaVector3d, b_xyz: &NaVector3d, tol: f64) {
        let diff = (*a_xyz - *b_xyz).norm();
        assert!(
            diff <= tol,
            "vec3 mismatch: left={a_xyz:?}, right={b_xyz:?}, diff={diff}"
        );
    }

    fn assert_na_vec4_close(a_xyzw: &na::Vector4<f64>, b_xyzw: &na::Vector4<f64>, tol: f64) {
        let diff = (*a_xyzw - *b_xyzw).norm();
        assert!(
            diff <= tol,
            "vec4 mismatch: left={a_xyzw:?}, right={b_xyzw:?}, diff={diff}"
        );
    }

    fn g_to_na_vec3(g_xyz: GVector3d) -> NaVector3d {
        NaVector3d::new(g_xyz.x, g_xyz.y, g_xyz.z)
    }

    fn g_to_na_vec4(g_xyzw: ga::DVec4) -> na::Vector4<f64> {
        na::Vector4::new(g_xyzw.x, g_xyzw.y, g_xyzw.z, g_xyzw.w)
    }

    #[test]
    fn matrix3_vec3_transform_matches_between_nalgebra_and_glam() {
        let data = build_transform_chains();
        let b_xyz_c = NaVector3d::new(0.3, -1.1, 0.7);
        let gb_xyz_c = GVector3d::new(b_xyz_c.x, b_xyz_c.y, b_xyz_c.z);

        for i in 0..NUM_TRANSFORMS {
            let na = data.na_matrix3_vec3_chain[i].rot * b_xyz_c + data.na_matrix3_vec3_chain[i].xyz;
            let g = data.g_matrix3_vec3_chain[i].rot * gb_xyz_c + data.g_matrix3_vec3_chain[i].xyz;
            assert_na_vec3_close(&na, &g_to_na_vec3(g), 1.0e-12);
        }
    }

    #[test]
    fn matrix4_transform_matches_between_nalgebra_and_glam() {
        let data = build_transform_chains();
        let b_xyzw_c = na::Vector4::new(0.3, -1.1, 0.7, 1.0);
        let gb_xyzw_c = ga::DVec4::new(b_xyzw_c.x, b_xyzw_c.y, b_xyzw_c.z, b_xyzw_c.w);

        for i in 0..NUM_TRANSFORMS {
            let na = data.na_matrix4_chain[i] * b_xyzw_c;
            let g = data.g_matrix4_chain[i] * gb_xyzw_c;
            assert_na_vec4_close(&na, &g_to_na_vec4(g), 1.0e-12);
        }
    }

    #[test]
    fn quat_vec3_transform_matches_between_nalgebra_and_glam() {
        let data = build_transform_chains();
        let b_xyz_c = NaVector3d::new(0.3, -1.1, 0.7);
        let gb_xyz_c = GVector3d::new(b_xyz_c.x, b_xyz_c.y, b_xyz_c.z);

        for i in 0..NUM_TRANSFORMS {
            let na = data.na_quat_vec3_chain[i]
                .rot
                .transform_vector(&b_xyz_c)
                + data.na_quat_vec3_chain[i].xyz;
            let g = data.g_quat_vec3_chain[i].rot.mul_vec3(gb_xyz_c) + data.g_quat_vec3_chain[i].xyz;
            assert_na_vec3_close(&na, &g_to_na_vec3(g), 1.0e-12);
        }
    }

    #[test]
    fn so3_vec3_transform_matches_between_nalgebra_and_glam() {
        let data = build_transform_chains();
        let b_xyz_c = NaVector3d::new(0.3, -1.1, 0.7);
        let gb_xyz_c = GVector3d::new(b_xyz_c.x, b_xyz_c.y, b_xyz_c.z);

        for i in 0..NUM_TRANSFORMS {
            let na_q_rot = NaQuaterniond::from_scaled_axis(data.na_so3_vec3_chain[i].rot_so3);
            let na = na_q_rot.transform_vector(&b_xyz_c) + data.na_so3_vec3_chain[i].xyz;

            let g_q_rot = GQuaterniond::from_scaled_axis(data.g_so3_vec3_chain[i].rot_so3);
            let g = g_q_rot.mul_vec3(gb_xyz_c) + data.g_so3_vec3_chain[i].xyz;
            assert_na_vec3_close(&na, &g_to_na_vec3(g), 1.0e-12);
        }
    }

    #[test]
    fn vec3_add_matches_between_nalgebra_and_glam() {
        let a_xyz_b = NaVector3d::new(1.2, -0.4, 3.1);
        let b_xyz_c = NaVector3d::new(-4.0, 2.3, 0.7);
        let ga_xyz_b = GVector3d::new(a_xyz_b.x, a_xyz_b.y, a_xyz_b.z);
        let gb_xyz_c = GVector3d::new(b_xyz_c.x, b_xyz_c.y, b_xyz_c.z);

        let na = a_xyz_b + b_xyz_c;
        let g = ga_xyz_b + gb_xyz_c;
        assert_na_vec3_close(&na, &g_to_na_vec3(g), 1.0e-12);
        assert_scalar_close(na.norm(), g.length(), 1.0e-12);
    }
}
