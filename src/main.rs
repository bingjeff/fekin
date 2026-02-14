pub mod frame;
pub mod types;

use crate::frame::{CoordinateType, Frame, FrameRef};
use std::mem::size_of;
use std::time::Instant;

const TREE_DEPTH: usize = 3;
const BRANCHES_PER_DEPTH: usize = 3;
const NODES_PER_BRANCH: usize = 5;
const NUM_LOOP_ITERATIONS: usize = 30;

struct BenchNode {
    frame: FrameRef,
    base_q: f64,
    base_qd: f64,
    omega: f64,
    phase: f64,
}

struct UpdateStats {
    mean_us: f64,
    stddev_us: f64,
    max_us: f64,
    total_ms: f64,
}

fn coordinate_type_for(index: usize) -> CoordinateType {
    match index % 7 {
        0 => CoordinateType::XTran,
        1 => CoordinateType::YTran,
        2 => CoordinateType::ZTran,
        3 => CoordinateType::XRot,
        4 => CoordinateType::YRot,
        5 => CoordinateType::ZRot,
        _ => CoordinateType::None,
    }
}

fn build_benchmark_tree(
    depth: usize,
    branches_per_depth: usize,
    nodes_per_branch: usize,
) -> (FrameRef, Vec<BenchNode>) {
    let world_frame = Frame::new(None, [0.0, 0.0], CoordinateType::None, true);
    let mut bench_nodes = Vec::new();
    let mut frontier = vec![world_frame.clone()];
    let mut node_index = 0usize;

    for _depth_i in 0..depth {
        let mut next_frontier = Vec::new();

        for parent_frame in &frontier {
            for _branch_i in 0..branches_per_depth {
                let mut branch_parent = parent_frame.clone();

                for _node_i in 0..nodes_per_branch {
                    let coordinate_type = coordinate_type_for(node_index);
                    let is_fixed = node_index.is_multiple_of(6);
                    let base_q = 0.01 * node_index as f64;
                    let base_qd = 0.02 * ((node_index % 9) as f64 - 4.0);
                    let frame = Frame::new(
                        Some(&branch_parent),
                        [base_q, base_qd],
                        coordinate_type,
                        is_fixed,
                    );

                    bench_nodes.push(BenchNode {
                        frame: frame.clone(),
                        base_q,
                        base_qd,
                        omega: 0.3 + 0.05 * (node_index % 8) as f64,
                        phase: 0.2 * node_index as f64,
                    });

                    branch_parent = frame;
                    node_index += 1;
                }

                next_frontier.push(branch_parent);
            }
        }

        frontier = next_frontier;
    }

    (world_frame, bench_nodes)
}

fn apply_iteration_coordinates(nodes: &[BenchNode], iteration: usize) {
    let time_s = iteration as f64 * 0.01;
    for node in nodes {
        let signal = node.phase + node.omega * time_s;
        let q = node.base_q + 0.25 * signal.sin();
        let qd = node.base_qd + 0.10 * signal.cos();
        Frame::set_coordinate_value(&node.frame, [q, qd]);
    }
}

fn run_update_benchmark(root_frame: &FrameRef, nodes: &[BenchNode], loop_count: usize) -> Vec<f64> {
    let mut samples_us = Vec::with_capacity(loop_count);
    for i in 0..loop_count {
        apply_iteration_coordinates(nodes, i);
        let loop_start = Instant::now();
        Frame::update(root_frame);
        samples_us.push(loop_start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    samples_us
}

fn compute_stats(samples_us: &[f64]) -> UpdateStats {
    let n = samples_us.len() as f64;
    let sum_us: f64 = samples_us.iter().sum();
    let mean_us = sum_us / n;
    let variance = samples_us
        .iter()
        .map(|sample_us| {
            let delta = sample_us - mean_us;
            delta * delta
        })
        .sum::<f64>()
        / n;

    let max_us = samples_us.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    UpdateStats {
        mean_us,
        stddev_us: variance.sqrt(),
        max_us,
        total_ms: sum_us / 1_000.0,
    }
}

fn count_tree_nodes(root_frame: &FrameRef) -> usize {
    let mut count = 1usize;
    let children = Frame::children(root_frame);
    for child in children {
        count += count_tree_nodes(&child);
    }
    count
}

fn main() {
    let construct_start = Instant::now();
    let (world_frame, bench_nodes) =
        build_benchmark_tree(TREE_DEPTH, BRANCHES_PER_DEPTH, NODES_PER_BRANCH);
    let construct_time = construct_start.elapsed();

    let node_count = count_tree_nodes(&world_frame);
    let frame_size_bytes = size_of::<Frame>();
    let frame_ref_size_bytes = size_of::<FrameRef>();
    let estimated_tree_bytes = node_count * frame_size_bytes;

    let update_samples_us = run_update_benchmark(&world_frame, &bench_nodes, NUM_LOOP_ITERATIONS);
    let update_stats = compute_stats(&update_samples_us);

    println!("Frame benchmark");
    println!(
        "config: depth={TREE_DEPTH}, branches/depth={BRANCHES_PER_DEPTH}, nodes/branch={NODES_PER_BRANCH}"
    );
    println!("tree nodes: {node_count}");
    println!(
        "construction time: {:.3} ms",
        construct_time.as_secs_f64() * 1_000.0
    );
    println!(
        "size (shallow estimate): Frame={frame_size_bytes} B, FrameRef={frame_ref_size_bytes} B, total~= {estimated_tree_bytes} B"
    );
    println!(
        "update loops: {} -- total time: {:.3} ms)",
        NUM_LOOP_ITERATIONS,
        update_stats.total_ms
    );
    println!(
        "update stats: mean={:.3} us, stddev={:.3} us, max={:.3} us",
        update_stats.mean_us, update_stats.stddev_us, update_stats.max_us
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Matrix4d;

    const GOLDEN_LOCAL_W: [f64; 16] = [
        -0.5677006028406701,
        -0.823235097365473,
        0.0,
        0.0,
        0.823235097365473,
        -0.5677006028406701,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ];
    const GOLDEN_GLOBAL_W: [f64; 16] = [
        0.5386040182396854,
        0.13631898784789567,
        0.831458264189003,
        1.0440040887163375,
        -0.7710173896467427,
        0.47767911877483693,
        0.42113518536073213,
        0.39731520254078867,
        -0.3397615287203476,
        -0.8678938835126219,
        0.36238420297108975,
        3.228048762896403,
        0.0,
        0.0,
        0.0,
        1.0,
    ];
    const GOLDEN_LOCAL_DW: [f64; 16] = [
        -0.823235097365473,
        0.5677006028406701,
        0.0,
        0.0,
        -0.5677006028406701,
        -0.823235097365473,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    const GOLDEN_LOCAL_DDW: [f64; 16] = [
        0.5677006028406701,
        0.823235097365473,
        0.0,
        0.0,
        -0.823235097365473,
        0.5677006028406701,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    const GOLDEN_GLOBAL_V: [f64; 16] = [
        -6.938893903907228e-18,
        0.012664640408229164,
        0.09266983177147843,
        -0.1451634985620628,
        -0.012664640408229164,
        -6.938893903907228e-18,
        -0.11407064581087706,
        -0.12982778452734112,
        -0.09266983177147843,
        0.11407064581087706,
        0.0,
        -0.06700177474701621,
        0.0,
        0.0,
        0.0,
        0.0,
    ];

    struct DistalSnapshot {
        local_w: Matrix4d,
        global_w: Matrix4d,
        local_dw: Matrix4d,
        local_ddw: Matrix4d,
        global_v: Matrix4d,
    }

    fn matrix_from_array_row_major(values: &[f64; 16]) -> Matrix4d {
        Matrix4d::from_row_slice(values)
    }

    fn matrix_to_array_row_major(matrix: &Matrix4d) -> [f64; 16] {
        let mut out = [0.0; 16];
        let mut k = 0usize;
        for i in 0..4 {
            for j in 0..4 {
                out[k] = matrix[(i, j)];
                k += 1;
            }
        }
        out
    }

    fn assert_matrix_close(a: &Matrix4d, b: &Matrix4d) {
        let tol = 1.0e-12;
        for i in 0..4 {
            for j in 0..4 {
                let diff = (a[(i, j)] - b[(i, j)]).abs();
                assert!(diff <= tol, "matrix mismatch at ({i}, {j}): {diff}");
            }
        }
    }

    fn capture_final_distal_snapshot(loop_count: usize) -> DistalSnapshot {
        let (world_frame, bench_nodes) =
            build_benchmark_tree(TREE_DEPTH, BRANCHES_PER_DEPTH, NODES_PER_BRANCH);

        let _samples_us = run_update_benchmark(&world_frame, &bench_nodes, loop_count);

        let distal_frame = &bench_nodes
            .last()
            .expect("benchmark tree should include at least one non-root node")
            .frame;

        DistalSnapshot {
            local_w: Frame::local_w(distal_frame),
            global_w: Frame::global_w(distal_frame),
            local_dw: Frame::local_dw(distal_frame),
            local_ddw: Frame::local_ddw(distal_frame),
            global_v: Frame::global_v(distal_frame),
        }
    }

    fn expected_nodes(depth: usize, branches_per_depth: usize, nodes_per_branch: usize) -> usize {
        let mut total = 1usize;
        let mut branch_count = branches_per_depth;
        for _ in 0..depth {
            total += branch_count * nodes_per_branch;
            branch_count *= branches_per_depth;
        }
        total
    }

    #[test]
    fn benchmark_tree_shape_matches_requested_layout() {
        let depth = 3;
        let branches = 2;
        let nodes_per_branch = 5;
        let (world_frame, bench_nodes) = build_benchmark_tree(depth, branches, nodes_per_branch);

        assert_eq!(count_tree_nodes(&world_frame), expected_nodes(depth, branches, nodes_per_branch));
        assert_eq!(bench_nodes.len(), expected_nodes(depth, branches, nodes_per_branch) - 1);
    }

    #[test]
    fn update_stats_reports_expected_summary_values() {
        let stats = compute_stats(&[10.0, 20.0, 30.0]);
        assert!((stats.mean_us - 20.0).abs() < 1.0e-12);
        assert!((stats.stddev_us - 8.164_965_809_277_26).abs() < 1.0e-12);
        assert!((stats.max_us - 30.0).abs() < 1.0e-12);
        assert!((stats.total_ms - 0.06).abs() < 1.0e-12);
    }

    #[test]
    fn distal_node_state_matches_across_runs_at_final_iteration() {
        let run_a = capture_final_distal_snapshot(NUM_LOOP_ITERATIONS);
        let run_b = capture_final_distal_snapshot(NUM_LOOP_ITERATIONS);

        assert_matrix_close(&run_a.local_w, &run_b.local_w);
        assert_matrix_close(&run_a.global_w, &run_b.global_w);
        assert_matrix_close(&run_a.local_dw, &run_b.local_dw);
        assert_matrix_close(&run_a.local_ddw, &run_b.local_ddw);
        assert_matrix_close(&run_a.global_v, &run_b.global_v);
    }

    #[test]
    fn distal_node_state_matches_golden_snapshot_at_final_iteration() {
        let run = capture_final_distal_snapshot(NUM_LOOP_ITERATIONS);

        assert_matrix_close(&run.local_w, &matrix_from_array_row_major(&GOLDEN_LOCAL_W));
        assert_matrix_close(&run.global_w, &matrix_from_array_row_major(&GOLDEN_GLOBAL_W));
        assert_matrix_close(&run.local_dw, &matrix_from_array_row_major(&GOLDEN_LOCAL_DW));
        assert_matrix_close(&run.local_ddw, &matrix_from_array_row_major(&GOLDEN_LOCAL_DDW));
        assert_matrix_close(&run.global_v, &matrix_from_array_row_major(&GOLDEN_GLOBAL_V));
    }

    #[test]
    #[ignore = "used only when refreshing the benchmark golden snapshot"]
    fn print_distal_snapshot_for_golden_refresh() {
        let run = capture_final_distal_snapshot(NUM_LOOP_ITERATIONS);
        println!("local_w   = {:?}", matrix_to_array_row_major(&run.local_w));
        println!("global_w  = {:?}", matrix_to_array_row_major(&run.global_w));
        println!("local_dw  = {:?}", matrix_to_array_row_major(&run.local_dw));
        println!("local_ddw = {:?}", matrix_to_array_row_major(&run.local_ddw));
        println!("global_v  = {:?}", matrix_to_array_row_major(&run.global_v));
    }
}
