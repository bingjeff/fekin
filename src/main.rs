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
}
