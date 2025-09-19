use std::collections::HashMap;
use std::usize;

use crate::Kernel;
use crate::debug::display_graph;
use crate::run::{assign_buffers, compile_kernels, run_graph};
use crate::translate::InitData;
use crate::utils::{build_search_space, generate_proof, print_kernels};
use crate::{GPUArch, GraphTerm};
use colored::Colorize;
use egraph_serialize::{ClassId, EGraph, NodeId};
use itertools::Itertools;
use luminal::prelude::NodeIndex;
use luminal::prelude::petgraph::prelude::StableGraph;
use luminal::prelude::petgraph::{Directed, Direction};
use luminal::shape::{Expression, Term};
use rand::seq::SliceRandom;
use rand::{Rng, rng};
use rustc_hash::{FxHashMap, FxHashSet};
#[cfg(feature = "metal")]
use {
    crate::{Buffer, Device},
    objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions},
    std::{ffi::c_void, ptr::NonNull},
};
#[cfg(feature = "cuda")]
use {
    cudarc::driver::{CudaContext, CudaSlice},
    std::sync::Arc,
};

const WARMUP_TRIALS: usize = 2;
const TRIALS: usize = 3;
const MAX_SEARCHED_GRAPHS: usize = 10000;
const MAX_CYCLES: usize = 1;
const INVALID_IR: &[&str] = &[
    "SwapLoops",
    "TileLoop",
    "UnpadLoop",
    "MReplace",
    "MergeLoops",
    "TiledMatmulInputA",
    "TiledMatmulInputB",
    "TiledMatmulAcc",
    "loop_level",
    "vec-of",
];

#[cfg(feature = "metal")]
#[inline]
fn with_autoreleasepool<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    objc2::rc::autoreleasepool(|_| f())
}

#[cfg(feature = "cuda")]
#[inline]
fn with_autoreleasepool<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    f()
}

type Cost = u128; // Execution time in microseconds

fn is_expression_enode(enode_label: &str) -> bool {
    matches!(
        enode_label,
        "MNum"
            | "MVar"
            | "MAdd"
            | "MSub"
            | "MMul"
            | "MDiv"
            | "MMod"
            | "MMin"
            | "MMax"
            | "MAnd"
            | "MOr"
            | "MGte"
            | "MLt"
            | "MFloorTo"
            | "MReplace"
            | "MAccum"
    ) || enode_label.starts_with("MNum:")
        || enode_label.starts_with("MVar:")
}

fn shortest_from_enode<'a>(
    egraph: &'a EGraph,
    enode: &'a NodeId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk: &mut FxHashSet<&'a NodeId>,
    cache: &mut FxHashMap<&'a NodeId, Option<Vec<&'a NodeId>>>,
) -> Option<Vec<&'a NodeId>> {
    if let Some(cached) = cache.get(enode) {
        return cached.clone();
    }
    if INVALID_IR.contains(&egraph.nodes[enode].op.as_str()) || junk.contains(enode) {
        cache.insert(enode, None);
        return None;
    }
    if seen.get(&enode).copied().unwrap_or(0) >= MAX_CYCLES {
        cache.insert(enode, None);
        return None;
    }

    *seen.entry(enode).or_insert(0) += 1;

    let out = if egraph.nodes[enode].children.is_empty() {
        // Leaf → path is just this enode
        Some(vec![enode])
    } else {
        // For each child class, take its shortest; if any child has no path → this enode invalid
        let mut acc: Vec<&'a NodeId> = vec![enode];
        let mut ok = true;

        for child in &egraph.nodes[enode].children {
            let child_class = egraph.nid_to_cid(child);
            if let Some(child_path) = extract_shortest(egraph, child_class, seen, junk, cache) {
                acc.extend(child_path);
            } else {
                ok = false;
                break;
            }
        }

        if ok { Some(acc) } else { None }
    };

    *seen.get_mut(&enode).unwrap() -= 1;

    if out.is_none() {
        junk.insert(enode);
    }
    cache.insert(enode, out.clone());
    out
}

pub fn extract_shortest<'a>(
    egraph: &'a EGraph,
    class: &'a ClassId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk: &mut FxHashSet<&'a NodeId>,
    cache: &mut FxHashMap<&'a NodeId, Option<Vec<&'a NodeId>>>,
) -> Option<Vec<&'a NodeId>> {
    // Try all enodes in the class and keep the shortest
    let mut best: Option<Vec<&'a NodeId>> = None;
    for enode in &egraph.classes()[class].nodes {
        if INVALID_IR.contains(&egraph.nodes[enode].op.as_str()) || junk.contains(enode) {
            junk.insert(enode);
            continue;
        }
        if seen.get(&enode).copied().unwrap_or(0) >= MAX_CYCLES {
            continue;
        }

        if let Some(path) = shortest_from_enode(egraph, enode, seen, junk, cache) {
            if best.as_ref().map_or(true, |b| path.len() < b.len()) {
                best = Some(path);
            }
        } else {
            junk.insert(enode);
        }
    }
    best
}

fn extract_trajectories<'a>(
    egraph: &'a EGraph,
    current_class: &'a ClassId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk_cache: &mut FxHashSet<&'a NodeId>,
    trajectory_cache: &mut FxHashMap<&'a NodeId, Vec<Vec<&'a NodeId>>>,
    waiting: usize,
) -> Vec<Vec<&'a NodeId>> {
    let mut trajectories = vec![];
    let mut enodes = egraph.classes()[current_class].nodes.iter().collect_vec();
    enodes.shuffle(&mut rng());
    'enode_loop: for enode in enodes {
        if INVALID_IR.contains(&egraph.nodes[enode].op.as_str()) {
            junk_cache.insert(enode);
            continue;
        } else if junk_cache.contains(&enode)
            || seen.get(&enode).copied().unwrap_or_default() >= MAX_CYCLES
        {
            continue;
        }
        let mut enode_trajectories = vec![];
        *seen.entry(enode).or_insert(0) += 1;
        for child in &egraph.nodes[enode].children {
            let child_first_enode = child;
            // let child = egraph.nid_to_cid(child);
            // Ask what's the child's trajectories
            if !trajectory_cache.contains_key(child) {
                let child_trajectories = if is_expression_enode(&egraph.nodes[child_first_enode].op)
                {
                    extract_shortest(
                        egraph,
                        egraph.nid_to_cid(child),
                        seen,
                        junk_cache,
                        &mut FxHashMap::default(),
                    )
                    .map(|i| vec![i])
                    .unwrap_or_default()
                } else {
                    extract_trajectories(
                        egraph,
                        egraph.nid_to_cid(child),
                        seen,
                        junk_cache,
                        trajectory_cache,
                        (waiting * enode_trajectories.len().max(1)) + trajectories.len(),
                    )
                };
                if child_trajectories.is_empty() {
                    // bad enode
                    junk_cache.insert(enode);
                    *seen.get_mut(&enode).unwrap() -= 1;
                    continue 'enode_loop;
                }
                trajectory_cache.insert(child, child_trajectories.clone());
            }

            if enode_trajectories.is_empty() {
                // First child
                for mut child_trajectory in trajectory_cache[child].clone() {
                    if egraph.nodes[enode].op != "Fused" {
                        child_trajectory.insert(0, enode);
                    }
                    enode_trajectories.push(child_trajectory);
                }
            } else if !trajectory_cache[child].is_empty() {
                // Cartisian product the current trajectories with the new trajectories
                let n_enode_traj = enode_trajectories.len();
                enode_trajectories = enode_trajectories
                    .into_iter()
                    .cartesian_product(
                        trajectory_cache[child]
                            .iter()
                            .take((MAX_SEARCHED_GRAPHS / n_enode_traj).max(1)),
                    )
                    .map(|(p, n)| {
                        if egraph.nodes[enode].op != "Fused" {
                            [p, n.clone()].concat()
                        } else {
                            n.clone()
                        }
                    })
                    .collect();
            }
            if egraph.nodes[enode].op == "Fused" {
                break;
            }
        }
        *seen.get_mut(&enode).unwrap() -= 1;

        if egraph.nodes[enode].children.is_empty() {
            // Leaf node → single-element trajectory
            trajectories.push(vec![enode]);
        } else {
            // Add combined trajectories
            trajectories.extend(enode_trajectories);
        }
        if trajectories.len() * waiting > MAX_SEARCHED_GRAPHS {
            break; // Only pick the first valid (non cycling) enode for expressions
        }
    }
    trajectories
}

fn count_trajectories<'a>(
    egraph: &'a EGraph,
    current_class: &'a ClassId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk_cache: &mut FxHashSet<&'a NodeId>,
    trajectory_cache: &mut FxHashMap<&'a NodeId, usize>,
) -> usize {
    let mut trajectories = 0;
    'enode_loop: for enode in egraph.classes()[current_class].nodes.iter().rev() {
        if INVALID_IR.contains(&egraph.nodes[enode].op.as_str()) {
            junk_cache.insert(enode);
            continue;
        } else if junk_cache.contains(&enode)
            || seen.get(&enode).copied().unwrap_or_default() >= MAX_CYCLES
        {
            continue;
        }
        let mut enode_trajectories = 0;
        *seen.entry(enode).or_insert(0) += 1;
        for child in &egraph.nodes[enode].children {
            let child_first_enode = child;
            // let child = egraph.nid_to_cid(child);
            // Ask what's the child's trajectories
            if !trajectory_cache.contains_key(child) {
                let child_trajectories = if is_expression_enode(&egraph.nodes[child_first_enode].op)
                {
                    1
                } else {
                    count_trajectories(
                        egraph,
                        egraph.nid_to_cid(child),
                        seen,
                        junk_cache,
                        trajectory_cache,
                    )
                };
                if child_trajectories == 0 {
                    // bad enode
                    junk_cache.insert(enode);
                    *seen.get_mut(&enode).unwrap() -= 1;
                    continue 'enode_loop;
                }
                trajectory_cache.insert(child, child_trajectories.clone());
            }

            if enode_trajectories == 0 {
                // First child
                enode_trajectories = trajectory_cache[child];
            } else if trajectory_cache[child] != 0 {
                // Cartisian product the current trajectories with the new trajectories
                enode_trajectories = enode_trajectories.max(1) * trajectory_cache[child].max(1);
            }
            if egraph.nodes[enode].op == "Fused" {
                break;
            }
        }
        *seen.get_mut(&enode).unwrap() -= 1;

        if egraph.nodes[enode].children.is_empty() {
            // Leaf node → single-element trajectory
            trajectories += 1;
        } else {
            // Add combined trajectories
            trajectories += enode_trajectories;
        }
    }
    trajectories
}

pub fn human_readable(n: usize) -> String {
    const THOUSAND: f64 = 1_000.0;
    const MILLION: f64 = 1_000_000.0;
    const BILLION: f64 = 1_000_000_000.0;
    const TRILLION: f64 = 1_000_000_000_000.0;

    let n_f = n as f64;

    if n_f >= TRILLION {
        format!("{:.1} trillion", n_f / TRILLION)
    } else if n_f >= BILLION {
        format!("{:.1} billion", n_f / BILLION)
    } else if n_f >= MILLION {
        format!("{:.1} million", n_f / MILLION)
    } else if n_f >= THOUSAND {
        format!("{:.1} thousand", n_f / THOUSAND)
    } else {
        n.to_string()
    }
}

pub fn search(
    graph: &StableGraph<GraphTerm, ()>,
    steps: usize,
    inputs: &[(String, InitData)],
    arch: GPUArch,
    dyn_vars: &FxHashMap<char, usize>,
) -> Option<StableGraph<GraphTerm, ()>> {
    let og = graph.clone();
    let egraph = build_search_space(graph, steps);
    if option_env!("PRINT_EGGLOG").is_some() {
        println!(
            "Total Trajectories in E-Graph: {}",
            human_readable(count_trajectories(
                &egraph,
                &egraph.root_eclasses[0],
                &mut FxHashMap::default(),
                &mut FxHashSet::default(),
                &mut FxHashMap::default(),
            ))
            .bold()
        );
    }
    // display_egraph(&egraph);
    let mut trajectories = extract_trajectories(
        &egraph,
        &egraph.root_eclasses[0],
        &mut FxHashMap::default(),
        &mut FxHashSet::default(),
        &mut FxHashMap::default(),
        1,
    );
    trajectories.shuffle(&mut rng());
    // build loop level -> enode mapping
    let mut loop_level_values = FxHashMap::default();
    for (id, _) in &egraph.class_data {
        if egraph.classes()[id]
            .nodes
            .iter()
            .any(|n| egraph.nodes[n].op == "loop_level")
        {
            loop_level_values.insert(
                id,
                egraph.classes()[id]
                    .nodes
                    .iter()
                    .find_map(|n| egraph.nodes[n].op.parse::<i32>().ok())
                    .unwrap(),
            );
        }
    }
    let mut loop_level_map = FxHashMap::default();
    for (id, node) in &egraph.nodes {
        if node.op == "loop_level" {
            for child in &node.children {
                for node in &egraph.classes()[egraph.nid_to_cid(child)].nodes {
                    loop_level_map.insert(node, loop_level_values[egraph.nid_to_cid(id)]);
                }
            }
        }
    }

    // if std::env::var("DEBUG").is_ok() {
    //     // make sure all IR nodes loop levels
    //     for (id, node) in &egraph.nodes {
    //         if node.eclass.to_string().starts_with("IR-") {
    //             assert!(
    //                 loop_level_map.contains_key(id),
    //                 "Loop level not found for {}",
    //                 node.op
    //             );
    //         }
    //     }
    // }

    // Now we have DFS trajectories
    let mut ref_outputs: Vec<Vec<f32>> = vec![];
    let mut best_time = u128::MAX;
    let mut fastest = "".to_string();
    let mut best_graph = None;
    let mut valid_graphs = 0;
    let total_trajectories = trajectories.len().min(MAX_SEARCHED_GRAPHS);
    let mut og_kernels = "".to_string();
    let mut ui_functions = None;
    if option_env!("DEBUG").is_none() {
        ui_functions = Some(crate::utils::search_ui());
    };
    let mut seen = FxHashSet::default();
    let mut kernel_timings = FxHashMap::default();
    let mut possibles = 0;
    'trajectory_loop: for (n, trajectory) in trajectories.into_iter().enumerate() {
        // crate::egraph_debugger::display_egraph_with_path(&egraph, &trajectory);
        // Build termdag
        let graph = extraction_to_graph(&egraph, &trajectory, &loop_level_map);

        let Some((kernels, gmem_mapping)) =
            crate::codegen::codegen(graph.clone(), arch.clone(), dyn_vars)
        else {
            continue;
        };
        possibles += 1;
        let inputs = inputs
        	.into_iter()
         	.filter_map(|(l, d)|
          		graph.node_indices().find(|n| matches!(graph.node_weight(*n).unwrap(), GraphTerm::GMEM { label } if label == l)).map(|i| (i, d))
          	)
          	.collect_vec();
        match &arch {
            GPUArch::CUDA => {
                let k = print_kernels(&kernels);
                if seen.contains(&k) {
                    continue;
                } else {
                    seen.insert(k);
                }
                if kernels.node_weights().any(|k| {
                    kernel_timings
                        .get(&k.code)
                        .map(|i| *i > best_time)
                        .unwrap_or_default()
                }) {
                    // At least one kernel is slower than the fastest graph time
                    continue;
                }
                if let Some((us, outs)) = cost(
                    &kernels,
                    &inputs,
                    &gmem_mapping,
                    dyn_vars,
                    &mut kernel_timings,
                ) {
                    valid_graphs += 1;
                    if let Some((progress, logs, title, _)) = &ui_functions {
                        progress(((n as f32 / total_trajectories as f32) * 100.0) as u16);
                        logs(print_kernels(&kernels));
                        title(format!(
                            "Graph {valid_graphs} Best {best_time}µs Current {us}µs"
                        ));
                    } else if option_env!("DEBUG").is_some() {
                        println!("{}", print_kernels(&kernels));
                        println!("Graph {valid_graphs} Best {best_time}µs Current {us}µs");
                        if ref_outputs.is_empty() {
                            ref_outputs = outs;
                        } else {
                            for (a, b) in ref_outputs.iter().zip(&outs) {
                                for (x, y) in a.iter().zip(b) {
                                    if (x - y).abs() >= 1e-4 {
                                        if option_env!("DEBUG").is_some() {
                                            // display_graph(&graph, &[]);
                                            println!(
                                                "REF: {:?}",
                                                &ref_outputs
                                                    .iter()
                                                    .map(|v| &v[..v.len().min(20)])
                                                    .collect_vec()
                                            );
                                            println!(
                                                "New: {:?}",
                                                &outs
                                                    .iter()
                                                    .map(|v| &v[..v.len().min(20)])
                                                    .collect_vec()
                                            );
                                            crate::debug::display_graph(&og);
                                            crate::debug::display_graph(&graph);
                                            generate_proof(&og, &graph);
                                            println!("{}", og_kernels);
                                            println!("{}", print_kernels(&kernels));
                                            panic!(
                                                "{} {x} != {y} {}",
                                                "Output Mismatch".bold().on_bright_red(),
                                                (x - y).abs()
                                            );
                                        }
                                        continue 'trajectory_loop;
                                    }
                                }
                            }
                            println!("{}", "Outputs Validated".bold().on_bright_green());
                        }
                    }
                    let kernel_string = print_kernels(&kernels);
                    if og_kernels.is_empty() {
                        og_kernels = kernel_string.clone();
                    }
                    if us < best_time {
                        best_time = us;
                        best_graph = Some(graph);
                        fastest = kernel_string;
                    }
                }
            }
            GPUArch::Metal(_) => {
                let k = print_kernels(&kernels);
                if seen.contains(&k) {
                    continue;
                } else {
                    seen.insert(k.clone());
                }
                if kernels.node_weights().any(|k| {
                    kernel_timings
                        .get(&k.code)
                        .map(|i| *i > best_time)
                        .unwrap_or_default()
                }) {
                    // At least one kernel is slower than the fastest graph time
                    continue;
                }
                if let Some((us, outs)) = cost(
                    &kernels,
                    &inputs,
                    &gmem_mapping,
                    dyn_vars,
                    &mut kernel_timings,
                ) {
                    valid_graphs += 1;
                    if let Some((progress, logs, title, _)) = &ui_functions {
                        progress(((n as f32 / total_trajectories as f32) * 100.0) as u16);
                        logs(k);
                        title(format!(
                            "Graph {valid_graphs} Best {best_time}µs Current {us}µs"
                        ));
                    } else if option_env!("DEBUG").is_some() {
                        println!("{k}");
                        println!("Graph {valid_graphs} Best {best_time}µs Current {us}µs");
                        if ref_outputs.is_empty() {
                            ref_outputs = outs;
                            println!("{}", "Initial".bold().on_bright_green());
                        } else {
                            for (a, b) in ref_outputs.iter().zip(&outs) {
                                for (x, y) in a.iter().zip(b) {
                                    if (x - y).abs() >= 1e-4 {
                                        if option_env!("DEBUG").is_some() {
                                            // display_graph(&graph, &[]);
                                            println!(
                                                "REF: {:?}",
                                                &ref_outputs
                                                    .iter()
                                                    .map(|v| &v[..v.len().min(20)])
                                                    .collect_vec()
                                            );
                                            println!(
                                                "New: {:?}",
                                                &outs
                                                    .iter()
                                                    .map(|v| &v[..v.len().min(20)])
                                                    .collect_vec()
                                            );
                                            // crate::utils::generate_proof(&og, &graph);
                                            println!("{}", og_kernels);
                                            println!("{}", print_kernels(&kernels));
                                            crate::debug::display_multiple_graphs(&[&og, &graph]);
                                            panic!(
                                                "{} {x} != {y}",
                                                "Output Mismatch".bold().on_bright_red()
                                            );
                                        }
                                        continue 'trajectory_loop;
                                    }
                                }
                            }
                            println!("{}", "Outputs Validated".bold().on_bright_green());
                        }
                    }
                    let kernel_string = print_kernels(&kernels);
                    if og_kernels.is_empty() {
                        og_kernels = kernel_string.clone();
                    }
                    // let us = kernels.node_count() as u128;
                    if us < best_time {
                        best_time = us;
                        best_graph = Some(graph);
                        fastest = kernel_string;
                    }
                }
            }
        }
    }
    if let Some((_, _, _, exit)) = &ui_functions {
        exit();
    }
    println!("FASTEST ({}ms): {fastest}", best_time / 1000);
    println!("Valids: {:?} / {:?}", possibles, total_trajectories);
    best_graph
}

pub fn extraction_to_graph(
    egraph: &EGraph,
    trajectory: &[&NodeId],
    loop_level_map: &FxHashMap<&NodeId, i32>,
) -> StableGraph<GraphTerm, (), Directed> {
    let mut g = StableGraph::new();

    fn build_expression<'a>(
        egraph: &EGraph,
        trajectory: &[&'a NodeId],
        current: &mut usize,
    ) -> Expression {
        let nid = trajectory[*current];
        let enode = &egraph.nodes[nid];
        let op = enode.op.as_str();

        match op {
            // unary math
            "MNeg" | "MRecip" => {
                *current += 1;
                let c0 = build_expression(egraph, trajectory, current);
                match op {
                    "MNeg" => c0 * -1,
                    "MRecip" => 1 / c0,
                    _ => unreachable!(),
                }
            }

            // binary math
            "MAdd" | "MSub" | "MMul" | "MDiv" | "MMod" | "MMin" | "MMax" | "MAnd" | "MOr"
            | "MGte" | "MLt" | "MFloorTo" => {
                *current += 1;
                let lhs = build_expression(egraph, trajectory, current);
                *current += 1;
                let rhs = build_expression(egraph, trajectory, current);
                match op {
                    "MAdd" => lhs + rhs,
                    "MSub" => lhs - rhs,
                    "MMul" => lhs * rhs,
                    "MDiv" => lhs / rhs,
                    "MMod" => lhs % rhs,
                    "MMin" => lhs.min(rhs),
                    "MMax" => lhs.max(rhs),
                    "MAnd" => lhs & rhs,
                    "MOr" => lhs | rhs,
                    "MGte" => lhs.gte(rhs),
                    "MLt" => lhs.lt(rhs),
                    "MFloorTo" => lhs / rhs * rhs, // TODO: real floorto in Expression
                    _ => unreachable!(),
                }
            }

            // wrappers around a literal/var child
            "MNum" | "MVar" => {
                *current += 1;
                build_expression(egraph, trajectory, current)
            }

            // “accumulator” token
            "MAccum" => {
                *current += 1;
                Expression::from(Term::Acc('a'))
            }

            // inline literals / names encoded in `op`
            _ if op.starts_with("MNum:") => {
                let num: i64 = op["MNum:".len()..].parse().expect("invalid MNum literal");
                Expression::from(num as usize)
            }
            _ if op.starts_with("MVar:") => {
                let name = &op["MVar:".len()..];
                Expression::from(name.chars().next().unwrap())
            }
            _ if op.starts_with("Boxed(\"") => {
                let name = op.replace("Boxed(\"", "").replace("\")", "");
                Expression::from(name.chars().next().unwrap())
            }
            op => enode
                .op
                .parse::<usize>()
                .map(|i| i.into())
                .unwrap_or_else(|_| panic!("unsupported expression op '{op}'")),
        }
    }

    // --- IR builder: places nodes in `g` and returns NodeIndex ---
    fn build_ir<'a, 'b>(
        egraph: &EGraph,
        trajectory: &[&'a NodeId],
        current: &mut usize,
        g: &mut StableGraph<GraphTerm, (), Directed>,
        loop_level_map: &FxHashMap<&'a NodeId, i32>,
        prev_placed: &'b mut FxHashMap<&'a NodeId, NodeIndex>,
        no_place: bool,
    ) -> NodeIndex {
        let node_choice = trajectory[*current];
        let enode = &egraph.nodes[node_choice];
        let op = enode.op.as_str();

        match op {
            // Leaf-ish memory node
            "GMEM" => {
                *current += 1;
                if no_place {
                    NodeIndex::default()
                } else {
                    *prev_placed.entry(node_choice).or_insert_with(|| {
                        let label = egraph.nodes[&enode.children[0]]
                            .op
                            .replace("Boxed(\"", "")
                            .replace("\")", "");
                        g.add_node(GraphTerm::GMEM { label })
                    })
                }
            }

            // LoopIn/LoopOut = (Loop* <expr> <Math> <Math>)
            "LoopIn" | "LoopOut" => {
                *current += 1;
                let already = prev_placed.contains_key(node_choice);
                // child expr
                let child_one = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );

                // range, stride (expressions)
                *current += 1;
                let range = build_expression(egraph, trajectory, current);
                *current += 1;
                let stride = build_expression(egraph, trajectory, current);

                if no_place {
                    NodeIndex::default()
                } else if let Some(&n) = prev_placed.get(node_choice) {
                    n
                } else {
                    let term = match op {
                        "LoopIn" => GraphTerm::LoopIn { range, stride },
                        "LoopOut" => GraphTerm::LoopOut { range, stride },
                        _ => unreachable!(),
                    };
                    let r = g.add_node(term);
                    prev_placed.insert(node_choice, r);
                    g.add_edge(child_one, r, ());
                    r
                }
            }

            // TCMatmul = (TCMatmul <A> <B> <a_k> <b_k> <a_in> <b_in> <c_in> <k_outer>)
            "TCMatmul" => {
                *current += 1;
                let already = prev_placed.contains_key(node_choice);

                let src_a = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );
                *current += 1;
                let src_b = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );

                *current += 1;
                let a_k_stride = build_expression(egraph, trajectory, current);
                *current += 1;
                let b_k_stride = build_expression(egraph, trajectory, current);
                *current += 1;
                let a_inner_stride = build_expression(egraph, trajectory, current);
                *current += 1;
                let b_inner_stride = build_expression(egraph, trajectory, current);
                *current += 1;
                let c_inner_stride = build_expression(egraph, trajectory, current);
                *current += 1;
                let k_outer_loops = build_expression(egraph, trajectory, current);

                if no_place {
                    NodeIndex::default()
                } else if let Some(&n) = prev_placed.get(node_choice) {
                    n
                } else {
                    let r = g.add_node(GraphTerm::TCMatmul {
                        a_k_stride,
                        b_k_stride,
                        a_inner_stride,
                        b_inner_stride,
                        c_inner_stride,
                        k_outer_loops,
                    });
                    prev_placed.insert(node_choice, r);
                    g.add_edge(src_a, r, ());
                    g.add_edge(src_b, r, ());
                    r
                }
            }

            // Binary = (Binary <Op> <Expr> <Expr>)
            "Binary" => {
                *current += 1;
                let already = prev_placed.contains_key(node_choice);
                let op_term = match egraph.nodes[trajectory[*current]].op.as_str() {
                    "Add" => GraphTerm::Add,
                    "Mul" => GraphTerm::Mul,
                    "Max" => GraphTerm::Max,
                    other => panic!("unknown binary IR operator: {other}"),
                };
                *current += 1;

                let child_one = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );
                *current += 1;
                let child_two = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );

                if no_place {
                    NodeIndex::default()
                } else if let Some(&n) = prev_placed.get(node_choice) {
                    n
                } else {
                    let r = g.add_node(op_term);
                    prev_placed.insert(node_choice, r);
                    g.add_edge(child_one, r, ());
                    g.add_edge(child_two, r, ());
                    r
                }
            }

            // Unary = (Unary <Op> <Expr>)
            "Unary" => {
                *current += 1;
                let already = prev_placed.contains_key(node_choice);
                let op_term = match egraph.nodes[trajectory[*current]].op.as_str() {
                    "Exp2" => GraphTerm::Exp2,
                    "Log2" => GraphTerm::Log2,
                    "Sin" => GraphTerm::Sin,
                    "Recip" => GraphTerm::Recip,
                    "Neg" => GraphTerm::Neg,
                    "Sqrt" => GraphTerm::Sqrt,
                    other => panic!("unknown unary IR operator: {other}"),
                };
                *current += 1;

                let child = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );

                if no_place {
                    NodeIndex::default()
                } else if let Some(&n) = prev_placed.get(node_choice) {
                    n
                } else {
                    let r = g.add_node(op_term);
                    prev_placed.insert(node_choice, r);
                    g.add_edge(child, r, ());
                    r
                }
            }
            op => panic!("unsupported IR op: {op}"),
        }
    }

    build_ir(
        egraph,
        trajectory,
        &mut 0,
        &mut g,
        loop_level_map,
        &mut FxHashMap::default(),
        false,
    );
    for n in g.node_indices() {
        if g.neighbors_undirected(n).next().is_none() {
            display_graph(&g);
            panic!("free-standing node found in graph");
        }
    }
    g
}

fn cost<'a>(
    kernels: &StableGraph<Kernel, (usize, usize), Directed>,
    inputs: &[(NodeIndex, &InitData)],
    gmem_mapping: &HashMap<NodeIndex, usize>,
    dyn_vars: &FxHashMap<char, usize>,
    kernel_timings: &mut FxHashMap<String, u128>,
) -> Option<(Cost, Vec<Vec<f32>>)> {
    with_autoreleasepool(|| {
        // Get buffer info
        let (int_buffers, int_buffer_map) = assign_buffers(&kernels);
        let compiled_kernels = compile_kernels(&kernels);
        #[cfg(feature = "metal")]
        let device = MTLCreateSystemDefaultDevice().unwrap();
        #[cfg(feature = "cuda")]
        let ctx = CudaContext::new(0).unwrap(); // will need to expand beyond single host
        // Copy input buffers over
        let mut inputs = inputs
            .into_iter()
            .filter(|(n, _)| gmem_mapping.contains_key(n))
            .map(|(n, b)| {
                (
                    gmem_mapping[n],
                    (
                        #[cfg(feature = "metal")]
                        match b {
                            InitData::Data(d) => copy_metal_buffer(d, &device),
                            InitData::Expr(e) => {
                                copy_metal_buffer(&vec![e.exec(dyn_vars).unwrap() as f32], &device)
                            }
                        },
                        #[cfg(feature = "cuda")]
                        match b {
                            InitData::Data(d) => copy_cuda_buffer(d, ctx.clone()),
                            InitData::Expr(e) => copy_cuda_buffer(
                                &vec![e.exec(dyn_vars).unwrap() as f32],
                                ctx.clone(),
                            ),
                        },
                        false,
                    ),
                )
            })
            .collect::<FxHashMap<_, _>>();
        // Warm up resources (buffer allocation, kernel compiler, etc.)
        for _ in 0..WARMUP_TRIALS {
            #[cfg(feature = "metal")]
            run_graph(
                &mut inputs,
                &kernels,
                dyn_vars,
                &compiled_kernels,
                &int_buffers,
                &int_buffer_map,
            );
            #[cfg(feature = "cuda")]
            run_graph(
                &mut inputs,
                &kernels,
                dyn_vars,
                &compiled_kernels,
                &int_buffers,
                &int_buffer_map,
            );
        }
        // Test runtime
        let mut micros = vec![];
        let mut outputs = vec![];

        for _ in 0..TRIALS {
            let (o, m_val) = {
                #[cfg(feature = "metal")]
                {
                    crate::run::run_graph(
                        &mut inputs,
                        &kernels,
                        dyn_vars,
                        &compiled_kernels,
                        &int_buffers,
                        &int_buffer_map,
                    )
                }

                #[cfg(feature = "cuda")]
                {
                    run_graph(
                        &mut inputs,
                        &kernels,
                        dyn_vars,
                        &compiled_kernels,
                        &int_buffers,
                        &int_buffer_map,
                    )
                }
            };
            // for node in kernels.node_indices() {
            //     let kernel = &kernels[node];
            //     if kernel.code != "Inputs" && kernel.code != "Outputs" {
            //         kernel_timings.insert(
            //             kernel.code.clone(),
            //             timings
            //                 .iter()
            //                 .find(|(n, _)| *n == node)
            //                 .map(|(_, i)| *i)
            //                 .unwrap(),
            //         );
            //     }
            // }
            // println!("timings: {timings:?}");
            outputs = o;
            micros.push(m_val);
        }
        Some((
            micros.into_iter().sum::<u128>() / TRIALS as u128,
            #[cfg(feature = "metal")]
            outputs.iter().map(copy_metal_buffer_back).collect_vec(),
            #[cfg(feature = "cuda")]
            outputs.iter().map(copy_cuda_buffer_back).collect_vec(),
        ))
    })
}

#[cfg(feature = "cuda")]
pub fn copy_cuda_buffer(v: &[f32], ctx: Arc<CudaContext>) -> CudaSlice<f32> {
    assert!(!v.is_empty(), "Can't copy empty slice to device");

    // Then copy host data to the allocated device memory
    let stream = ctx.default_stream();
    let mut dst = stream.alloc_zeros::<f32>(v.len()).unwrap();
    stream.memcpy_htod(v, &mut dst).unwrap();
    dst
}

/// Device -> Host (like contents() memcpy back)
#[cfg(feature = "cuda")]
pub fn copy_cuda_buffer_back(buf: &CudaSlice<f32>) -> Vec<f32> {
    buf.stream().memcpy_dtov(buf).unwrap()
}

#[cfg(feature = "metal")]
pub fn copy_metal_buffer(v: &Vec<f32>, device: &Device) -> Buffer {
    assert!(v.len() > 0);
    unsafe {
        let ptr = NonNull::new(v.as_ptr() as *mut c_void).unwrap();
        device
            .newBufferWithBytes_length_options(
                ptr,
                (v.len() * 4) as _,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    }
}
#[cfg(feature = "metal")]
pub fn copy_metal_buffer_back(v: &Buffer) -> Vec<f32> {
    let mut data = vec![0f32; v.length() as usize / size_of::<f32>()];
    let ptr = v.contents().as_ptr() as *mut f32;
    for (i, d) in data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }
    data
}

pub fn make_test_inputs(
    graph: &StableGraph<GraphTerm, ()>,
    dyn_map: &FxHashMap<char, usize>,
    inits: &[(String, InitData)],
) -> Vec<(String, InitData)> {
    // Go through each GMEM and work out the size
    let mut inputs = vec![];
    let mut rng = rng();
    for node in graph.externals(Direction::Incoming) {
        if let GraphTerm::GMEM { label } = graph.node_weight(node).unwrap() {
            if let Some(init) = inits.iter().find(|(n, _)| n == label) {
                inputs.push(init.clone());
                continue;
            }
            // Walk down the loopins to find the max size
            let mut size = Expression::from(1);
            let mut curr = graph
                .neighbors_directed(node, Direction::Outgoing)
                .next()
                .unwrap();
            loop {
                if let GraphTerm::LoopIn { range, stride, .. } = graph.node_weight(curr).unwrap() {
                    size = size.max(stride.substitute('z', *range));
                    // size = size.max(stride.substitute('z', *range - 1) + 1); // why were we doing this?
                    curr = graph
                        .neighbors_directed(curr, Direction::Outgoing)
                        .next()
                        .unwrap();
                } else {
                    break;
                }
            }
            inputs.push((
                label.clone(),
                InitData::Data(
                    (0..size.exec(&dyn_map).unwrap())
                        .map(|_| rng.random_range(-1e-3..1e-3))
                        .collect(),
                ),
            ));
        }
    }
    inputs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        translate::{MetaGraph, SubGraph, translate_graph},
        utils::build_search_space,
    };

    fn build_minimal_add_graph() -> (luminal::graph::Graph, MetaGraph, SubGraph) {
        use luminal::graph::Graph;

        let mut cx = Graph::new();
        let a = cx.tensor(3).set([1., 2., 3.]);
        let b = cx.tensor(3).set([4., 5., 6.]);
        let c = (a + b).sqrt();
        let d = c * a;
        let _e = d.sum(0).retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);
        let meta_node = meta_graph
            .node_indices()
            .next()
            .expect("MetaGraph unexpectedly empty");
        let sub = meta_graph
            .node_weight(meta_node)
            .expect("Missing subgraph at meta node")
            .clone();

        (cx, meta_graph, sub)
    }

    fn build_nonempty_egraph() -> EGraph {
        // Keep `cx` and `meta_graph` alive while we build the egraph
        let (_cx, meta_graph, sub) = build_minimal_add_graph();
        let e = build_search_space(&sub, /*iters=*/ 2);
        // `_cx` and `meta_graph` can drop now; `e` no longer needs them
        drop(meta_graph);
        e
    }

    #[test]
    fn test_egraph_is_nonempty_and_has_root() {
        let egraph = build_nonempty_egraph();
        assert!(!egraph.classes().is_empty(), "EGraph should have classes");
        assert!(
            !egraph.root_eclasses.is_empty(),
            "EGraph should have a root"
        );
    }

    #[test]
    fn test_extract_trajectories_invalid_ir_filtering() {
        let egraph = build_nonempty_egraph();

        if egraph.classes().is_empty() || egraph.root_eclasses.is_empty() {
            return;
        }

        let root_class = &egraph.root_eclasses[0];
        let mut seen = FxHashMap::default();
        let mut junk_cache = FxHashSet::default();
        let mut trajectory_cache = FxHashMap::default();

        let trajectories = extract_trajectories(
            &egraph,
            root_class,
            &mut seen,
            &mut junk_cache,
            &mut trajectory_cache,
            1,
        );

        // Check that trajectories don't contain INVALID_IR operations
        for trajectory in trajectories {
            for &node in &trajectory {
                let op_name = &egraph.nodes[node].op;
                assert!(
                    !INVALID_IR.contains(&op_name.as_str()),
                    "Trajectory contains invalid IR operation: {}",
                    op_name
                );
            }
        }
    }

    #[test]
    fn test_is_expression_enode() {
        // Test that expression enodes are correctly identified
        assert!(is_expression_enode("MNum"));
        assert!(is_expression_enode("MVar"));
        assert!(is_expression_enode("MAdd"));
        assert!(is_expression_enode("MNum:42"));
        assert!(is_expression_enode("MVar:x"));

        // Test that non-expression enodes are not identified
        assert!(!is_expression_enode("GMEM"));
        assert!(!is_expression_enode("LoopIn"));
        assert!(!is_expression_enode("Add"));
        assert!(!is_expression_enode("Invalid"));
    }
}
