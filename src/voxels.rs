use bytemuck::{Pod, Zeroable};
use cgmath::{InnerSpace, Vector3, Vector4, Zero};
use rand::distributions::{Distribution, Uniform};

enum GraphRef {
    Ref(Box<Voxel>),
    Recurse(u32),
    Empty,
}

struct Voxel {
    pub average_colour: Vector4<f32>,
    pub node_ftl: GraphRef, // Front-Top-Left
    pub node_ftr: GraphRef, // Front-Top-Right
    pub node_fbl: GraphRef, // Front-Bottom-Left
    pub node_fbr: GraphRef, // Front-Bottom-Right
    pub node_btl: GraphRef, // Back-Top-Left
    pub node_btr: GraphRef, // Back-Top-Right
    pub node_bbl: GraphRef, // Back-Bottom-Left
    pub node_bbr: GraphRef, // Back-Bottom-Right
    pub flags: u32,
    pub id: u32,
}

const FTL_CELL: Vector3<f32> = Vector3::new(-0.5, 0.5, -0.5);
const FTR_CELL: Vector3<f32> = Vector3::new(0.5, 0.5, -0.5);
const FBL_CELL: Vector3<f32> = Vector3::new(-0.5, -0.5, -0.5);
const FBR_CELL: Vector3<f32> = Vector3::new(0.5, -0.5, -0.5);
const BTL_CELL: Vector3<f32> = Vector3::new(-0.5, 0.5, 0.5);
const BTR_CELL: Vector3<f32> = Vector3::new(0.5, 0.5, 0.5);
const BBL_CELL: Vector3<f32> = Vector3::new(-0.5, -0.5, 0.5);
const BBR_CELL: Vector3<f32> = Vector3::new(0.5, -0.5, 0.5);

pub const MINIMUM_GOAL_DEPTH: u32 = 6;
pub const NULL_VOXEL_INDEX: u32 = 0xFFFFFFFF;

const LEAF_VOXEL: Voxel = Voxel {
    average_colour: Vector4::new(0., 0., 0., 0.),
    node_ftl: GraphRef::Recurse(0),
    node_ftr: GraphRef::Recurse(0),
    node_fbl: GraphRef::Recurse(0),
    node_fbr: GraphRef::Recurse(0),
    node_btl: GraphRef::Recurse(0),
    node_btr: GraphRef::Recurse(0),
    node_bbl: GraphRef::Recurse(0),
    node_bbr: GraphRef::Recurse(0),
    flags: 1,
    id: NULL_VOXEL_INDEX,
};

#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct VoxelCompact {
    pub average_colour: [f32; 4],
    pub node_ftl: u32,
    pub node_ftr: u32,
    pub node_fbl: u32,
    pub node_fbr: u32,
    pub node_btl: u32,
    pub node_btr: u32,
    pub node_bbl: u32,
    pub node_bbr: u32,
    pub flags: u32,
    _alignment_space0: u32,
    _alignment_space1: u64,
}

impl VoxelCompact {
    pub fn new(
        average_colour: [f32; 4],
        node_ftl: u32,
        node_ftr: u32,
        node_fbl: u32,
        node_fbr: u32,
        node_btl: u32,
        node_btr: u32,
        node_bbl: u32,
        node_bbr: u32,
        flags: u32,
    ) -> Self {
        VoxelCompact {
            average_colour,
            node_ftl,
            node_ftr,
            node_fbl,
            node_fbr,
            node_btl,
            node_btr,
            node_bbl,
            node_bbr,
            flags,
            _alignment_space0: 0,
            _alignment_space1: 0,
        }
    }
}

fn compact_octree_from_root(root_voxel: Voxel, voxel_count: u32) -> Vec<VoxelCompact> {
    struct BfsVoxel {
        pub voxel: Box<Voxel>,
        pub parent_list: Vec<u32>, // Todo: Use an immutable singly linked list for efficiency.
    }
    fn breadth_first_search_octree(
        mut acc: Vec<VoxelCompact>,
        mut to_explore: Vec<BfsVoxel>,
        voxel_count: u32,
    ) -> Vec<VoxelCompact> {
        // If there are no more voxels to explore, return the accumulator.
        if to_explore.len() == 0 {
            // Ensure that we have explored all voxels before leaving.
            assert_eq!(voxel_count as usize, acc.len());
            return acc;
        }

        // Get the next voxel and its parent list.
        let BfsVoxel { voxel, parent_list } = to_explore.pop().unwrap();

        // Use the voxel count and our knowledge that there is no voxel with ID 1 to get the index of the voxel.
        assert!(voxel.id <= voxel_count);
        let self_index = voxel_count - voxel.id;

        // Helper for getting the index of a referenced voxel.
        let ref_to_index = |local_explore: &mut Vec<BfsVoxel>, graph_ref| {
            match graph_ref {
                GraphRef::Empty => NULL_VOXEL_INDEX,
                GraphRef::Recurse(0) => self_index,
                GraphRef::Recurse(n) => {
                    // Default to root index if we requested a parent past the root.
                    if let Some(i) = parent_list.get(parent_list.len() - n as usize) {
                        *i
                    } else {
                        0
                    }
                }
                GraphRef::Ref(voxel) => {
                    // TODO: Use a dictionary to allow for more complicated voxel graphs.
                    let mut parents = parent_list.clone();
                    parents.push(self_index);
                    let index = voxel_count - voxel.id;
                    local_explore.push(BfsVoxel {
                        voxel,
                        parent_list: parents,
                    });
                    index
                }
            }
        };

        // Get the indices of the referenced voxels and add new voxels to the breadth-first search.
        let mut bfs = vec![];
        let ftl = ref_to_index(&mut bfs, voxel.node_ftl);
        let ftr = ref_to_index(&mut bfs, voxel.node_ftr);
        let fbl = ref_to_index(&mut bfs, voxel.node_fbl);
        let fbr = ref_to_index(&mut bfs, voxel.node_fbr);
        let btl = ref_to_index(&mut bfs, voxel.node_btl);
        let btr = ref_to_index(&mut bfs, voxel.node_btr);
        let bbl = ref_to_index(&mut bfs, voxel.node_bbl);
        let bbr = ref_to_index(&mut bfs, voxel.node_bbr);

        // Add the new compact voxel to the accumulator.
        let compact_voxel = VoxelCompact::new(
            voxel.average_colour.into(),
            ftl,
            ftr,
            fbl,
            fbr,
            btl,
            btr,
            bbl,
            bbr,
            voxel.flags,
        );
        *acc.get_mut(self_index as usize).unwrap() = compact_voxel;

        // Maintain a sorted list of voxels to explore.
        bfs.append(&mut to_explore);

        // Continue the breadth-first search.
        breadth_first_search_octree(acc, bfs, voxel_count)
    }

    // Initialize an empty array of compact voxel data.
    let mut voxel_array = Vec::<VoxelCompact>::with_capacity(voxel_count as usize);
    voxel_array.resize_with(voxel_count as usize, VoxelCompact::default);

    let to_explore = vec![BfsVoxel {
        voxel: Box::new(root_voxel),
        parent_list: vec![],
    }];

    breadth_first_search_octree(voxel_array, to_explore, voxel_count)
}

pub fn generate_recursive_voxel_octree(desired_voxel_count: u32) -> Vec<VoxelCompact> {
    pub struct RandomFloats {
        random: rand::rngs::ThreadRng,
        dist: Uniform<f32>,
    }

    // Helper struct for creating random floats in the range [0, 1).
    impl RandomFloats {
        pub fn default() -> Self {
            RandomFloats {
                random: rand::thread_rng(),
                dist: Uniform::new(0., 1.),
            }
        }

        pub fn sample(&mut self) -> f32 {
            self.dist.sample(&mut self.random)
        }
    }

    // Generate a random voxel-colour.
    fn random_colour(random: &mut RandomFloats) -> Vector4<f32> {
        Vector4::new(random.sample(), random.sample(), random.sample(), 1.)
    }

    // Generate a random leaf-voxel.
    fn random_leaf(random: &mut RandomFloats, id: &mut u32, depth: u32) -> Voxel {
        let colour = random_colour(random);
        *id += 1;

        if depth >= MINIMUM_GOAL_DEPTH {
            let r = random.sample();
            if r > 0.15 {
                Voxel {
                    average_colour: colour,
                    id: *id,
                    ..LEAF_VOXEL
                }
            } else if r > 0.095 {
                Voxel {
                    average_colour: colour,
                    flags: 2, // TODO: Create a proper enum type for flags.
                    id: *id,
                    ..LEAF_VOXEL
                }
            } else {
                Voxel {
                    average_colour: colour,
                    flags: 3, // TODO: Create a proper enum type for flags.
                    id: *id,
                    ..LEAF_VOXEL
                }
            }
        } else {
            if random.sample() > 0.125 {
                Voxel {
                    average_colour: colour,
                    id: *id,
                    ..LEAF_VOXEL
                }
            } else {
                Voxel {
                    average_colour: colour,
                    flags: 3, // TODO: Create a proper enum type for flags.
                    id: *id,
                    ..LEAF_VOXEL
                }
            }
        }
    }

    // Get a random recursion depth from an exponential distribution.
    fn random_recurse(random: &mut RandomFloats, max_depth: u32) -> u32 {
        let roll = 1.0 - random.sample();
        (roll.log2().abs() as u32).min(max_depth)
    }

    // Recursively form a graph of voxels in a depth-first manner.
    fn roll_voxel_graph(random: &mut RandomFloats, depth: u32, id: &mut u32) -> Voxel {
        let mut sum_colour = Vector4::zero();
        let mut sum_count: f32 = 0.;
        let mut pop_node_option = |colour: &mut Vector4<f32>| -> GraphRef {
            let random_type = random.sample();
            // TODO: Change the moving target function to better approach the desired voxel count. Current is tuned for 256.
            let moving_target = 1. / (0.5 * depth as f32 + 1.);
            if random_type < 0.45 {
                // 45% chance of empty node
                GraphRef::Empty
            } else if random_type < 0.5_f32.powf(moving_target) {
                // Next most likely is a leaf node, but not at the first depths.
                let v = random_leaf(random, id, depth);
                *colour += v.average_colour;
                sum_count += 1.;

                GraphRef::Ref(Box::new(v))
            } else if random_type < 0.85_f32.powf(moving_target.powf(0.75)) {
                // Next most likely is a non-recursive voxel, however, should be less likely at latter depths.
                let v = roll_voxel_graph(random, depth + 1, id);
                *colour += v.average_colour;
                sum_count += 1.;

                GraphRef::Ref(Box::new(v))
            } else {
                GraphRef::Recurse(random_recurse(random, depth))
            }
        };

        let ftl = pop_node_option(&mut sum_colour);
        let ftr = pop_node_option(&mut sum_colour);
        let fbl = pop_node_option(&mut sum_colour);
        let fbr = pop_node_option(&mut sum_colour);
        let btl = pop_node_option(&mut sum_colour);
        let btr = pop_node_option(&mut sum_colour);
        let bbl = pop_node_option(&mut sum_colour);
        let bbr = pop_node_option(&mut sum_colour);

        // Ensure that the sum count is never still 0.
        sum_count = sum_count.max(1.);

        *id += 1;
        Voxel {
            average_colour: sum_colour / sum_count,
            node_ftl: ftl,
            node_ftr: ftr,
            node_fbl: fbl,
            node_fbr: fbr,
            node_btl: btl,
            node_btr: btr,
            node_bbl: bbl,
            node_bbr: bbr,
            flags: 0,
            id: *id,
        }
    }

    let random = &mut RandomFloats::default();
    loop {
        // Loop through random graphs.
        let mut id = 0;
        let v = roll_voxel_graph(random, 0, &mut id);

        // If we have generated enough voxels, compactify the octree and return it.
        if id >= desired_voxel_count {
            return compact_octree_from_root(v, id);
        }
    }
}

pub enum Intersection {
    Empty(f32),
    Collision,
    Portal(u32),
}

pub fn octree_scale_and_collision_of_point(
    position: Vector3<f32>,
    octree: &[VoxelCompact],
) -> Intersection {
    const MAX_DEPTH: u32 = 17;
    const GOAL_RADIUS_SQUARED: f32 = 0.75;
    fn f(
        scale: f32,
        iter: u32,
        p: Vector3<f32>,
        index: u32,
        octree: &[VoxelCompact],
    ) -> Intersection {
        if iter == MAX_DEPTH {
            Intersection::Collision
        } else if index == NULL_VOXEL_INDEX {
            Intersection::Empty(scale)
        } else {
            let voxel = octree[index as usize];
            if voxel.flags == 1 || voxel.flags == 4 {
                Intersection::Collision
            } else if voxel.flags == 2 {
                if p.dot(p) <= GOAL_RADIUS_SQUARED {
                    // Subtract 1 from depth since this function asserts the root as depth zero, others do not.
                    Intersection::Portal((scale.log2() - 1.).max(0.) as u32)
                } else {
                    Intersection::Empty(scale)
                }
            } else {
                let func =
                    |p: Vector3<f32>, index: u32| f(scale + scale, iter + 1, p + p, index, octree);
                if p.x > 0.0 {
                    if p.y > 0.0 {
                        if p.z > 0.0 {
                            func(p - BTR_CELL, voxel.node_btr)
                        } else {
                            func(p - FTR_CELL, voxel.node_ftr)
                        }
                    } else {
                        if p.z > 0.0 {
                            func(p - BBR_CELL, voxel.node_bbr)
                        } else {
                            func(p - FBR_CELL, voxel.node_fbr)
                        }
                    }
                } else {
                    if p.y > 0.0 {
                        if p.z > 0.0 {
                            func(p - BTL_CELL, voxel.node_btl)
                        } else {
                            func(p - FTL_CELL, voxel.node_ftl)
                        }
                    } else {
                        if p.z > 0.0 {
                            func(p - BBL_CELL, voxel.node_bbl)
                        } else {
                            func(p - FBL_CELL, voxel.node_fbl)
                        }
                    }
                }
            }
        }
    }

    if position.x.abs() > 1. || position.y.abs() > 1. || position.z.abs() > 1. {
        Intersection::Empty(1.)
    } else {
        f(1., 0, position, 0, octree)
    }
}
