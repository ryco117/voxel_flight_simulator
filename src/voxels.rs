use arr_macro::arr;
use bytemuck::{Pod, Zeroable};
use cgmath::{InnerSpace, Vector3, Vector4, Zero};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::StdRng,
    SeedableRng,
};

use fast_loaded_dice_roller as fldr;

// The types of reference that a voxel can have to its child voxels.
enum GraphRef {
    Ref(Box<Voxel>),
    Recurse(u32),
    Empty,
}

// The types of voxels that can be used.
#[derive(Clone, Copy)]
enum VoxelType {
    Complex = 0,
    Colour,
    Portal,
    Mirror,
}

struct Voxel {
    pub average_colour: Vector4<f32>,
    pub children: [GraphRef; 8],
    pub vtype: VoxelType,
    pub id: u32,
}

pub const MINIMUM_GOAL_DEPTH: u32 = 6;
pub const MAXIMUM_VOXEL_DEPTH: u32 = 15;
pub const MAXIMUM_GOAL_DEPTH: u32 = MAXIMUM_VOXEL_DEPTH - 1;
pub const NULL_VOXEL_INDEX: u32 = 0xFFFF_FFFF;

const LEAF_VOXEL: Voxel = Voxel {
    average_colour: Vector4::new(0., 0., 0., 0.),
    children: arr![GraphRef::Recurse(0); 8],
    vtype: VoxelType::Colour,
    id: NULL_VOXEL_INDEX,
};

#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct VoxelCompact {
    pub average_colour: [f32; 4],
    pub children: [u32; 8],
    pub flags: u32,
    _alignment_space: [u32; 3],
}

impl VoxelCompact {
    pub fn new(average_colour: [f32; 4], children: [u32; 8], flags: u32) -> Self {
        VoxelCompact {
            average_colour,
            children,
            flags,
            _alignment_space: [0; 3],
        }
    }
}

fn compact_octree_from_root(root_voxel: Voxel, voxel_count: u32) -> Vec<VoxelCompact> {
    struct BfsVoxel {
        pub voxel: Box<Voxel>,
        pub parent_list: Vec<u32>,
    }
    fn breadth_first_search_octree(
        mut acc: Vec<VoxelCompact>,
        mut to_explore: Vec<BfsVoxel>,
        voxel_count: u32,
    ) -> Vec<VoxelCompact> {
        // If there are no more voxels to explore, return the accumulator.
        if to_explore.is_empty() {
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
        let children = voxel.children.map(|voxel| ref_to_index(&mut bfs, voxel));

        // Add the new compact voxel to the accumulator.
        let compact_voxel =
            VoxelCompact::new(voxel.average_colour.into(), children, voxel.vtype as u32);
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

#[derive(Default)]
pub struct OctreeStats {
    pub goal_count: u32,
    pub voxel_count: u32,
}

// Helper struct for creating random floats uniformly in the range [0, 1).
pub struct RandomFloats {
    leaf_after_portal_depth: fldr::Generator,
    leaf_before_portal_depth: fldr::Generator,
    fair_coin: fldr::rand::RngCoin<StdRng>,

    random: rand::rngs::StdRng,
    dist: Uniform<f32>,
}

// Generate a random voxel-octree stored in a contiguous array.
pub fn generate_recursive_voxel_octree(
    random: &mut RandomFloats,
    desired_voxel_count: u32,
    desired_portal_count: u32,
) -> (Vec<VoxelCompact>, OctreeStats) {
    // Helper to generate a random voxel-colour.
    fn random_colour(random: &mut RandomFloats) -> Vector4<f32> {
        Vector4::new(random.samplef(), random.samplef(), random.samplef(), 1.)
    }

    // Helper to generate a random leaf-voxel.
    fn random_leaf(random: &mut RandomFloats, depth: u32, stats: &mut OctreeStats) -> Voxel {
        let colour = random_colour(random);
        stats.voxel_count += 1;
        let id = stats.voxel_count;

        match random.sample_leaf(depth >= MINIMUM_GOAL_DEPTH) {
            VoxelType::Colour => Voxel {
                average_colour: colour,
                id,
                ..LEAF_VOXEL
            },
            VoxelType::Portal => {
                stats.goal_count += 1;
                Voxel {
                    average_colour: colour,
                    vtype: VoxelType::Portal,
                    id,
                    ..LEAF_VOXEL
                }
            }
            VoxelType::Mirror => Voxel {
                average_colour: colour,
                vtype: VoxelType::Mirror,
                id,
                ..LEAF_VOXEL
            },
            VoxelType::Complex => unreachable!(),
        }
    }

    // Get a random recursion depth from an exponential distribution.
    fn random_recurse(random: &mut RandomFloats, max_depth: u32) -> u32 {
        let roll = 1.0 - random.samplef();

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        (roll.log(1.8).abs() as u32).min(max_depth)
    }

    // Recursively form a graph of voxels in a depth-first manner.
    fn roll_voxel_graph(random: &mut RandomFloats, depth: u32, stats: &mut OctreeStats) -> Voxel {
        let mut sum_colour = Vector4::zero();
        let mut sum_count: f32 = 0.;
        let mut pop_node_option = |colour: &mut Vector4<f32>| -> GraphRef {
            let random_type = random.samplef();
            // TODO: Change the moving target function to better approach the desired voxel count. Current is tuned for 256.
            let moving_target = 1. / (0.65 * depth as f32 + 1.);
            if random_type < 0.45 {
                // 45% chance of empty node
                GraphRef::Empty
            } else if random_type < 0.45_f32.powf(moving_target) {
                // Next most likely is a leaf node, but not at the first depths.
                let v = random_leaf(random, depth, stats);
                *colour += v.average_colour;
                sum_count += 1.;

                GraphRef::Ref(Box::new(v))
            } else if random_type < 0.825_f32.powf(moving_target.powf(0.625)) {
                // Next most likely is a non-recursive voxel, however, should be less likely at latter depths.
                let v = roll_voxel_graph(random, depth + 1, stats);
                *colour += v.average_colour;
                sum_count += 1.;

                GraphRef::Ref(Box::new(v))
            } else {
                GraphRef::Recurse(random_recurse(random, depth))
            }
        };

        // Build a random voxel for each sub-voxel.
        let children = arr![pop_node_option(&mut sum_colour); 8];

        // Ensure that the sum count is never still 0.
        sum_count = sum_count.max(1.);

        stats.voxel_count += 1;
        Voxel {
            average_colour: sum_colour / sum_count,
            children,
            vtype: VoxelType::Complex,
            id: stats.voxel_count,
        }
    }

    loop {
        // Loop through random graphs until one satisfies all conditions.
        let mut stats = OctreeStats::default();
        let v = roll_voxel_graph(random, 0, &mut stats);

        // If we have generated enough voxels, compactify the octree into an array and return it.
        if stats.voxel_count >= desired_voxel_count && stats.goal_count >= desired_portal_count {
            return (compact_octree_from_root(v, stats.voxel_count), stats);
        }
    }
}

pub enum Intersection {
    Empty(f32),
    Collision,
    Portal(u32),
}

// Determine where in the octree a point is, and whether it is colliding with a voxel.
pub fn octree_scale_and_collision_of_point(
    position: Vector3<f32>,
    octree: &[VoxelCompact],
) -> Intersection {
    const GOAL_RADIUS_SQUARED: f32 = 0.75;
    fn search_graph(
        scale: f32,
        iter: u32,
        p: Vector3<f32>,
        index: u32,
        octree: &[VoxelCompact],
    ) -> Intersection {
        if iter > MAXIMUM_VOXEL_DEPTH {
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
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    Intersection::Portal((scale.log2() - 1.).max(0.) as u32)
                } else {
                    Intersection::Empty(scale)
                }
            } else {
                // The center of each sub-voxel (cell) relative to the parent.
                // The order here must be aligned with the subvoxel-order in the `ray_march_voxels.frag` shader.
                const CELL_CENTERS: [Vector3<f32>; 8] = [
                    Vector3::new(-0.5, 0.5, -0.5),
                    Vector3::new(0.5, 0.5, -0.5),
                    Vector3::new(-0.5, -0.5, -0.5),
                    Vector3::new(0.5, -0.5, -0.5),
                    Vector3::new(-0.5, 0.5, 0.5),
                    Vector3::new(0.5, 0.5, 0.5),
                    Vector3::new(-0.5, -0.5, 0.5),
                    Vector3::new(0.5, -0.5, 0.5),
                ];

                // Determine which sub-voxel the point is in by assigning a bit to each axis depending on which side of the axis the point is on.
                let cell_index = (usize::from(p.z > 0.) << 2)
                    + (usize::from(p.y <= 0.) << 1)
                    + usize::from(p.x > 0.);

                // Rescale the point to be relative to the sub-voxel and recurse.
                search_graph(
                    scale + scale,
                    iter + 1,
                    2. * (p - CELL_CENTERS[cell_index]),
                    voxel.children[cell_index],
                    octree,
                )
            }
        }
    }

    if position.x.abs() > 1. || position.y.abs() > 1. || position.z.abs() > 1. {
        // If the point is outside the root voxel then there cannot be an intersection.
        Intersection::Empty(1.)
    } else {
        search_graph(1., 0, position, 0, octree)
    }
}

impl RandomFloats {
    // Create a new instance for generating random values for voxel generation.
    pub fn new(seed: u64) -> Self {
        let leaf_before_portal_depth = fldr::Generator::new(&[9, 1]);
        let leaf_after_portal_depth = fldr::Generator::new(&[21, 2, 2]);

        let random = StdRng::seed_from_u64(seed);

        // TODO: Have the fair_coin make its RNG public so that we can make independent samples using the same RNG.
        let fair_coin = fldr::rand::RngCoin::<StdRng>::new(StdRng::seed_from_u64(seed + 1));

        Self {
            leaf_after_portal_depth,
            leaf_before_portal_depth,
            fair_coin,

            random,
            dist: Uniform::new(0., 1.),
        }
    }

    // Default to a simple time based seed and create instance.
    pub fn default() -> Self {
        Self::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        )
    }

    // Sample a random float uniformly between 0 and 1.
    pub fn samplef(&mut self) -> f32 {
        self.dist.sample(&mut self.random)
    }

    // Sample a random voxel type depending on the depth of the voxel.
    fn sample_leaf(&mut self, depth_reached: bool) -> VoxelType {
        if depth_reached {
            const LEAVES: [VoxelType; 3] =
                [VoxelType::Colour, VoxelType::Portal, VoxelType::Mirror];
            LEAVES[self.leaf_after_portal_depth.sample(&mut self.fair_coin)]
        } else {
            const LEAVES: [VoxelType; 2] = [VoxelType::Colour, VoxelType::Mirror];
            LEAVES[self.leaf_before_portal_depth.sample(&mut self.fair_coin)]
        }
    }
}
