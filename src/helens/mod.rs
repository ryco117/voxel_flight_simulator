use std::sync::Arc;

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferUsage, RenderPassBeginInfo,
        SecondaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet},
    device::{Device, Queue},
    format::Format,
    image::{ImageAccess, SampleCount},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};
use vulkano_util::renderer::{SwapchainImageView, VulkanoWindowRenderer};

pub struct Allocators {
    pub memory: Arc<StandardMemoryAllocator>,
    pub command_buffer: StandardCommandBufferAllocator,
    pub descriptor_set: StandardDescriptorSetAllocator,
}

pub struct Engine {
    allocators: Allocators,
    app_renderer: RenderAppWithOverlay,
}

impl Engine {
    pub fn new(queue: &Arc<Queue>, image_format: Format) -> Self {
        let allocators = Allocators {
            memory: Arc::new(StandardMemoryAllocator::new_default(queue.device().clone())),
            command_buffer: StandardCommandBufferAllocator::new(
                queue.device().clone(),
                Default::default(),
            ),
            descriptor_set: StandardDescriptorSetAllocator::new(queue.device().clone()),
        };

        let render_pass = RenderAppWithOverlay::new(queue.clone(), image_format);

        Engine {
            allocators,
            app_renderer: render_pass,
        }
    }

    pub fn render_frame(
        &mut self,
        renderer: &mut VulkanoWindowRenderer,
        gui_command_buffer: Option<SecondaryAutoCommandBuffer>,
        push_constants: ray_march_voxels_fs::Push,
        descriptor_set: Arc<PersistentDescriptorSet>,
    ) -> Box<dyn GpuFuture> {
        // Acquire swapchain future.
        let before_future = renderer.acquire().unwrap();

        self.app_renderer.render(
            &self.allocators.command_buffer,
            before_future,
            renderer.swapchain_image_view(),
            gui_command_buffer,
            push_constants,
            descriptor_set,
        )
    }

    pub fn gui_pass(&self) -> Subpass {
        Subpass::from(self.render_pass().clone(), 1).unwrap()
    }

    // Getters
    pub fn allocators(&self) -> &Allocators {
        &self.allocators
    }
    pub fn pipeline(&self) -> &Arc<GraphicsPipeline> {
        self.app_renderer.app_pipeline.pipeline()
    }
    pub fn render_pass(&self) -> &Arc<RenderPass> {
        self.app_renderer.render_pass()
    }
}

/// A render pass which places an incoming image over frame filling it.
struct RenderAppWithOverlay {
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    app_pipeline: AppPipeline,
}

impl RenderAppWithOverlay {
    pub fn new(queue: Arc<Queue>, image_format: Format) -> Self {
        let render_pass = Self::create_render_pass(queue.device().clone(), image_format);

        // Create pipeline for the main app's render pass
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let app_pipeline = AppPipeline::new(queue.clone(), subpass);

        RenderAppWithOverlay {
            queue,
            render_pass,
            app_pipeline,
        }
    }

    fn create_render_pass(device: Arc<Device>, format: Format) -> Arc<RenderPass> {
        vulkano::ordered_passes_renderpass!(
            device,
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: format,
                    samples: SampleCount::Sample1,
                }
            },
            passes: [
                // Main app pass
                { color: [color], depth_stencil: {}, input: [] },

                // GUI pass
                { color: [color], depth_stencil: {}, input: [] }
            ]
        )
        .unwrap()
    }

    pub fn render(
        &self,
        allocator: &StandardCommandBufferAllocator,
        before_future: Box<dyn GpuFuture>,
        image: SwapchainImageView,
        gui_command_buffer: Option<SecondaryAutoCommandBuffer>,
        push_constants: ray_march_voxels_fs::Push,
        descriptor_set: Arc<PersistentDescriptorSet>,
    ) -> Box<dyn GpuFuture> {
        // Create a primary command buffer builder with intent for one-time submission.
        let mut builder = AutoCommandBufferBuilder::primary(
            allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Create framebuffer (Note: attachments must be in same order as render pass creation).
        let dimensions = image.image().dimensions().width_height();
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![image],
                ..FramebufferCreateInfo::default()
            },
        )
        .unwrap();

        // Begin render pass.
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0; 4].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassContents::SecondaryCommandBuffers,
            )
            .unwrap();

        // Create secondary command buffer to run main app pipeline
        let app_command_buffer =
            self.app_pipeline
                .draw(allocator, dimensions, push_constants, descriptor_set);

        // Add app commands to primary command buffer and move to next subpass.
        builder.execute_commands(app_command_buffer).unwrap();
        builder
            .next_subpass(SubpassContents::SecondaryCommandBuffers)
            .unwrap();

        // Add optional GUI command buffer to primary command buffer.
        if let Some(command_buffer) = gui_command_buffer {
            builder.execute_commands(command_buffer).unwrap();
        }

        // End render pass and execute primary command buffer.
        builder.end_render_pass().unwrap();
        let command_buffer = builder.build().unwrap();
        let after_future = before_future
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap();

        after_future.boxed()
    }

    // Getters
    pub fn render_pass(&self) -> &Arc<RenderPass> {
        &self.render_pass
    }
}

struct AppPipeline {
    queue: Arc<Queue>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
}

impl AppPipeline {
    pub fn new(queue: Arc<Queue>, subpass: Subpass) -> Self {
        let device = queue.device();
        let vs = entire_view_vs::load(device.clone()).expect("failed to create shader module");
        let fs = ray_march_voxels_fs::load(device.clone()).expect("failed to create shader module");

        let pipeline = GraphicsPipeline::start()
            // A Vulkan shader may contain multiple entry points, so we specify which one.
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            // Indicate the type of the primitives (the default is a list of triangles).
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip),
            )
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            // Set a dynamic viewport.
            // TODO: Consider making this fixed since resizing is rare and requires recreating the swapchain anyway.
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .render_pass(subpass.clone())
            .build(device.clone())
            .unwrap();

        AppPipeline {
            queue,
            subpass,
            pipeline,
        }
    }

    pub fn draw(
        &self,
        allocator: &StandardCommandBufferAllocator,
        dimensions: [u32; 2],
        push_constants: ray_march_voxels_fs::Push,
        descriptor_set: Arc<PersistentDescriptorSet>,
    ) -> SecondaryAutoCommandBuffer {
        let mut builder = AutoCommandBufferBuilder::secondary(
            allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..CommandBufferInheritanceInfo::default()
            },
        )
        .unwrap();

        builder
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .bind_pipeline_graphics(self.pipeline.clone())
            .set_viewport(
                0,
                vec![Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                }],
            )
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .draw(4, 1, 0, 0)
            .unwrap();

        builder.build().unwrap()
    }

    // Getters
    pub fn pipeline(&self) -> &Arc<GraphicsPipeline> {
        &self.pipeline
    }
}

mod entire_view_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
layout (location = 0) out vec2 coord;
vec2 quad[4] = vec2[] (
        vec2(-1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0)
);
void main() {
        gl_Position = vec4(quad[gl_VertexIndex], 0.0, 1.0);
        coord = quad[gl_VertexIndex];
}",
    }
}

pub mod ray_march_voxels_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/ray_march_voxels.frag",
    }
}
