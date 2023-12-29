/*
    voxel_flight_simulator - A simple game where you fly around randomly generated, recursive, voxel worlds.
    Copyright (C) 2023 Ryan Andersen

    voxel_flight_simulator is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    voxel_flight_simulator is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with voxel_flight_simulator. If not, see <https://www.gnu.org/licenses/>.
*/

use std::sync::Arc;

use smallvec::smallvec;
use vulkano::{
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
        RenderPassBeginInfo, SecondaryAutoCommandBuffer, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        PersistentDescriptorSet,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, SampleCount},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexInputState,
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};
use vulkano_util::renderer::VulkanoWindowRenderer;

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
    pub fn new(queue: Arc<Queue>, image_format: Format, viewport: Viewport) -> Self {
        let allocators = Allocators {
            memory: Arc::new(StandardMemoryAllocator::new_default(queue.device().clone())),
            command_buffer: StandardCommandBufferAllocator::new(
                queue.device().clone(),
                StandardCommandBufferAllocatorCreateInfo {
                    secondary_buffer_count: 16,
                    ..StandardCommandBufferAllocatorCreateInfo::default()
                },
            ),
            descriptor_set: StandardDescriptorSetAllocator::new(
                queue.device().clone(),
                StandardDescriptorSetAllocatorCreateInfo::default(),
            ),
        };

        let render_pass = RenderAppWithOverlay::new(queue, image_format, viewport);

        Engine {
            allocators,
            app_renderer: render_pass,
        }
    }

    pub fn render_frame(
        &mut self,
        renderer: &mut VulkanoWindowRenderer,
        gui_command_buffer: Option<Arc<SecondaryAutoCommandBuffer>>,
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

    // Get subpass for the GUI overlay.
    pub fn gui_pass(&self) -> Subpass {
        Subpass::from(self.render_pass().clone(), 1).unwrap()
    }

    // Recreate the graphics pipeline given a new viewport.
    pub fn recreate_pipeline(&mut self, viewport: Viewport) {
        self.app_renderer.app_pipeline = AppPipeline::new(
            self.app_renderer.queue.device(),
            self.app_renderer.app_pipeline.subpass.clone(),
            viewport,
        );
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
    pub queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pub app_pipeline: AppPipeline,
}

impl RenderAppWithOverlay {
    pub fn new(queue: Arc<Queue>, image_format: Format, viewport: Viewport) -> Self {
        let render_pass = Self::create_render_pass(queue.device().clone(), image_format);

        // Create a graphics pipeline for the app's render pass.
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let app_pipeline = AppPipeline::new(queue.device(), subpass, viewport);

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
                    format: format,
                    samples: SampleCount::Sample1,
                    load_op: Clear,
                    store_op: Store,
                }
            },
            passes: [
                // Main app pass.
                { color: [color], depth_stencil: {}, input: [] },

                // GUI pass.
                { color: [color], depth_stencil: {}, input: [] }
            ]
        )
        .unwrap()
    }

    pub fn render(
        &self,
        allocator: &StandardCommandBufferAllocator,
        before_future: Box<dyn GpuFuture>,
        image: Arc<ImageView>,
        gui_command_buffer: Option<Arc<SecondaryAutoCommandBuffer>>,
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

        // Create framebuffer. Only one attachment is needed for this app.
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
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..SubpassBeginInfo::default()
                },
            )
            .unwrap();

        // Create secondary command buffer to run main app pipeline
        let app_command_buffer =
            self.app_pipeline
                .draw(allocator, &self.queue, push_constants, descriptor_set);

        // Add app commands to primary command buffer and move to next subpass.
        builder.execute_commands(app_command_buffer).unwrap();
        builder
            .next_subpass(
                SubpassEndInfo::default(),
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..SubpassBeginInfo::default()
                },
            )
            .unwrap();

        // Add optional GUI command buffer to primary command buffer.
        if let Some(command_buffer) = gui_command_buffer {
            builder.execute_commands(command_buffer).unwrap();
        }

        // End render pass and execute primary command buffer.
        builder.end_render_pass(SubpassEndInfo::default()).unwrap();
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
    pub subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
}

impl AppPipeline {
    // Create a graphics pipeline for the main app render pass.
    pub fn new(device: &Arc<Device>, subpass: Subpass, viewport: Viewport) -> Self {
        // Setup relevant context for creating the pipeline from these shaders.
        let vs = entire_view_vs::load(device.clone())
            .expect("Failed to create shader module.")
            .entry_point("main")
            .unwrap();
        let fs = ray_march_voxels_fs::load(device.clone())
            .expect("Failed to create shader module.")
            .entry_point("main")
            .unwrap();
        let stages = smallvec![
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages,
                vertex_input_state: Some(VertexInputState::default()),

                // Indicate the type of the primitive drawn (the default is a list of triangles).
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleStrip,
                    ..InputAssemblyState::default()
                }),
                // Set a fixed viewport.
                viewport_state: Some(ViewportState {
                    viewports: smallvec![viewport],
                    ..ViewportState::default()
                }),

                // Necessary defaults.
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState {
                    attachments: (0..subpass.num_color_attachments())
                        .map(|_| ColorBlendAttachmentState::default())
                        .collect(),
                    ..Default::default()
                }),
                // Specify the subpass where this pipeline will be used.
                subpass: Some(subpass.clone().into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .expect("Failed to create graphics pipeline");

        AppPipeline { subpass, pipeline }
    }

    pub fn draw(
        &self,
        allocator: &StandardCommandBufferAllocator,
        queue: &Arc<Queue>,
        push_constants: ray_march_voxels_fs::Push,
        descriptor_set: Arc<PersistentDescriptorSet>,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        let mut builder = AutoCommandBufferBuilder::secondary(
            allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..CommandBufferInheritanceInfo::default()
            },
        )
        .unwrap();

        builder
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .draw(4, 1, 0, 0)
            .expect("Failed to complete draw command");

        builder.build().unwrap()
    }

    // Getters
    pub fn pipeline(&self) -> &Arc<GraphicsPipeline> {
        &self.pipeline
    }
}

/// Minimal vertex shader which draws a quad over the entire viewport.
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

/// Import the fragment shader by file path.
pub mod ray_march_voxels_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/ray_march_voxels.frag",
    }
}
