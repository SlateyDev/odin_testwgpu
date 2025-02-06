package test

import "base:runtime"
import "core:fmt"
import "core:mem"
import "vendor:wgpu"

Vector2 :: distinct [2]f32
Vector3 :: distinct [3]f32
Quaternion :: distinct quaternion128

Node2D :: struct {
    position : Vector2,
    orientation : Quaternion,
}
Node3D :: struct {
    position : Vector3,
    orientation : f32,
}

State :: struct {
	ctx: runtime.Context,
	os:  OS,

	instance:        wgpu.Instance,
	surface:         wgpu.Surface,
	adapter:         wgpu.Adapter,
	device:          wgpu.Device,
	config:          wgpu.SurfaceConfiguration,
	queue:           wgpu.Queue,
	module:          wgpu.ShaderModule,
	pipeline_layout: wgpu.PipelineLayout,
	pipeline:        wgpu.RenderPipeline,
}

Material :: struct {
	pipeline: wgpu.RenderPipeline,
}

BUFFER_SIZE :: 16384

Mesh :: struct {
	material: Material,
	vertexBuffer: wgpu.Buffer,
	vertBuffer: [BUFFER_SIZE * 8]f32,
}

Vertex :: struct {
	position: [3]f32,
	uv:       [2]f32,
	color:    [4]f32,
	data:     [3]f32,
}

mesh : Mesh
material : Material

@(private="file")
state: State

main :: proc() {
    when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)

		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not freed: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			if len(track.bad_free_array) > 0 {
				fmt.eprintf("=== %v incorrect frees: ===\n", len(track.bad_free_array))
				for entry in track.bad_free_array {
					fmt.eprintf("- %p @ %v\n", entry.memory, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	} else {
		_ :: mem
	}

	game()
}

game :: proc() {
	state.ctx = context

	os_init(&state.os)

	state.instance = wgpu.CreateInstance(nil)
	if state.instance == nil {
		panic("WebGPU is not supported")
	}
	state.surface = os_get_surface(&state.os, state.instance)

	wgpu.InstanceRequestAdapter(state.instance, &{ compatibleSurface = state.surface }, on_adapter, nil)

	on_adapter :: proc "c" (status: wgpu.RequestAdapterStatus, adapter: wgpu.Adapter, message: cstring, userdata: rawptr) {
		context = state.ctx
		if status != .Success || adapter == nil {
			fmt.panicf("request adapter failure: [%v] %s", status, message)
		}
		state.adapter = adapter
		wgpu.AdapterRequestDevice(adapter, nil, on_device)
	}

	on_device :: proc "c" (status: wgpu.RequestDeviceStatus, device: wgpu.Device, message: cstring, userdata: rawptr) {
		context = state.ctx
		if status != .Success || device == nil {
			fmt.panicf("request device failure: [%v] %s", status, message)
		}
		state.device = device 

		width, height := os_get_render_bounds(&state.os)

		state.config = wgpu.SurfaceConfiguration {
			device      = state.device,
			usage       = { .RenderAttachment },
			format      = .BGRA8Unorm,
			width       = width,
			height      = height,
			presentMode = .Fifo,
			alphaMode   = .Opaque,
		}
		wgpu.SurfaceConfigure(state.surface, &state.config)

		state.queue = wgpu.DeviceGetQueue(state.device)

		shader :: `
			struct VertexOutput {
				@builtin(position) position_clip: vec4<f32>,
				@location(0) position: vec3<f32>,
				@location(1) uv: vec2<f32>,
				@location(2) color: vec4<f32>,
			};

			@vertex
			fn vs_main(
				@builtin(vertex_index) in_vertex_index: u32,
				@location(0) pos: vec3<f32>,
				@location(1) uv: vec2<f32>,
				@location(2) color: vec4<f32>,
			) -> VertexOutput {
				var output: VertexOutput;
				output.position_clip = vec4<f32>(pos, 1);
				output.position = pos;
				output.uv = uv;
				output.color = color;
				return output;
			}

			@fragment
			fn fs_main(
				@location(0) position: vec3<f32>,
				@location(1) uv: vec2<f32>,
				@location(2) color: vec4<f32>,
			) -> @location(0) vec4<f32> {
				return color;
			}`

		state.module = wgpu.DeviceCreateShaderModule(state.device, &{
			nextInChain = &wgpu.ShaderModuleWGSLDescriptor{
				sType = .ShaderModuleWGSLDescriptor,
				code  = shader,
			},
		})

		mesh.vertexBuffer = wgpu.DeviceCreateBuffer(state.device, &wgpu.BufferDescriptor{
			label = "Vertex Buffer",
			usage = {.Vertex, .CopyDst},
			size = size_of(mesh.vertBuffer),
		})

		copy(mesh.vertBuffer[(0 * size_of(Vertex) + offset_of(Vertex, position)) / size_of(f32):], []f32{-1,-1,0})
		copy(mesh.vertBuffer[(0 * size_of(Vertex) + offset_of(Vertex, color)) / size_of(f32):], []f32{1,0,0,1})

		copy(mesh.vertBuffer[(1 * size_of(Vertex) + offset_of(Vertex, position)) / size_of(f32):], []f32{0,1,0})
		copy(mesh.vertBuffer[(1 * size_of(Vertex) + offset_of(Vertex, color)) / size_of(f32):], []f32{0,1,0,1})

		copy(mesh.vertBuffer[(2 * size_of(Vertex) + offset_of(Vertex, position)) / size_of(f32):], []f32{1,-1,0})
		copy(mesh.vertBuffer[(2 * size_of(Vertex) + offset_of(Vertex, color)) / size_of(f32):], []f32{0,0,1,1})

		wgpu.QueueWriteBuffer(state.queue, mesh.vertexBuffer, 0, &mesh.vertBuffer, 3 * size_of(Vertex))

		state.pipeline_layout = wgpu.DeviceCreatePipelineLayout(state.device, &{})
		state.pipeline = wgpu.DeviceCreateRenderPipeline(state.device, &{
			layout = state.pipeline_layout,
			vertex = {
				module     = state.module,
				entryPoint = "vs_main",
				bufferCount = 1,
				buffers = raw_data(
					[]wgpu.VertexBufferLayout {
						{
							arrayStride = size_of(Vertex),
							stepMode = .Vertex,
							attributeCount = 4,
							attributes = raw_data(
								[]wgpu.VertexAttribute {
									{
										format = .Float32x3,
										offset = u64(offset_of(Vertex, position)),
										shaderLocation = 0,
									},
									{
										format = .Float32x2,
										offset = u64(offset_of(Vertex, uv)),
										shaderLocation = 1,
									},
									{
										format = .Float32x4,
										offset = u64(offset_of(Vertex, color)),
										shaderLocation = 2,
									},
									{
										format = .Float32x3,
										offset = u64(offset_of(Vertex, data)),
										shaderLocation = 3,
									},
								},
							),
						},
					},
				),
			},
			fragment = &{
				module      = state.module,
				entryPoint  = "fs_main",
				targetCount = 1,
				targets     = &wgpu.ColorTargetState{
					format    = .BGRA8Unorm,
					blend = &{
						alpha = {
							srcFactor = .SrcAlpha,
							dstFactor = .OneMinusSrcAlpha,
							operation = .Add,
						},
						color = {
							srcFactor = .SrcAlpha,
							dstFactor = .OneMinusSrcAlpha,
							operation = .Add,
						},
					},
					writeMask = wgpu.ColorWriteMaskFlags_All,
				},
			},
			primitive = {
				topology = .TriangleList,
				cullMode = .None,
			},
			multisample = {
				count = 1,
				mask  = 0xFFFFFFFF,
			},
		})

		os_run(&state.os)
	}

    init_game_state()
    defer destroy_game_state()

    display_game_state()
}

resize :: proc "c" () {
	context = state.ctx
	
	state.config.width, state.config.height = os_get_render_bounds(&state.os)
	wgpu.SurfaceConfigure(state.surface, &state.config)
}

frame :: proc "c" (dt: f32) {
	context = state.ctx

	surface_texture := wgpu.SurfaceGetCurrentTexture(state.surface)
	switch surface_texture.status {
	case .Success:
		// All good, could check for `surface_texture.suboptimal` here.
	case .Timeout, .Outdated, .Lost:
		// Skip this frame, and re-configure surface.
		if surface_texture.texture != nil {
			wgpu.TextureRelease(surface_texture.texture)
		}
		resize()
		return
	case .OutOfMemory, .DeviceLost:
		// Fatal error
		fmt.panicf("[triangle] get_current_texture status=%v", surface_texture.status)
	}
	defer wgpu.TextureRelease(surface_texture.texture)

	frame := wgpu.TextureCreateView(surface_texture.texture, nil)
	defer wgpu.TextureViewRelease(frame)

	command_encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)
	defer wgpu.CommandEncoderRelease(command_encoder)

	render_pass_encoder := wgpu.CommandEncoderBeginRenderPass(
		command_encoder, &{
			colorAttachmentCount = 1,
			colorAttachments     = &wgpu.RenderPassColorAttachment{
				view       = frame,
				loadOp     = .Clear,
				storeOp    = .Store,
				depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
				clearValue = { 0.2, 0.2, 0.2, 1 },
			},
		},
	)

	wgpu.RenderPassEncoderSetPipeline(render_pass_encoder, state.pipeline)
	wgpu.RenderPassEncoderSetVertexBuffer(render_pass_encoder, 0, mesh.vertexBuffer, 0, 3 * size_of(Vertex))
	wgpu.RenderPassEncoderDraw(render_pass_encoder, vertexCount=3, instanceCount=1, firstVertex=0, firstInstance=0)

	wgpu.RenderPassEncoderEnd(render_pass_encoder)
	wgpu.RenderPassEncoderRelease(render_pass_encoder)

	command_buffer := wgpu.CommandEncoderFinish(command_encoder, nil)
	defer wgpu.CommandBufferRelease(command_buffer)

	wgpu.QueueSubmit(state.queue, { command_buffer })
	wgpu.SurfacePresent(state.surface)
}

finish :: proc() {
	wgpu.BufferRelease(mesh.vertexBuffer)
	wgpu.RenderPipelineRelease(state.pipeline)
	wgpu.PipelineLayoutRelease(state.pipeline_layout)
	wgpu.ShaderModuleRelease(state.module)
	wgpu.QueueRelease(state.queue)
	wgpu.DeviceRelease(state.device)
	wgpu.AdapterRelease(state.adapter)
	wgpu.SurfaceRelease(state.surface)
	wgpu.InstanceRelease(state.instance)
}