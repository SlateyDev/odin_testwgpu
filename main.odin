package test

import "base:runtime"
import "core:fmt"
import "core:math"
import la "core:math/linalg"
import "core:mem"
import "core:time"
import "vendor:wgpu"

Node3D :: struct {
	translation: la.Vector3f32,
	rotation:    la.Quaternionf32,
	scale:       la.Vector3f32,
}

State :: struct {
	ctx:                       runtime.Context,
	os:                        OS,
	instance:                  wgpu.Instance,
	surface:                   wgpu.Surface,
	adapter:                   wgpu.Adapter,
	device:                    wgpu.Device,
	config:                    wgpu.SurfaceConfiguration,
	queue:                     wgpu.Queue,
	storage_buffer:            wgpu.Buffer,
	storage_bind_group_layout: wgpu.BindGroupLayout,
	storage_bind_group:        wgpu.BindGroup,
}

shaders: map[string]wgpu.ShaderModule
pipelineLayouts: map[string]wgpu.PipelineLayout
pipelines: map[string]wgpu.RenderPipeline
meshes: map[string]Mesh

Mesh :: struct {
	materialResourceName: string,
	vertexBuffer:         wgpu.Buffer,
	vertBuffer:           [BUFFER_SIZE * 8]f32,
}

BUFFER_SIZE :: 16384

Vertex :: struct {
	position: [3]f32,
	uv:       [2]f32,
	color:    [4]f32,
	data:     [3]f32,
}

gameObject1 := Node3D {
	translation = {0, 0, 0},
	rotation    = la.quaternion_from_euler_angles_f32(0, 0, 0, la.Euler_Angle_Order.ZYX),
	scale       = {1, 1, 1},
}

modelMatrix := la.MATRIX4F32_IDENTITY
viewMatrix := la.MATRIX4F32_IDENTITY
projectionMatrix: la.Matrix4f32

start_time := time.now()

@(private = "file")
state: State

depthTexture : wgpu.Texture
depthTextureView : wgpu.TextureView

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

	wgpu.InstanceRequestAdapter(
		state.instance,
		&{compatibleSurface = state.surface},
		on_adapter,
		nil,
	)

	on_adapter :: proc "c" (
		status: wgpu.RequestAdapterStatus,
		adapter: wgpu.Adapter,
		message: cstring,
		userdata: rawptr,
	) {
		context = state.ctx
		if status != .Success || adapter == nil {
			fmt.panicf("request adapter failure: [%v] %s", status, message)
		}
		state.adapter = adapter
		wgpu.AdapterRequestDevice(adapter, nil, on_device)
	}

	on_device :: proc "c" (
		status: wgpu.RequestDeviceStatus,
		device: wgpu.Device,
		message: cstring,
		userdata: rawptr,
	) {
		context = state.ctx
		if status != .Success || device == nil {
			fmt.panicf("request device failure: [%v] %s", status, message)
		}
		state.device = device

		width, height := os_get_render_bounds(&state.os)

		state.config = wgpu.SurfaceConfiguration {
			device      = state.device,
			usage       = {.RenderAttachment},
			format      = .BGRA8Unorm,
			width       = width,
			height      = height,
			presentMode = .Fifo,
			alphaMode   = .Opaque,
		}
		wgpu.SurfaceConfigure(state.surface, &state.config)

		projectionMatrix = la.matrix4_perspective(
			2 * math.PI / 5,
			f32(width) / f32(height),
			1.0,
			100.0,
		)
		viewMatrix = la.matrix4_look_at(
			la.Vector3f32{0.0, 0.0, 4.0},
			la.Vector3f32{0.0, 0.0, 0.0},
			la.Vector3f32{0.0, 1.0, 0.0},
		)

		state.queue = wgpu.DeviceGetQueue(state.device)

		shader :: cstring(#load("shader.wgsl"))

		shaders["testShader"] = wgpu.DeviceCreateShaderModule(
			state.device,
			&{
				nextInChain = &wgpu.ShaderModuleWGSLDescriptor {
					sType = .ShaderModuleWGSLDescriptor,
					code = shader,
				},
			},
		)

		meshes["test"] = Mesh {
			vertexBuffer = wgpu.DeviceCreateBuffer(
				state.device,
				&wgpu.BufferDescriptor {
					label = "Vertex Buffer",
					usage = {.Vertex, .CopyDst},
					size = BUFFER_SIZE * 8 * size_of(f32),
				},
			),
		}

		mesh := &meshes["test"]
		copy(
			mesh.vertBuffer[(0 * size_of(Vertex) + offset_of(Vertex, position)) / size_of(f32):],
			[]f32{-1, -1, 0},
		)
		copy(
			mesh.vertBuffer[(0 * size_of(Vertex) + offset_of(Vertex, color)) / size_of(f32):],
			[]f32{1, 0, 0, 1},
		)

		copy(
			mesh.vertBuffer[(1 * size_of(Vertex) + offset_of(Vertex, position)) / size_of(f32):],
			[]f32{0, 1, 0},
		)
		copy(
			mesh.vertBuffer[(1 * size_of(Vertex) + offset_of(Vertex, color)) / size_of(f32):],
			[]f32{0, 1, 0, 1},
		)

		copy(
			mesh.vertBuffer[(2 * size_of(Vertex) + offset_of(Vertex, position)) / size_of(f32):],
			[]f32{1, -1, 0},
		)
		copy(
			mesh.vertBuffer[(2 * size_of(Vertex) + offset_of(Vertex, color)) / size_of(f32):],
			[]f32{0, 0, 1, 1},
		)

		wgpu.QueueWriteBuffer(
			state.queue,
			mesh.vertexBuffer,
			0,
			&mesh.vertBuffer,
			3 * size_of(Vertex),
		)

		state.storage_buffer = wgpu.DeviceCreateBuffer(
			state.device,
			&wgpu.BufferDescriptor {
				label = "Storage Buffer",
				usage = {.Storage, .CopyDst},
				size = size_of(matrix[4, 4]f32) * 2,
			},
		)
		defer wgpu.BufferRelease(state.storage_buffer)

		state.storage_bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
			state.device,
			&wgpu.BindGroupLayoutDescriptor {
				label = "Bind Group Layout",
				entryCount = 1,
				entries = raw_data(
					[]wgpu.BindGroupLayoutEntry {
						{
							binding = 0,
							visibility = {.Vertex},
							buffer = {type = .ReadOnlyStorage},
						},
					},
				),
			},
		)
		defer wgpu.BindGroupLayoutRelease(state.storage_bind_group_layout)

		state.storage_bind_group = wgpu.DeviceCreateBindGroup(
			state.device,
			&wgpu.BindGroupDescriptor {
				layout = state.storage_bind_group_layout,
				entryCount = 1,
				entries = raw_data(
					[]wgpu.BindGroupEntry {
						{
							binding = 0,
							buffer = state.storage_buffer,
							size = size_of(matrix[4, 4]f32) * 2,
						},
					},
				),
			},
		)
		defer wgpu.BindGroupRelease(state.storage_bind_group)

		pipelineLayouts["default"] = wgpu.DeviceCreatePipelineLayout(
			state.device,
			&{bindGroupLayoutCount = 1, bindGroupLayouts = &state.storage_bind_group_layout},
		)

		depthTexture = wgpu.DeviceCreateTexture(
			state.device,
			&wgpu.TextureDescriptor{
				size = wgpu.Extent3D{
					width = width,
					height = height,
					depthOrArrayLayers = 1,
				},
				format = .Depth24Plus,
				usage = {.RenderAttachment},
				dimension = ._2D,
				sampleCount = 1,
				mipLevelCount = 1,
				viewFormatCount = 0,
			},
		)

		depthTextureView = wgpu.TextureCreateView(depthTexture, &wgpu.TextureViewDescriptor{
			aspect = .DepthOnly,
			baseArrayLayer = 0,
			arrayLayerCount = 1,
			baseMipLevel = 0,
			mipLevelCount = 1,
			dimension = ._2D,
			format = .Depth24Plus,
		})
	
		pipelines["test"] = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&wgpu.RenderPipelineDescriptor{
				layout = pipelineLayouts["default"],
				vertex = {
					module = shaders["testShader"],
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
					module = shaders["testShader"],
					entryPoint = "fs_main",
					targetCount = 1,
					targets = &wgpu.ColorTargetState {
						format = .BGRA8Unorm,
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
				primitive = {topology = .TriangleList, cullMode = .None},
				multisample = {count = 1, mask = 0xFFFFFFFF},
				depthStencil = &wgpu.DepthStencilState{
					depthCompare = .Less,
					stencilReadMask = 0,
					stencilWriteMask = 0,
					depthWriteEnabled = true,
					format = .Depth24Plus,
					stencilFront = {
						compare = .Always,
						failOp = .Keep,
						depthFailOp = .Keep,
						passOp = .Keep,
					},
					stencilBack = {
						compare = .Always,
						failOp = .Keep,
						depthFailOp = .Keep,
						passOp = .Keep,
					},
				},
			},
		)

		init_game_state()
		defer destroy_game_state()

		display_game_state()

		os_run(&state.os)
	}
}

resize :: proc "c" () {
	context = state.ctx

	state.config.width, state.config.height = os_get_render_bounds(&state.os)

	projectionMatrix = la.matrix4_perspective(
		2 * math.PI / 5,
		f32(state.config.width) / f32(state.config.height),
		1.0,
		100.0,
	)

	wgpu.SurfaceConfigure(state.surface, &state.config)
}

//Used to scale and translate our scene from OpenGL's coordinate system to WGPU's
OPEN_GL_TO_WGPU_MATRIX :: la.Matrix4f32 {
	1.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 0.5, 0.5,
	0.0, 0.0, 0.0, 1.0,
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

	frame := wgpu.TextureCreateView(surface_texture.texture, &{
		format = .BGRA8Unorm,
		baseMipLevel = 0,
		mipLevelCount = 1,
		baseArrayLayer = 0,
		arrayLayerCount = 1,
		aspect = .All,
		dimension = ._2D,
	})
	defer wgpu.TextureViewRelease(frame)

	command_encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)
	defer wgpu.CommandEncoderRelease(command_encoder)

	render_pass_encoder := wgpu.CommandEncoderBeginRenderPass(
		command_encoder,
		&wgpu.RenderPassDescriptor{
			colorAttachmentCount = 1,
			colorAttachments = &wgpu.RenderPassColorAttachment {
				view = frame,
				loadOp = .Clear,
				storeOp = .Store,
				// depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
				clearValue = {0.2, 0.2, 0.2, 1},
			},
			depthStencilAttachment = &{
				view = depthTextureView,
				depthClearValue = 1.0,
				depthLoadOp = .Clear,
				depthStoreOp = .Store,
				depthReadOnly = false,
				stencilClearValue = 0,
				stencilLoadOp = .Clear,
				stencilStoreOp = .Store,
				stencilReadOnly = true,
			},
		},
	)

	mesh := &meshes["test"]
	wgpu.RenderPassEncoderSetPipeline(render_pass_encoder, pipelines["test"])
	wgpu.RenderPassEncoderSetBindGroup(render_pass_encoder, 0, state.storage_bind_group)
	wgpu.RenderPassEncoderSetVertexBuffer(
		render_pass_encoder,
		0,
		mesh.vertexBuffer,
		0,
		3 * size_of(Vertex),
	)

	now := f32(time.duration_seconds(time.since(start_time)))
	gameObject1.rotation = la.quaternion_from_euler_angles_f32(
		0,
		math.sin(now) * 1.2,
		0,
		la.Euler_Angle_Order.XYZ,
	)
	modelMatrix = la.matrix4_from_trs_f32(
		gameObject1.translation,
		gameObject1.rotation,
		gameObject1.scale,
	)

	transform := OPEN_GL_TO_WGPU_MATRIX * projectionMatrix * viewMatrix * modelMatrix
	wgpu.QueueWriteBuffer(state.queue, state.storage_buffer, 0, &transform, size_of(transform))

	wgpu.RenderPassEncoderDraw(
		render_pass_encoder,
		vertexCount = 3,
		instanceCount = 1,
		firstVertex = 0,
		firstInstance = 0,
	)

	wgpu.RenderPassEncoderSetVertexBuffer(
		render_pass_encoder,
		0,
		mesh.vertexBuffer,
		0,
		3 * size_of(Vertex),
	)

	gameObject1.rotation = la.quaternion_from_euler_angles_f32(
		math.sin(now) * 1.2,
		0,
		0,
		la.Euler_Angle_Order.XYZ,
	)
	modelMatrix = la.matrix4_from_trs_f32(
		gameObject1.translation,
		gameObject1.rotation,
		gameObject1.scale,
	)

	transform = OPEN_GL_TO_WGPU_MATRIX * projectionMatrix * viewMatrix * modelMatrix
	wgpu.QueueWriteBuffer(state.queue, state.storage_buffer, size_of(matrix[4, 4]f32), &transform, size_of(transform))

	wgpu.RenderPassEncoderDraw(
		render_pass_encoder,
		vertexCount = 3,
		instanceCount = 1,
		firstVertex = 0,
		firstInstance = 1,
	)

	wgpu.RenderPassEncoderEnd(render_pass_encoder)
	wgpu.RenderPassEncoderRelease(render_pass_encoder)

	command_buffer := wgpu.CommandEncoderFinish(command_encoder, nil)
	defer wgpu.CommandBufferRelease(command_buffer)

	wgpu.QueueSubmit(state.queue, {command_buffer})
	wgpu.SurfacePresent(state.surface)
}

finish :: proc() {
	cleanup_meshes()
	cleanup_pipelines()
	wgpu.TextureViewRelease(depthTextureView)
	wgpu.TextureRelease(depthTexture)
	cleanup_pipeline_layouts()
	cleanup_shaders()
	wgpu.QueueRelease(state.queue)
	wgpu.DeviceRelease(state.device)
	wgpu.AdapterRelease(state.adapter)
	wgpu.SurfaceRelease(state.surface)
	wgpu.InstanceRelease(state.instance)
}

cleanup_meshes :: proc() {
	for key in meshes {
		mesh := &meshes[key]
		wgpu.BufferRelease(mesh.vertexBuffer)
	}
	delete_map(meshes)
}

cleanup_pipelines :: proc() {
	for _, pipeline in pipelines {
		wgpu.RenderPipelineRelease(pipeline)
	}
	delete_map(pipelines)
}

cleanup_pipeline_layouts :: proc() {
	for _, pipelineLayout in pipelineLayouts {
		wgpu.PipelineLayoutRelease(pipelineLayout)
	}
	delete_map(pipelineLayouts)
}

cleanup_shaders :: proc() {
	for _, shader in shaders {
		wgpu.ShaderModuleRelease(shader)
	}
	delete_map(shaders)
}
