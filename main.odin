package test

import "base:runtime"
import "core:fmt"
import "core:image"
import png "core:image/png"
import "core:log"
import "core:math"
import la "core:math/linalg"
import "core:mem"
import "core:strings"
import "core:time"
import "vendor:cgltf"
import mu   "vendor:microui"
import "vendor:wgpu"

_ :: png
_ :: cgltf
_ :: strings

WGPU_LOGGING :: false

Node3D :: struct {
	translation: la.Vector3f32,
	rotation:    la.Quaternionf32,
	scale:       la.Vector3f32,
}

MeshInstance :: struct {
	using node3d : Node3D,
	uniform_buffer: wgpu.Buffer,
	uniform_bind_group: wgpu.BindGroup,
	mesh: string,
}

LightUniform :: struct {
	view_proj: la.Matrix4x4f32,
	position: la.Vector4f32,
}

CameraUniform :: struct {
	view_proj: la.Matrix4x4f32,
	position: la.Vector4f32,
}

State :: struct {
	ctx:                       runtime.Context,
	os:                        OS,
	instance:                  wgpu.Instance,
	surface:                   wgpu.Surface,
	adapter:                   wgpu.Adapter,
	device:                    wgpu.Device,
	config:                        wgpu.SurfaceConfiguration,
	queue:                     wgpu.Queue,
	camera_uniform_buffer:      wgpu.Buffer,
	mesh_bind_group_layout:    wgpu.BindGroupLayout,
	light_uniform_buffer:      wgpu.Buffer,
	scene_bind_group_layout:   wgpu.BindGroupLayout,
	scene_bind_group:          wgpu.BindGroup,
}

muState := struct {
	log_buf:         [1<<16]byte,
	log_buf_len:     int,
	log_buf_updated: bool,
	bg:              mu.Color,
}{
	bg = {90, 95, 100, 255},
}

shaders: map[string]wgpu.ShaderModule
pipelineLayouts: map[string]wgpu.PipelineLayout
pipelines: map[string]wgpu.RenderPipeline
meshes: map[string]Mesh

directionalLightPosition : [4]f32 = {50, 100, -100, 1}
directionalLightViewMatrix := la.matrix4_look_at_f32(directionalLightPosition.xyz, 0, la.VECTOR3F32_Y_AXIS)
directionalLightProjectionMatrix := la.matrix_ortho3d_f32(-80, 80, -80, 80, -200, 300)
directionalLightViewProjMatrix := la.matrix_mul(
	directionalLightProjectionMatrix,
	directionalLightViewMatrix,
)

directional_light := LightUniform {
	view_proj = directionalLightViewProjMatrix,
	position = directionalLightPosition,
}

objects: [dynamic]^MeshInstance

Mesh :: struct {
	materialResourceName: string,
	vertexBuffer:         wgpu.Buffer,
	vertices: uint,
	indexBuffer:		  wgpu.Buffer,
	indices: uint,
}

BUFFER_SIZE :: 16384

Vertex :: struct {
	position:   [3]f32,
	tex_coords: [2]f32,
	normal:     [3]f32,
}

Camera :: struct {
	position:	la.Vector3f32,
	rotation:	la.Vector3f32,
	up:			la.Vector3f32,
	fov: f32,
	near: f32,
	far: f32,
}

FlyCamera :: struct {
	using camera : Camera,
	pitch: f32,
	yaw: f32,
}

flyCamera := FlyCamera {
	Camera {
		la.Vector3f32{0.0, 0.0, -4.0},
		la.Vector3f32{0.0, 0.0, 1.0},
		la.Vector3f32{0.0, -1.0, 0.0},
		72 * la.RAD_PER_DEG,
		0.01,
		10.0,
	},
	0.0,
	0.0,
}

camera_move_forward :: proc(camera : ^FlyCamera, delta: f32) {
	camera.position += delta * camera.rotation
}

camera_move_right :: proc(camera : ^FlyCamera, delta: f32) {
	camera.position += delta * la.normalize(la.vector_cross(camera.rotation, la.VECTOR3F32_Y_AXIS))
}

camera_adjust_pitch :: proc(camera : ^FlyCamera, delta : f32) {
   //Clamp to 90 and -90
   camera.pitch = math.max(-89.0 * la.RAD_PER_DEG, math.min(89.0 * la.RAD_PER_DEG, camera.pitch + delta * la.RAD_PER_DEG))
   camera_update_direction(camera)
}

camera_adjust_yaw :: proc(camera : ^FlyCamera, delta : f32) {
   camera.yaw += delta * la.RAD_PER_DEG
   camera_update_direction(camera)
}

camera_update_direction :: proc(camera : ^FlyCamera) {
	xzLen := la.cos(camera.pitch)
	camera.rotation.x = xzLen * la.cos(camera.yaw + la.PI / 2)
	camera.rotation.y = la.sin(camera.pitch)
	camera.rotation.z = xzLen * la.sin(camera.yaw + la.PI / 2)
}

gameObject1 := MeshInstance {
	translation = {0, 0, 0},
	rotation    = la.quaternion_from_euler_angles_f32(0, 0, 0, la.Euler_Angle_Order.ZYX),
	scale       = {1, 1, 1},
	mesh = "cube",
}

when ODIN_OS != .JS {
	gameObject2 := MeshInstance {
		translation = {2, -2, 0},
		rotation    = la.quaternion_from_euler_angles_f32(0, 0, 0, la.Euler_Angle_Order.ZYX),
		scale       = {10, 10, 10},
		mesh = "duck",
	}
} else {
	gameObject2 := MeshInstance {
		translation = {2, -2, 0},
		rotation    = la.quaternion_from_euler_angles_f32(0, 0, 0, la.Euler_Angle_Order.ZYX),
		scale       = {1, 1, 1},
		mesh = "cube",
	}	
}

gameObject3 := MeshInstance {
	translation = {0, -4, 0},
	rotation = la.quaternion_from_euler_angles_f32(0, 0, -0.5*la.PI, la.Euler_Angle_Order.ZYX),
	scale = {10, 10, 10},
	mesh = "plane",
}

modelMatrix := la.MATRIX4F32_IDENTITY
viewMatrix := la.MATRIX4F32_IDENTITY
projectionMatrix: la.Matrix4f32

start_time := time.now()

// @(private = "file")
state: State

depthTexture : wgpu.Texture
depthTextureView : wgpu.TextureView

uvTexture : wgpu.Texture
uvTextureView : wgpu.TextureView
uvTextureSampler : wgpu.Sampler
samplerBindGroupLayout : wgpu.BindGroupLayout
samplerBindGroup : wgpu.BindGroup

shadowSamplerBindGroupLayout : wgpu.BindGroupLayout
shadowSamplerBindGroup : wgpu.BindGroup

DEPTH_FORMAT :: wgpu.TextureFormat.Depth32Float

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

	context.logger = log.create_console_logger()
	defer log.destroy_console_logger(context.logger)

	game()
}

createMeshFromData :: proc(vertex_data : ^[]Vertex, index_data : ^[]u32) -> Mesh {
	new_mesh := Mesh {
		vertices = len(vertex_data^),
		indices = index_data != nil ? len(index_data^) : 0,
	}
	new_mesh.vertexBuffer = wgpu.DeviceCreateBufferWithData(
		state.device,
		&wgpu.BufferWithDataDescriptor {
			label = "Vertex Buffer",
			usage = {.Vertex},
		},
		vertex_data^,
	)
	if new_mesh.indices > 0 {
		new_mesh.indexBuffer = wgpu.DeviceCreateBufferWithData(
			state.device,
			&wgpu.BufferWithDataDescriptor {
				label = "Index Buffer",
				usage = {.Index},
			},
			index_data^,
		)
	}
	return new_mesh
}

game :: proc() {
	state.ctx = context

	os_init()

	when WGPU_LOGGING && ODIN_OS != .JS {
		wgpu.SetLogLevel(wgpu.LogLevel.Debug)
		wgpu.SetLogCallback(proc "c" (wgpulevel: wgpu.LogLevel, message: string, user: rawptr) {
			context = state.ctx
			logger := context.logger
			if logger.procedure == nil {
				return
			}

			level := wgpu.ConvertLogLevel(wgpulevel)
			if level < logger.lowest_level {
				return
			}

			smessage := strings.concatenate({"[nais][wgpu]: ", string(message)}, context.temp_allocator)
			logger.procedure(logger.data, level, smessage, logger.options, {})
		}, nil)
	}

	state.instance = wgpu.CreateInstance(nil/*&wgpu.InstanceDescriptor{nextInChain = &wgpu.InstanceExtras{sType = .InstanceExtras, backends = {.Vulkan}, flags = wgpu.InstanceFlags_Default}}*/)
	if state.instance == nil {
		panic("WebGPU is not supported")
	}
	state.surface = os_get_surface(state.instance)

	wgpu.InstanceRequestAdapter(
		state.instance,
		&{compatibleSurface = state.surface,/* powerPreference = wgpu.PowerPreference.HighPerformance*/},
		{callback = on_adapter},
	)

	on_adapter :: proc "c" (
		status: wgpu.RequestAdapterStatus,
		adapter: wgpu.Adapter,
		message: string,
		userdata: rawptr,
		userdata2: rawptr,
	) {
		context = state.ctx
		if status != .Success || adapter == nil {
			fmt.panicf("request adapter failure: [%v] %s", status, message)
		}
		state.adapter = adapter
		wgpu.AdapterRequestDevice(adapter, nil, {callback =  on_device})
	}

	on_device :: proc "c" (
		status: wgpu.RequestDeviceStatus,
		device: wgpu.Device,
		message: string,
		userdata: rawptr,
		userdata2: rawptr,
	) {
		context = state.ctx
		if status != .Success || device == nil {
			fmt.panicf("request device failure: [%v] %s", status, message)
		}
		state.device = device

		width, height := os_get_render_bounds()

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
			flyCamera.fov,
			f32(width) / f32(height),
			1.0,
			100.0,
		)
		viewMatrix = la.matrix4_look_at(
			la.Vector3f32{0.0, 0.0, 4.0},
			la.Vector3f32{0.0, 0.0, 0.0},
			la.Vector3f32{0.0, 1.0, 0.0},
		)

		append(
			&objects,
			&gameObject1,
			&gameObject2,
			&gameObject3,
		)

		state.queue = wgpu.DeviceGetQueue(state.device)

		shaders["testShader"] = wgpu.DeviceCreateShaderModule(
			state.device,
			&{
				nextInChain = &wgpu.ShaderSourceWGSL {
					sType = .ShaderSourceWGSL,
					code = string(#load("shaders/shader.wgsl")),
				},
			},
		)

		shaders["shadowCaster"] = wgpu.DeviceCreateShaderModule(
			state.device,
			&{
				nextInChain = &wgpu.ShaderSourceWGSL{
					sType = .ShaderSourceWGSL,
					code = string(#load("shaders/shadow_caster.wgsl")),
				},
			},
		)

		meshes["cube"] = createMeshFromData(&cube_vertex_data, &cube_index_data)

		meshes["triangle"] = createMeshFromData(&triangle_vertex_data, nil)

		meshes["plane"] = createMeshFromData(&plane_vertex_data, &plane_index_data)

		//currently only supporting - position: vec3, texcoord: vec2, color: vec4 (optional), with an index buffer
		when ODIN_OS != .JS {
			//TODO: Move into model loader. Have extra scoping here for defers. When above doesn't do scoping also.
			{
				cgltf_options : cgltf.options
				data, result := cgltf.parse_file(cgltf_options, "./assets/rubber_duck_toy_1k.gltf")
				if result != .success {
					return
				}
				defer cgltf.free(data)

				buffers_result := cgltf.load_buffers(cgltf_options, data, "./assets/rubber_duck_toy_1k.gltf")
				if buffers_result != .success {
					return
				}

				if len(data.meshes) == 0 do return
				mesh_data := &data.meshes[0]
				if len(mesh_data.primitives) == 0 do return
				mesh_primitive := &mesh_data.primitives[0]

				verts : Maybe(uint) = nil

				vert_data : []Vertex
				defer delete(vert_data)

				// hasColor := false

				for attr in mesh_primitive.attributes {
					#partial switch attr.type {
					case .position:
						if verts == nil {
							verts = attr.data.count
							vert_data = make([]Vertex, verts.?)
						}
						if verts != attr.data.count do return
						if attr.data.type != .vec3 do return

						for i in 0..<verts.? {
							raw_vertex_data : [^]f32 = raw_data(vert_data[i].position[:])
							read_result := cgltf.accessor_read_float(attr.data, i, raw_vertex_data, 3)
							if read_result == false {
								fmt.println("Error while reading gltf")
								return
							}
						}
					case .texcoord:
						if verts == nil {
							verts = attr.data.count
							vert_data = make([]Vertex, verts.?)
						}
						if verts != attr.data.count do return
						if attr.data.type != .vec2 do return
						for i in 0..<verts.? {
							raw_vertex_data : [^]f32 = raw_data(vert_data[i].tex_coords[:])
							read_result := cgltf.accessor_read_float(attr.data, i, raw_vertex_data, 2)
							if read_result == false {
								fmt.println("Error while reading gltf")
								return
							}
						}
					case .normal:
						if verts == nil {
							verts = attr.data.count
							vert_data = make([]Vertex, verts.?)
						}
						if verts != attr.data.count do return
						if attr.data.type != .vec3 do return
						for i in 0..<verts.? {
							raw_vertex_data : [^]f32 = raw_data(vert_data[i].normal[:])
							read_result := cgltf.accessor_read_float(attr.data, i, raw_vertex_data, 4)
							if read_result == false {
								fmt.println("Error while reading gltf")
								return
							}
						}
					}
				}

				// if !hasColor {
				// 	for i in 0..<verts.? {
				// 		copy(vert_data[i].color[:], []f32{1,1,1,1})
				// 	}
				// }

				indices : uint = mesh_primitive.indices.count

				index_data : []u32
				index_data = make([]u32, indices)
				defer delete(index_data)

				for i in 0..<indices {
					raw_index_data : [^]u32 = raw_data(index_data[i:i+1])
					read_result := cgltf.accessor_read_uint(mesh_primitive.indices, i, raw_index_data, 1)
					if read_result == false {
						fmt.println("Error while reading gltf")
						return
					}
				}

				meshes["duck"] = createMeshFromData(&vert_data, &index_data)
			}
		}

		state.light_uniform_buffer = wgpu.DeviceCreateBuffer(
			state.device,
			&wgpu.BufferDescriptor {
				label = "Light Uniform Buffer",
				usage = {.Uniform, .CopyDst},
				size = size_of(LightUniform),
			},
		)

		state.camera_uniform_buffer = wgpu.DeviceCreateBuffer(
			state.device,
			&wgpu.BufferDescriptor {
				label = "Uniform Buffer",
				usage = {.Uniform, .CopyDst},
				size = size_of(CameraUniform),
			},
		)

		state.mesh_bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
			state.device,
			&wgpu.BindGroupLayoutDescriptor {
				label = "Uniform Bind Group Layout",
				entryCount = 1,
				entries = raw_data(
					[]wgpu.BindGroupLayoutEntry {
						{
							binding = 0,
							visibility = {.Vertex},
							buffer = {type = .Uniform},
						},
					},
				),
			},
		)

		for &object in objects {
			object.uniform_buffer = wgpu.DeviceCreateBuffer(
				state.device,
				&wgpu.BufferDescriptor {
					label = "Mesh Uniform Buffer",
					usage = {.Uniform, .CopyDst},
					size = size_of(matrix[4, 4]f32),
				},
			)
			object.uniform_bind_group = wgpu.DeviceCreateBindGroup(
				state.device,
				&wgpu.BindGroupDescriptor {
					label = "Mesh Bind Group",
					layout = state.mesh_bind_group_layout,
					entryCount = 1,
					entries = raw_data(
						[]wgpu.BindGroupEntry {
							{
								binding = 0,
								buffer = object.uniform_buffer,
								size = size_of(matrix[4, 4]f32),
							},
						},
					),
				},
			)
		}

		state.scene_bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
			state.device,
			&wgpu.BindGroupLayoutDescriptor {
				label = "Scene Bind Group Layout",
				entryCount = 2,
				entries = raw_data(
					[]wgpu.BindGroupLayoutEntry {
						{
							binding = 0,
							visibility = {.Vertex, .Fragment},
							buffer = {type = .Uniform},
						},
						{
							binding = 1,
							visibility = {.Vertex, .Fragment},
							buffer = {type = .Uniform},
						},
					},
				),
			},
		)

		state.scene_bind_group = wgpu.DeviceCreateBindGroup(
			state.device,
			&wgpu.BindGroupDescriptor {
				label = "Scene Bind Group",
				layout = state.scene_bind_group_layout,
				entryCount = 2,
				entries = raw_data(
					[]wgpu.BindGroupEntry {
						{
							binding = 0,
							buffer = state.camera_uniform_buffer,
							size = size_of(CameraUniform),
						},
						{
							binding = 1,
							buffer = state.light_uniform_buffer,
							size = size_of(LightUniform),
						},
					},
				),
			},
		)

		sample_image, _ := image.load_from_bytes(#load("./assets/textures/sample.png"))
		defer image.destroy(sample_image)
		// Load the image and upload it into a Texture.
		uvTexture = queue_copy_image_to_texture(
			state.device,
			state.queue,
			sample_image,
		)
		uvTextureView = wgpu.TextureCreateView(uvTexture)

		// Create a sampler with linear filtering for smooth interpolation.
		sampler_descriptor := wgpu.SamplerDescriptor {
			label          = "Sampler Descriptor",
			addressModeU = .ClampToEdge,
			addressModeV = .ClampToEdge,
			addressModeW = .ClampToEdge,
			magFilter     = .Linear,
			minFilter     = .Linear,
			mipmapFilter  = .Nearest,
			lodMinClamp  = 0.0,
			lodMaxClamp  = 32.0,
			compare        = .Undefined,
			maxAnisotropy = 1,
		}
		uvTextureSampler = wgpu.DeviceCreateSampler(state.device, &sampler_descriptor)

		samplerBindGroupLayout = wgpu.DeviceCreateBindGroupLayout(
			state.device,
			&wgpu.BindGroupLayoutDescriptor {
				label = "Bind Group Layout",
				entryCount = 2,
				entries = raw_data(
					[]wgpu.BindGroupLayoutEntry {
						{
							binding = 0,
							visibility = { .Fragment },
							texture = {
								sampleType = .Float,
								viewDimension = ._2D,
								multisampled = false,
							},
						},
						{
							binding = 1,
							visibility = { .Fragment },
							sampler = {
								type = .Filtering,
							},
						},
					},
				),
			},
		)

		samplerBindGroup = wgpu.DeviceCreateBindGroup(
			state.device,
			&{
				layout = samplerBindGroupLayout,
				entryCount = 2,
				entries = raw_data(
					[]wgpu.BindGroupEntry {
						{binding = 0, textureView = uvTextureView},
						{binding = 1, sampler = uvTextureSampler},
					},
				),
			},
		)

		shadowSamplerBindGroupLayout = wgpu.DeviceCreateBindGroupLayout(
			state.device,
			&wgpu.BindGroupLayoutDescriptor {
				label = "Shadow Sampler Bind Group Layout",
				entryCount = 2,
				entries = raw_data(
					[]wgpu.BindGroupLayoutEntry {
						{
							binding = 0,
							visibility = { .Fragment },
							texture = wgpu.TextureBindingLayout{
								sampleType = .Depth,
								viewDimension = ._2D,
								multisampled = false,
							},
						},
						{
							binding = 1,
							visibility = { .Fragment },
							sampler = wgpu.SamplerBindingLayout{
								type = .Comparison,
							},
						},
					},
				),
			},
		)

		testSampler := wgpu.DeviceCreateSampler(state.device, &wgpu.SamplerDescriptor{
			label = "Shadow Sampler",
			addressModeU = .ClampToEdge,
			addressModeV = .ClampToEdge,
			addressModeW = .ClampToEdge,
			magFilter = .Linear,
			minFilter = .Linear,
			mipmapFilter = .Nearest,
			lodMinClamp = 0.0,
			lodMaxClamp = 32.0,
			compare = .Less,
			maxAnisotropy = 1,
		})

		createShadowCamera()

		shadowSamplerBindGroup = wgpu.DeviceCreateBindGroup(
			state.device,
			&{
				layout = shadowSamplerBindGroupLayout,
				entryCount = 2,
				entries = raw_data(
					[]wgpu.BindGroupEntry {
						{binding = 0, textureView = shadowDepthTextureView},
						{binding = 1, sampler = testSampler},
					},
				),
			},
		)

		pipelineLayouts["default"] = wgpu.DeviceCreatePipelineLayout(
			state.device,
			&{
				bindGroupLayoutCount = 4,
				bindGroupLayouts = raw_data(
					[]wgpu.BindGroupLayout {
						state.scene_bind_group_layout,
						state.mesh_bind_group_layout,
						samplerBindGroupLayout,
						shadowSamplerBindGroupLayout,
					},
				),
			},
		)

		create_depth_texture()

		pipelines["test"] = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&wgpu.RenderPipelineDescriptor{
				label = "Test Pipeline",
				layout = pipelineLayouts["default"],
				vertex = {
					module = shaders["testShader"],
					entryPoint = "vs_main",
					bufferCount = 1,
					buffers = raw_data(
						[]wgpu.VertexBufferLayout {
							{
								stepMode = .Vertex,
								arrayStride = size_of(Vertex),
								attributeCount = 3,
								attributes = raw_data(
									[]wgpu.VertexAttribute {
										{
											format = .Float32x3,
											offset = u64(offset_of(Vertex, position)),
											shaderLocation = 0,
										},
										{
											format = .Float32x2,
											offset = u64(offset_of(Vertex, tex_coords)),
											shaderLocation = 1,
										},
										{
											format = .Float32x3,
											offset = u64(offset_of(Vertex, normal)),
											shaderLocation = 2,
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
				primitive = {topology = .TriangleList, cullMode = .Back, frontFace = .CCW},
				multisample = {count = 1, mask = 0xFFFFFFFF},
				depthStencil = &wgpu.DepthStencilState{
					depthCompare = .Less,
					stencilReadMask = 0,
					stencilWriteMask = 0,
					depthWriteEnabled = .True,
					format = DEPTH_FORMAT,
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

		pipelineLayouts["shadow"] = wgpu.DeviceCreatePipelineLayout(
			state.device,
			&{
				bindGroupLayoutCount = 2,
				bindGroupLayouts = raw_data(
					[]wgpu.BindGroupLayout {
						state.scene_bind_group_layout,
						state.mesh_bind_group_layout,
					},
				),
			},
		)

		pipelines["shadow"] = wgpu.DeviceCreateRenderPipeline(
			state.device,
			&wgpu.RenderPipelineDescriptor{
				label = "Shadow Pipeline",
				layout = pipelineLayouts["shadow"],
				vertex = {
					module = shaders["shadowCaster"],
					entryPoint = "vs_main",
					bufferCount = 1,
					buffers = raw_data(
						[]wgpu.VertexBufferLayout {
							{
								stepMode = .Vertex,
								arrayStride = size_of(Vertex),
								attributeCount = 1,
								attributes = raw_data(
									[]wgpu.VertexAttribute {
										{
											format = .Float32x3,
											offset = u64(offset_of(Vertex, position)),
											shaderLocation = 0,
										},
									},
								),
							},
						},
					),
				},
				primitive = {topology = .TriangleList, cullMode = .Back, frontFace = .CCW},
				multisample = {count = 1, mask = 0xFFFFFFFF},
				depthStencil = &wgpu.DepthStencilState{
					depthWriteEnabled = .True,
					depthCompare = .Less,
					format = .Depth32Float,
				},
			},
		)

		mu_init()

		init_game_state()
		defer destroy_game_state()

		display_game_state()

		os_run()
	}
}

create_depth_texture :: proc(){
	if depthTexture != nil do wgpu.TextureRelease(depthTexture)
	depthTexture = wgpu.DeviceCreateTexture(
		state.device,
		&wgpu.TextureDescriptor{
			size = wgpu.Extent3D{
				width = state.config.width,
				height = state.config.height,
				depthOrArrayLayers = 1,
			},
			format = DEPTH_FORMAT,
			usage = {.RenderAttachment},
			dimension = ._2D,
			sampleCount = 1,
			mipLevelCount = 1,
			viewFormatCount = 0,
		},
	)

	if depthTextureView != nil do wgpu.TextureViewRelease(depthTextureView)
	depthTextureView = wgpu.TextureCreateView(depthTexture, &wgpu.TextureViewDescriptor{
		aspect = .DepthOnly,
		baseArrayLayer = 0,
		arrayLayerCount = 1,
		baseMipLevel = 0,
		mipLevelCount = 1,
		dimension = ._2D,
		format = DEPTH_FORMAT,
	})
}

//These dummy's are only needed for passing to a shaders that has the layout but does not actually use the values.
//This is not currently used so is commented.
// dummyStorageBuffer : wgpu.Buffer
// dummyShadowTexture : wgpu.Texture
shadowDepthTexture : wgpu.Texture
shadowDepthTextureView : wgpu.TextureView

createShadowCamera :: proc() {
	shadowDepthTexture = wgpu.DeviceCreateTexture(state.device, &wgpu.TextureDescriptor{
		size = wgpu.Extent3D{
			width = 2048,
			height = 2048,
			depthOrArrayLayers = 1,
		},
		usage = {.RenderAttachment, .TextureBinding},
		format = DEPTH_FORMAT,
		dimension = ._2D,
		sampleCount = 1,
		mipLevelCount = 1,
	})
	shadowDepthTextureView = wgpu.TextureCreateView(shadowDepthTexture)
}

resize :: proc "c" () {
	context = state.ctx

	state.config.width, state.config.height = os_get_render_bounds()

	projectionMatrix = la.matrix4_perspective(
		2 * math.PI / 5,
		f32(state.config.width) / f32(state.config.height),
		1.0,
		100.0,
	)

	wgpu.SurfaceConfigure(state.surface, &state.config)

	create_depth_texture()

	mu_resize()
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
	case .SuccessOptimal, .SuccessSuboptimal:
	// All good, could check for `surface_texture.suboptimal` here.
	case .Timeout, .Outdated, .Lost:
		// Skip this frame, and re-configure surface.
		if surface_texture.texture != nil {
			wgpu.TextureRelease(surface_texture.texture)
		}
		resize()
		return
	case .OutOfMemory, .DeviceLost, .Error:
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

	//Transform objects
	now := f32(time.duration_seconds(time.since(start_time)))
	gameObject1.rotation = la.quaternion_from_euler_angles_f32(
		0,
		math.sin(now) * 1.2,
		0,
		la.Euler_Angle_Order.XYZ,
	)
	gameObject2.rotation = la.quaternion_from_euler_angles_f32(
		math.sin(now) * 1.2,
		math.cos(now) * 1,
		0,
		la.Euler_Angle_Order.XYZ,
	)

	//Setup matrices and write positions to uniform buffers
	viewMatrix = la.MATRIX4F32_IDENTITY
	viewMatrix *= la.matrix4_rotate(flyCamera.pitch, la.VECTOR3F32_X_AXIS)
	viewMatrix *= la.matrix4_rotate(flyCamera.yaw, la.VECTOR3F32_Y_AXIS)
	viewMatrix *= la.matrix4_translate(flyCamera.camera.position)

	transform := OPEN_GL_TO_WGPU_MATRIX * projectionMatrix * viewMatrix
	cameraData := CameraUniform {
		view_proj = transform,
		position = la.Vector4f32 {flyCamera.position.x, flyCamera.position.y, flyCamera.position.z, 1.0},
	}

	wgpu.QueueWriteBuffer(state.queue, state.camera_uniform_buffer, 0, &cameraData, size_of(CameraUniform))
	wgpu.QueueWriteBuffer(state.queue, state.light_uniform_buffer, 0, &directional_light, size_of(LightUniform))
	for &object in objects {
		modelMatrix = la.matrix4_from_trs_f32(
			object.translation,
			object.rotation,
			object.scale,
		)
		wgpu.QueueWriteBuffer(state.queue, object.uniform_buffer, 0, &modelMatrix, size_of(transform))
	}

	//Shadow render pass
	shadow_command_encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)
	defer wgpu.CommandEncoderRelease(shadow_command_encoder)

	shadow_render_pass_encoder := wgpu.CommandEncoderBeginRenderPass(
		shadow_command_encoder,
		&wgpu.RenderPassDescriptor{
			colorAttachmentCount = 0,
			depthStencilAttachment = &wgpu.RenderPassDepthStencilAttachment{
				view = shadowDepthTextureView,
				depthClearValue = 1.0,
				depthLoadOp = .Clear,
				depthStoreOp = .Store,
			},
		},
	)
	wgpu.RenderPassEncoderSetPipeline(shadow_render_pass_encoder, pipelines["shadow"])
	// RENDER EVERYTHING IN SHADOW PASS
	wgpu.RenderPassEncoderSetBindGroup(shadow_render_pass_encoder, 0, state.scene_bind_group)
	render_objects(shadow_render_pass_encoder)

	wgpu.RenderPassEncoderEnd(shadow_render_pass_encoder)
	wgpu.RenderPassEncoderRelease(shadow_render_pass_encoder)
	shadow_command_buffer := wgpu.CommandEncoderFinish(shadow_command_encoder, nil)
	defer wgpu.CommandBufferRelease(shadow_command_buffer)
	wgpu.QueueSubmit(state.queue, {shadow_command_buffer})


	//Render pass
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
				depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
				clearValue = {0.2, 0.2, 0.2, 1},
			},
			depthStencilAttachment = &wgpu.RenderPassDepthStencilAttachment {
				view = depthTextureView,
				depthClearValue = 1.0,
				depthLoadOp = .Clear,
				depthStoreOp = .Store,
				depthReadOnly = false,
				stencilClearValue = 0,
				stencilLoadOp = .Undefined,
				stencilStoreOp = .Undefined,
				stencilReadOnly = true,
			},
		},
	)

	wgpu.RenderPassEncoderSetPipeline(render_pass_encoder, pipelines["test"])
	wgpu.RenderPassEncoderSetBindGroup(render_pass_encoder, 0, state.scene_bind_group)
	wgpu.RenderPassEncoderSetBindGroup(render_pass_encoder, 2, samplerBindGroup)
	wgpu.RenderPassEncoderSetBindGroup(render_pass_encoder, 3, shadowSamplerBindGroup)

	render_objects(render_pass_encoder)

	wgpu.RenderPassEncoderEnd(render_pass_encoder)
	wgpu.RenderPassEncoderRelease(render_pass_encoder)

	command_buffer := wgpu.CommandEncoderFinish(command_encoder, nil)
	defer wgpu.CommandBufferRelease(command_buffer)

	wgpu.QueueSubmit(state.queue, {command_buffer})

	mu.begin(&mu_ctx)
    demo_windows(&mu_ctx)
    mu.end(&mu_ctx)
    mu_render(frame)

	wgpu.SurfacePresent(state.surface)
}

render_objects :: proc(render_pass_encoder : wgpu.RenderPassEncoder) {
	for &object, object_index in objects {
		wgpu.RenderPassEncoderSetBindGroup(render_pass_encoder, 1, object.uniform_bind_group)

		mesh := &meshes[object.mesh]
	
		if (mesh.vertices != 0 && mesh.vertexBuffer != nil){
			wgpu.RenderPassEncoderSetVertexBuffer(
				render_pass_encoder,
				0,
				mesh.vertexBuffer,
				0,
				u64(mesh.vertices * size_of(Vertex)),
			)
		}
		if (mesh.indices != 0 && mesh.indexBuffer != nil){
			wgpu.RenderPassEncoderSetIndexBuffer(
				render_pass_encoder,
				mesh.indexBuffer,
				.Uint32,
				0,
				u64(mesh.indices * size_of(u32)),
			)
		}

		if mesh.indices != 0 && mesh.indexBuffer != nil {
			wgpu.RenderPassEncoderDrawIndexed(
				render_pass_encoder,
				u32(mesh.indices),
				instanceCount = 1,
				firstIndex = 0,
				baseVertex = 0,
				firstInstance = u32(object_index),
			)
		} else {
			wgpu.RenderPassEncoderDraw(
				render_pass_encoder,
				vertexCount = u32(mesh.vertices),
				instanceCount = 1,
				firstVertex = 0,
				firstInstance = u32(object_index),
			)
		}
	}
}

finish :: proc() {
	mu_shutdown()

	wgpu.SamplerRelease(uvTextureSampler)
	wgpu.TextureViewRelease(uvTextureView)
	wgpu.TextureRelease(uvTexture)
	cleanup_objects()
	cleanup_meshes()
	cleanup_pipelines()
	wgpu.TextureViewRelease(depthTextureView)
	wgpu.TextureRelease(depthTexture)
	cleanup_pipeline_layouts()
	cleanup_shaders()
	delete(objects)
	wgpu.BufferRelease(state.light_uniform_buffer)
	wgpu.BufferRelease(state.camera_uniform_buffer)

	wgpu.BindGroupRelease(state.scene_bind_group)
	wgpu.BindGroupLayoutRelease(state.scene_bind_group_layout)
	wgpu.BindGroupRelease(samplerBindGroup)
	wgpu.BindGroupLayoutRelease(samplerBindGroupLayout)
	wgpu.BindGroupRelease(shadowSamplerBindGroup)
	wgpu.BindGroupLayoutRelease(shadowSamplerBindGroupLayout)

	wgpu.QueueRelease(state.queue)
	wgpu.DeviceRelease(state.device)
	wgpu.AdapterRelease(state.adapter)
	wgpu.InstanceRelease(state.instance)
}

cleanup_objects :: proc() {
	for object in objects {
		wgpu.BindGroupRelease(object.uniform_bind_group)
		wgpu.BufferRelease(object.uniform_buffer)
	}
}

cleanup_meshes :: proc() {
	for key in meshes {
		mesh := &meshes[key]
		if mesh.vertexBuffer != nil {
			wgpu.BufferRelease(mesh.vertexBuffer)
		}
		if mesh.indexBuffer != nil {
			wgpu.BufferRelease(mesh.indexBuffer)
		}
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


u8_slider :: proc(ctx: ^mu.Context, val: ^u8, lo, hi: u8) -> (res: mu.Result_Set) {
	mu.push_id(ctx, uintptr(val))

	@static tmp: mu.Real
	tmp = mu.Real(val^)
	res = mu.slider(ctx, &tmp, mu.Real(lo), mu.Real(hi), 0, "%.0f", {.ALIGN_CENTER})
	val^ = u8(tmp)
	mu.pop_id(ctx)
	return
}

f32_slider :: proc(ctx: ^mu.Context, val: ^f32, lo, hi: f32, step: f32) -> (res: mu.Result_Set) {
	mu.push_id(ctx, uintptr(val))

	@static tmp: mu.Real
	tmp = val^
	res = mu.slider(ctx, &tmp, lo, hi, step, "%.0f", {.ALIGN_CENTER})
	val^ = tmp
	mu.pop_id(ctx)
	return
}

quat_to_euler_sliders :: proc(ctx: ^mu.Context, val: ^la.Quaternionf32) -> (res: mu.Result_Set) {
	mu.push_id(ctx, uintptr(val))

	@static x: mu.Real
	@static y: mu.Real
	@static z: mu.Real

	x, y, z = la.euler_angles_xyz_from_quaternion(val^)
	// x = la.pitch_from_quaternion(val^) * 180 / la.PI
	// y = la.yaw_from_quaternion(val^) * 180 / la.PI
	// z = la.roll_from_quaternion(val^) * 180 / la.PI
	x = x * la.DEG_PER_RAD
	y = y * la.DEG_PER_RAD
	z = z * la.DEG_PER_RAD
	res = mu.slider(ctx, &x, -180, 180, 0.1, "%.0f", {.ALIGN_CENTER})
	res = mu.slider(ctx, &y, -180, 180, 0.1, "%.0f", {.ALIGN_CENTER})
	res = mu.slider(ctx, &z, -180, 180, 0.1, "%.0f", {.ALIGN_CENTER})
	x = x * la.RAD_PER_DEG
	y = y * la.RAD_PER_DEG
	z = z * la.RAD_PER_DEG
	// val^ = la.quaternion_from_pitch_yaw_roll(x / 180 * la.PI, y / 180 * la.PI, z / 180 * la.PI)
	val^ = la.quaternion_from_euler_angles(x, y, z, .XYZ)
	mu.pop_id(ctx)
	return
}

write_log :: proc(str: string) {
	muState.log_buf_len += copy(muState.log_buf[muState.log_buf_len:], str)
	muState.log_buf_len += copy(muState.log_buf[muState.log_buf_len:], "\n")
	muState.log_buf_updated = true
}

read_log :: proc() -> string {
	return string(muState.log_buf[:muState.log_buf_len])
}
reset_log :: proc() {
	muState.log_buf_updated = true
	muState.log_buf_len = 0
}

demo_windows :: proc(ctx: ^mu.Context) {
	@static opts := mu.Options{.NO_CLOSE}

	// if mu.window(ctx, "Demo Window", {350, 40, 300, 450}, opts) {
	// 	if .ACTIVE in mu.header(ctx, "Window Info") {
	// 		win := mu.get_current_container(ctx)
	// 		mu.layout_row(ctx, {54, -1}, 0)
	// 		mu.label(ctx, "Position:")
	// 		mu.label(ctx, fmt.tprintf("%d, %d", win.rect.x, win.rect.y))
	// 		mu.label(ctx, "Size:")
	// 		mu.label(ctx, fmt.tprintf("%d, %d", win.rect.w, win.rect.h))
	// 	}

	// 	if .ACTIVE in mu.header(ctx, "Window Options") {
	// 		mu.layout_row(ctx, {120, 120, 120}, 0)
	// 		for opt in mu.Opt {
	// 			state := opt in opts
	// 			if .CHANGE in mu.checkbox(ctx, fmt.tprintf("%v", opt), &state)  {
	// 				if state {
	// 					opts += {opt}
	// 				} else {
	// 					opts -= {opt}
	// 				}
	// 			}
	// 		}
	// 	}

	// 	if .ACTIVE in mu.header(ctx, "Test Buttons", {.EXPANDED}) {
	// 		mu.layout_row(ctx, {86, -110, -1})
	// 		mu.label(ctx, "Test buttons 1:")
	// 		if .SUBMIT in mu.button(ctx, "Button 1") { write_log("Pressed button 1") }
	// 		if .SUBMIT in mu.button(ctx, "Button 2") { write_log("Pressed button 2") }
	// 		mu.label(ctx, "Test buttons 2:")
	// 		if .SUBMIT in mu.button(ctx, "Button 3") { write_log("Pressed button 3") }
	// 		if .SUBMIT in mu.button(ctx, "Button 4") { write_log("Pressed button 4") }
	// 	}

	// 	if .ACTIVE in mu.header(ctx, "Tree and Text", {.EXPANDED}) {
	// 		mu.layout_row(ctx, {140, -1})
	// 		mu.layout_begin_column(ctx)
	// 		if .ACTIVE in mu.treenode(ctx, "Test 1") {
	// 			if .ACTIVE in mu.treenode(ctx, "Test 1a") {
	// 				mu.label(ctx, "Hello")
	// 				mu.label(ctx, "world")
	// 			}
	// 			if .ACTIVE in mu.treenode(ctx, "Test 1b") {
	// 				if .SUBMIT in mu.button(ctx, "Button 1") { write_log("Pressed button 1") }
	// 				if .SUBMIT in mu.button(ctx, "Button 2") { write_log("Pressed button 2") }
	// 			}
	// 		}
	// 		if .ACTIVE in mu.treenode(ctx, "Test 2") {
	// 			mu.layout_row(ctx, {53, 53})
	// 			if .SUBMIT in mu.button(ctx, "Button 3") { write_log("Pressed button 3") }
	// 			if .SUBMIT in mu.button(ctx, "Button 4") { write_log("Pressed button 4") }
	// 			if .SUBMIT in mu.button(ctx, "Button 5") { write_log("Pressed button 5") }
	// 			if .SUBMIT in mu.button(ctx, "Button 6") { write_log("Pressed button 6") }
	// 		}
	// 		if .ACTIVE in mu.treenode(ctx, "Test 3") {
	// 			@static checks := [3]bool{true, false, true}
	// 			mu.checkbox(ctx, "Checkbox 1", &checks[0])
	// 			mu.checkbox(ctx, "Checkbox 2", &checks[1])
	// 			mu.checkbox(ctx, "Checkbox 3", &checks[2])

	// 		}
	// 		mu.layout_end_column(ctx)

	// 		mu.layout_begin_column(ctx)
	// 		mu.layout_row(ctx, {-1})
	// 		mu.text(ctx,
	// 			"Lorem ipsum dolor sit amet, consectetur adipiscing "+
	// 			"elit. Maecenas lacinia, sem eu lacinia molestie, mi risus faucibus "+
	// 			"ipsum, eu varius magna felis a nulla.",
	// 		)
	// 		mu.layout_end_column(ctx)
	// 	}

	// 	if .ACTIVE in mu.header(ctx, "Background Colour", {.EXPANDED}) {
	// 		mu.layout_row(ctx, {-78, -1}, 68)
	// 		mu.layout_begin_column(ctx)
	// 		{
	// 			mu.layout_row(ctx, {46, -1}, 0)
	// 			mu.label(ctx, "Red:");   u8_slider(ctx, &muState.bg.r, 0, 255)
	// 			mu.label(ctx, "Green:"); u8_slider(ctx, &muState.bg.g, 0, 255)
	// 			mu.label(ctx, "Blue:");  u8_slider(ctx, &muState.bg.b, 0, 255)
	// 		}
	// 		mu.layout_end_column(ctx)

	// 		r := mu.layout_next(ctx)
	// 		mu.draw_rect(ctx, r, muState.bg)
	// 		mu.draw_box(ctx, mu.expand_rect(r, 1), ctx.style.colors[.BORDER])
	// 		mu.draw_control_text(ctx, fmt.tprintf("#%02x%02x%02x", muState.bg.r, muState.bg.g, muState.bg.b), r, .TEXT, {.ALIGN_CENTER})
	// 	}
	// }

	if mu.window(ctx, "Log Window", {40, 40, 300, 200}, opts) {
		mu.layout_row(ctx, {-1}, -28)
		mu.begin_panel(ctx, "Log")
		mu.layout_row(ctx, {-1}, -1)
		mu.text(ctx, read_log())
		if muState.log_buf_updated {
			panel := mu.get_current_container(ctx)
			panel.scroll.y = panel.content_size.y
			muState.log_buf_updated = false
		}
		mu.end_panel(ctx)

		@static buf: [128]byte
		@static buf_len: int
		submitted := false
		mu.layout_row(ctx, {-70, -1})
		if .SUBMIT in mu.textbox(ctx, buf[:], &buf_len) {
			mu.set_focus(ctx, ctx.last_id)
			submitted = true
		}
		if .SUBMIT in mu.button(ctx, "Submit") {
			submitted = true
		}
		if submitted {
			write_log(string(buf[:buf_len]))
			buf_len = 0
		}
	}

	if mu.window(ctx, "Scene", {40, 250, 300, 240}) {
		sw := i32(f32(mu.get_current_container(ctx).body.w) * 0.14)
		mu.layout_row(ctx, {80, sw, sw, sw})
		// mu.label(ctx, "Light")
		// f32_slider(ctx, &point_light.position.x, -10, 10, 0.1)
		// f32_slider(ctx, &point_light.position.y, -10, 10, 0.1)
		// f32_slider(ctx, &point_light.position.z, -10, 10, 0.1)

		mu.label(ctx, "Cube Pos")
		f32_slider(ctx, &gameObject1.translation.x, -10, 10, 0.1)
		f32_slider(ctx, &gameObject1.translation.y, -10, 10, 0.1)
		f32_slider(ctx, &gameObject1.translation.z, -10, 10, 0.1)

		mu.label(ctx, "Duck Pos")
		f32_slider(ctx, &gameObject2.translation.x, -10, 10, 0.1)
		f32_slider(ctx, &gameObject2.translation.y, -10, 10, 0.1)
		f32_slider(ctx, &gameObject2.translation.z, -10, 10, 0.1)

		mu.label(ctx, "Plane Pos")
		f32_slider(ctx, &gameObject3.translation.x, -10, 10, 0.1)
		f32_slider(ctx, &gameObject3.translation.y, -10, 10, 0.1)
		f32_slider(ctx, &gameObject3.translation.z, -10, 10, 0.1)

		mu.label(ctx, "Plane Rot")
		quat_to_euler_sliders(ctx, &gameObject3.rotation)
	}

	// if mu.window(ctx, "Style Window", {40, 250, 300, 240}) {
	// 	@static colors := [mu.Color_Type]string{
	// 		.TEXT         = "text",
	// 		.BORDER       = "border",
	// 		.WINDOW_BG    = "window bg",
	// 		.TITLE_BG     = "title bg",
	// 		.TITLE_TEXT   = "title text",
	// 		.PANEL_BG     = "panel bg",
	// 		.BUTTON       = "button",
	// 		.BUTTON_HOVER = "button hover",
	// 		.BUTTON_FOCUS = "button focus",
	// 		.BASE         = "base",
	// 		.BASE_HOVER   = "base hover",
	// 		.BASE_FOCUS   = "base focus",
	// 		.SCROLL_BASE  = "scroll base",
	// 		.SCROLL_THUMB = "scroll thumb",
	// 		.SELECTION_BG = "selection bg",
	// 	}

	// 	sw := i32(f32(mu.get_current_container(ctx).body.w) * 0.14)
	// 	mu.layout_row(ctx, {80, sw, sw, sw, sw, -1})
	// 	for label, col in colors {
	// 		mu.label(ctx, label)
	// 		u8_slider(ctx, &ctx.style.colors[col].r, 0, 255)
	// 		u8_slider(ctx, &ctx.style.colors[col].g, 0, 255)
	// 		u8_slider(ctx, &ctx.style.colors[col].b, 0, 255)
	// 		u8_slider(ctx, &ctx.style.colors[col].a, 0, 255)
	// 		mu.draw_rect(ctx, mu.layout_next(ctx), ctx.style.colors[col])
	// 	}
	// }
}