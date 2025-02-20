package test

// Code modified from that found at https://github.com/Capati/wgpu-odin

// STD Library
import "base:runtime"
import "core:fmt"
import "core:image"
import "core:mem"
import "core:slice"
import "core:strings"
import "vendor:wgpu"

// Vendor
import stbi "vendor:stb/image"

COPY_BYTES_PER_ROW_ALIGNMENT: u32 : 256

Texture_Resource :: struct {
	texture: wgpu.Texture,
	sampler: wgpu.Sampler,
	view:    wgpu.TextureView,
}

Texture_Creation_Options :: struct {
	label            : string,
	srgb             : bool,
	usage            : wgpu.TextureUsageFlags,
	preferred_format : Maybe(wgpu.TextureFormat),
}

Image_Data_Type :: union {
	[]byte,
	[]u16,
	[]f32,
}

Image_Info :: struct {
	width, height, channels : int,
	is_hdr                  : bool,
	bits_per_channel        : int,
}

Image_Data :: struct {
	using info        : Image_Info,
	total_size        : int,
	bytes_per_channel : int,
	is_float          : bool,
	raw_data          : rawptr,
	data              : Image_Data_Type,
}

queue_copy_image_to_texture_from_image_data :: proc(
	self: wgpu.Device,
	queue: wgpu.Queue,
	data: Image_Data,
	options: Texture_Creation_Options = {},
	loc := #caller_location,
) -> (
	texture: wgpu.Texture,
	ok: bool,
) #optional_ok {
	options := options

	width, height := data.width, data.height

	// Default texture usage if none is given
	if options.usage == {} {
		options.usage = {.TextureBinding, .CopyDst, .RenderAttachment}
	}

	// Determine the texture format based on the image info
	format := options.preferred_format.? or_else image_info_texture_format(data.info)

	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	label: cstring = nil
	if options.label != "" {
		label = strings.clone_to_cstring(options.label, context.temp_allocator)
	}

	// Create the texture
	texture_desc := wgpu.TextureDescriptor {
		label = label,
		size = {width = u32(width), height = u32(height), depthOrArrayLayers = 1},
		mipLevelCount = 1,
		sampleCount = 1,
		dimension = ._2D,
		format = format,
		usage = options.usage,
	}
	texture = wgpu.DeviceCreateTexture(self, &texture_desc)
	defer if !ok do wgpu.TextureRelease(texture)

	bytes_per_row := texture_format_bytes_per_row(format, u32(width))

	// Prepare image data for upload
	image_copy_texture := texture_as_image_copy(texture)
	texture_data_layout := wgpu.TextureDataLayout {
		offset         = 0,
		bytesPerRow  = bytes_per_row,
		rowsPerImage = u32(height),
	}

	// Convert image data if necessary
	pixels_to_upload := _convert_image_data(
		data,
		format,
		bytes_per_row,
		context.temp_allocator,
	) or_return

	// Copy image data to texture
	wgpu.QueueWriteTexture(
		queue,
		&image_copy_texture,
		raw_data(pixels_to_upload),
        len(pixels_to_upload),
		&texture_data_layout,
		&texture_desc.size,
	)

	return texture, true
}

/* Make an `Image_Copy_Texture` representing the whole texture with the given origin. */
texture_as_image_copy :: proc "contextless" (
	self: wgpu.Texture,
	origin: wgpu.Origin3D = {},
) -> wgpu.ImageCopyTexture {
	return { texture = self, mipLevel = 0, origin = origin, aspect = wgpu.TextureAspect.All }
}

get_image_info_stbi_from_c_string_path :: proc(
	image_path: cstring,
	loc := #caller_location,
) -> (
	info: Image_Info,
	ok: bool,
) #optional_ok {
	w, h, c: i32
	if stbi.info(image_path, &w, &h, &c) == 0 {
		// error_reset_and_update(
		// 	.Load_Image_Failed,
			fmt.printf(
				"Failed to get image info for '%s': %s",
				image_path,
				stbi.failure_reason(),
			)
			// loc,
		// )
		return
	}

	info.width, info.height, info.channels = int(w), int(h), int(c)
	info.is_hdr = stbi.is_hdr(image_path) != 0

	// Determine bits per channel
	if info.is_hdr {
		info.bits_per_channel = 32 // Assuming 32-bit float for HDR
	} else {
		info.bits_per_channel = stbi.is_16_bit(image_path) ? 16 : 8
	}

	return info, true
}

get_image_info_stbi_from_string_path :: proc(
	image_path: string,
	loc := #caller_location,
) -> (
	info: Image_Info,
	ok: bool,
) #optional_ok {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	c_image_path := strings.clone_to_cstring(image_path, context.temp_allocator)
	return get_image_info_stbi_from_c_string_path(c_image_path, loc)
}

get_image_info_stbi :: proc {
	get_image_info_stbi_from_c_string_path,
	get_image_info_stbi_from_string_path,
}

Load_Method :: enum {
	Default, // 8-bit channels
	Load_16,
	Load_F32,
}

image_info_determine_load_method :: proc(info: Image_Info) -> Load_Method {
	if info.is_hdr {
		return .Load_F32
	} else if info.bits_per_channel == 16 {
		return .Load_16
	}
	return .Default
}

load_image_data_stbi :: proc(
	image_path: string,
	loc := #caller_location,
) -> (
	image_data: Image_Data,
	ok: bool,
) #optional_ok {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	c_image_path := strings.clone_to_cstring(image_path, context.temp_allocator)

	image_data.info = get_image_info_stbi(c_image_path, loc) or_return

	method := image_info_determine_load_method(image_data.info)

	width, height, channels: i32

	switch method {
	case .Default:
		image_data.raw_data = stbi.load(c_image_path, &width, &height, &channels, 0)
		image_data.bytes_per_channel = 1
	case .Load_16:
		image_data.raw_data = stbi.load_16(c_image_path, &width, &height, &channels, 0)
		image_data.bytes_per_channel = 2
	case .Load_F32:
		image_data.raw_data = stbi.loadf(c_image_path, &width, &height, &channels, 0)
		image_data.bytes_per_channel = 4
		image_data.is_float = true
	}

	if image_data.raw_data == nil {
		// error_reset_and_update(
		// 	.Load_Image_Failed,
			fmt.printf("Failed to load image '%s': %s", image_path, stbi.failure_reason())
		// 	loc,
		// )
		return
	}

	image_data.total_size = int(width * height * channels)

	switch method {
	case .Default:
		image_data.data = mem.slice_ptr(cast([^]byte)image_data.raw_data, image_data.total_size)
	case .Load_16:
		image_data.data = mem.slice_ptr(cast([^]u16)image_data.raw_data, image_data.total_size)
	case .Load_F32:
		image_data.data = mem.slice_ptr(cast([^]f32)image_data.raw_data, image_data.total_size)
	}

	return image_data, true
}

queue_copy_image_to_texture_from_path :: proc(
	self: wgpu.Device,
	queue: wgpu.Queue,
	image_path: string,
	options: Texture_Creation_Options = {},
	loc := #caller_location,
) -> (
	texture: wgpu.Texture,
	ok: bool,
) #optional_ok {
    image_data := load_image_data_stbi(image_path, loc) or_return
	defer stbi.image_free(image_data.raw_data)

	texture = queue_copy_image_to_texture_from_image_data(
		self,
		queue,
		image_data,
		options,
		loc,
	) or_return

	return texture, true
}

queue_copy_image_to_texture_image_paths :: proc(
	self: wgpu.Device,
	queue: wgpu.Queue,
	image_paths: []string,
	options: Texture_Creation_Options = {},
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	textures: []wgpu.Texture,
	ok: bool,
) #optional_ok {
	textures = make([]wgpu.Texture, len(image_paths), allocator)
	defer if !ok {
		for &t in textures {
			wgpu.TextureDestroy(t)
			wgpu.TextureRelease(t)
		}
	}

	for path, i in image_paths {
		image_data := load_image_data_stbi(path, loc) or_return
		defer stbi.image_free(image_data.raw_data)

		textures[i] = queue_copy_image_to_texture_from_image_data(
			self,
			queue,
			image_data,
			options,
			loc,
		) or_return
	}

	return textures, true
}

queue_copy_image_to_texture_image :: proc(
	self: wgpu.Device,
	queue: wgpu.Queue,
	image: ^image.Image,
	options: Texture_Creation_Options = {},
	loc := #caller_location,
) -> (
	texture: wgpu.Texture,
	ok: bool,
) #optional_ok {
	bytes_per_channel := 1
	if image.depth > 8 {
		bytes_per_channel = 2
	}

	total_size := image.width * image.height * image.channels

	typed_data: Image_Data_Type
	if bytes_per_channel == 1 {
		typed_data = mem.slice_ptr(raw_data(image.pixels.buf[:]), total_size)
	} else {
		typed_data = mem.slice_ptr(cast([^]u16)raw_data(image.pixels.buf[:]), total_size)
	}

	image_data := Image_Data {
		width             = image.width,
		height            = image.height,
		channels          = image.channels,
		bytes_per_channel = bytes_per_channel,
		data              = typed_data,
	}

	texture = queue_copy_image_to_texture_from_image_data(
		self,
		queue,
		image_data,
		options,
		loc,
	) or_return

	return texture, true
}

queue_copy_image_to_texture :: proc {
	queue_copy_image_to_texture_from_path,
	queue_copy_image_to_texture_image_paths,
	queue_copy_image_to_texture_image,
}

image_info_texture_format :: proc(info: Image_Info) -> wgpu.TextureFormat {
	if info.is_hdr {
		switch info.channels {
		case 1:
			return .R32Float
		case 2:
			return .RG32Float
		case 3, 4:
			return .RGBA32Float
		}
	} else if info.bits_per_channel == 16 {
		switch info.channels {
		case 1:
			return .R16Uint
		case 2:
			return .RG16Uint
		case 3, 4:
			return .RGBA16Uint
		}
	} else {
		switch info.channels {
		case 1:
			return .R8Unorm
		case 2:
			return .RG8Unorm
		case 3, 4:
			return .RGBA8Unorm
		}
	}

	return .RGBA8Unorm // Default to RGBA8 if channels are unexpected
}

queue_create_cubemap_texture :: proc(
	self: wgpu.Device,
	queue: wgpu.Queue,
	image_paths: [6]string,
	options: Texture_Creation_Options = {},
	loc := #caller_location,
) -> (
	out: Texture_Resource,
	ok: bool,
) #optional_ok {
	options := options

	// Get info of the first image
	first_info := get_image_info_stbi(image_paths[0], loc) or_return

	// Default texture usage if none is given
	if options.usage == {} {
		options.usage = {.TextureBinding, .CopyDst, .RenderAttachment}
	}

	// Determine the texture format based on the image info or use the preferred format
	format := options.preferred_format.? or_else image_info_texture_format(first_info)

	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	c_label: cstring = nil
	if options.label != "" {
		c_label = strings.clone_to_cstring(options.label, context.temp_allocator)
	}

	// Create the cubemap texture
	texture_desc := wgpu.TextureDescriptor {
		label = c_label,
		size = {
			width = u32(first_info.width),
			height = u32(first_info.height),
			depthOrArrayLayers = 6,
		},
		mipLevelCount = 1,
		sampleCount = 1,
		dimension = ._2D,
		format = format,
		usage = options.usage,
	}
	out.texture = wgpu.DeviceCreateTexture(self, &texture_desc)
	defer if !ok do wgpu.TextureRelease(out.texture)

	// Calculate bytes per row, ensuring it meets the WGPU alignment requirements
	bytes_per_row := texture_format_bytes_per_row(format, u32(first_info.width))

	// Load and copy each face of the cubemap
	for i in 0 ..< 6 {
		// Check info of each face
		face_info := get_image_info_stbi(image_paths[i], loc) or_return

		if face_info != first_info {
			// error_reset_and_update(
			// 	.Validation,
				fmt.printf("Cubemap face '%s' has different properties", image_paths[i])
			// 	loc,
			// )
			return
		}

		// Load the face image
		face_image_data := load_image_data_stbi(image_paths[i], loc) or_return
		defer stbi.image_free(face_image_data.raw_data)

		// Copy the face image to the appropriate layer of the cubemap texture
		origin := wgpu.Origin3D{0, 0, u32(i)}

		// Prepare image data for upload
		image_copy_texture := texture_as_image_copy(out.texture, origin)
		texture_data_layout := wgpu.TextureDataLayout {
			offset         = 0,
			bytesPerRow  = bytes_per_row,
			rowsPerImage = u32(face_image_data.height),
		}

		// Convert image data if necessary
		pixels_to_upload := _convert_image_data(
			face_image_data,
			format,
			bytes_per_row,
			context.temp_allocator,
		) or_return

		wgpu.QueueWriteTexture(
			queue,
			&image_copy_texture,
			&pixels_to_upload,
            len(pixels_to_upload),
			&texture_data_layout,
			&{u32(face_image_data.width), u32(face_image_data.height), 1},
		)
	}

	cube_view_descriptor := wgpu.TextureViewDescriptor {
		label             = "Cube Texture View",
		format            = wgpu.TextureGetFormat(out.texture), // Use the same format as the texture
		dimension         = .Cube,
		baseMipLevel    = 0,
		mipLevelCount   = 1, // Assume no mipmaps
		baseArrayLayer  = 0,
		arrayLayerCount = 6, // 6 faces of the cube
		aspect            = .All,
	}
	out.view = wgpu.TextureCreateView(out.texture, &cube_view_descriptor)
	defer if !ok do wgpu.TextureViewRelease(out.view)

	// Create a sampler with linear filtering for smooth interpolation.
	sampler_descriptor := wgpu.SamplerDescriptor {
		addressModeU = .Repeat,
		addressModeV = .Repeat,
		addressModeW = .Repeat,
		magFilter     = .Linear,
		minFilter     = .Linear,
		mipmapFilter  = .Linear,
		lodMinClamp  = 0.0,
		lodMaxClamp  = 1.0,
		compare        = .Undefined,
		maxAnisotropy = 1,
	}

	out.sampler = wgpu.DeviceCreateSampler(self, &sampler_descriptor)
	// defer if !ok do sampler_release(out.sampler)

	return out, true
}

texture_resource_release :: proc(res: Texture_Resource) {
	wgpu.SamplerRelease(res.sampler)
	wgpu.TextureViewRelease(res.view)
	wgpu.TextureDestroy(res.texture)
	wgpu.TextureRelease(res.texture)
}

@(private = "file")
_convert_image_data :: proc(
	image_data: Image_Data,
	format: wgpu.TextureFormat,
	aligned_bytes_per_row: u32,
	allocator := context.allocator,
) -> (
	data: []byte,
	ok: bool,
) {
	bytes_per_pixel := image_data.channels * image_data.bytes_per_channel

	if image_data.channels == 3 {
		// Convert RGB to RGBA
		new_bytes_per_pixel := 4 * image_data.bytes_per_channel
		data = make([]byte, int(aligned_bytes_per_row) * image_data.height, allocator)

		switch src in image_data.data {
		case []byte:
			for y in 0 ..< image_data.height {
				for x in 0 ..< image_data.width {
					src_idx := (y * image_data.width + x) * bytes_per_pixel
					dst_idx := y * int(aligned_bytes_per_row) + x * new_bytes_per_pixel
					copy(data[dst_idx:], src[src_idx:src_idx + bytes_per_pixel])
					data[dst_idx + 3] = 255 // Full alpha for 8-bit
				}
			}
		case []u16:
			for y in 0 ..< image_data.height {
				for x in 0 ..< image_data.width {
					src_idx := (y * image_data.width + x) * image_data.channels
					dst_idx := y * int(aligned_bytes_per_row) + x * 4 * 2
					for j in 0 ..< 3 {
						(^u16)(&data[dst_idx + j * 2])^ = src[src_idx + j]
					}
					(^u16)(&data[dst_idx + 6])^ = 65535 // Full alpha for 16-bit
				}
			}
		case []f32:
			for y in 0 ..< image_data.height {
				for x in 0 ..< image_data.width {
					src_idx := (y * image_data.width + x) * image_data.channels
					dst_idx := y * int(aligned_bytes_per_row) + x * 4 * 4
					for j in 0 ..< 3 {
						(^f32)(&data[dst_idx + j * 4])^ = src[src_idx + j]
					}
					(^f32)(&data[dst_idx + 12])^ = 1.0 // Full alpha for float
				}
			}
		}
	} else {
		// Check if the source data is already properly aligned
		src_bytes_per_row := u32(image_data.width * bytes_per_pixel)
		if src_bytes_per_row == aligned_bytes_per_row {
			// If already aligned, we can simply reinterpret the data
			switch src in image_data.data {
			case []byte:
				data = src
			case []u16:
				data = slice.reinterpret([]byte, src)
			case []f32:
				data = slice.reinterpret([]byte, src)
			}
			return data, true
		}

		// If not converting, create a byte slice of the data with proper alignment
		total_size := int(aligned_bytes_per_row) * image_data.height
		data = make([]byte, total_size, allocator)

		copy_image_data :: proc(
			$T: typeid,
			src: ^[]T,
			dst: ^[]byte,
			image_data: Image_Data,
			aligned_bytes_per_row: int,
		) {
			bytes_per_pixel := size_of(T) * image_data.channels

			for y in 0 ..< image_data.height {
				src_row := src[y * image_data.width * image_data.channels:]
				dst_row := dst[y * aligned_bytes_per_row:]

				when T == byte {
					copy(dst_row[:image_data.width * bytes_per_pixel], src_row)
				} else {
					copy_slice(
						dst_row[:image_data.width * bytes_per_pixel],
						slice.reinterpret(
							[]byte,
							src_row[:image_data.width * image_data.channels],
						),
					)
				}
			}
		}

		switch &src in image_data.data {
		case []byte:
			copy_image_data(byte, &src, &data, image_data, int(aligned_bytes_per_row))
		case []u16:
			copy_image_data(u16, &src, &data, image_data, int(aligned_bytes_per_row))
		case []f32:
			copy_image_data(f32, &src, &data, image_data, int(aligned_bytes_per_row))
		}
	}

	return data, true
}

/*
Returns the dimension of a [block](https://gpuweb.github.io/gpuweb/#texel-block) of texels.

Uncompressed formats have a block dimension of `(1, 1)`.
*/
texture_format_block_dimensions :: proc "contextless" (self: wgpu.TextureFormat) -> (w, h: u32) {
	#partial switch self {
	case .R8Unorm, .R8Snorm, .R8Uint, .R8Sint, .R16Uint, .R16Sint, .R16Unorm,
	     .R16Snorm, .R16Float, .RG8Unorm, .RG8Snorm, .RG8Uint, .RG8Sint, .R32Uint,
	     .R32Sint, .R32Float, .RG16Uint, .RG16Sint, .Rg16Unorm, .Rg16Snorm,
		 .RG16Float, .RGBA8Unorm, .RGBA8UnormSrgb, .RGBA8Snorm, .RGBA8Uint, .RGBA8Sint,
	     .BGRA8Unorm, .BGRA8UnormSrgb, .RGB9E5Ufloat, .RGB10A2Uint, .RGB10A2Unorm,
		 .RG11B10Ufloat, .RG32Uint, .RG32Sint, .RG32Float, .RGBA16Uint, .RGBA16Sint,
	     .Rgba16Unorm, .Rgba16Snorm, .RGBA16Float, .RGBA32Uint, .RGBA32Sint,
		 .RGBA32Float, .Stencil8, .Depth16Unorm, .Depth24Plus, .Depth24PlusStencil8,
	     .Depth32Float, .Depth32FloatStencil8:
		return 1, 1

	case .BC1RGBAUnorm, .BC1RGBAUnormSrgb, .BC2RGBAUnorm, .BC2RGBAUnormSrgb,
		 .BC3RGBAUnorm, .BC3RGBAUnormSrgb, .BC4RUnorm, .BC4RSnorm, .BC5RGUnorm,
	  	 .BC5RGSnorm, .BC6HRGBUfloat, .BC6HRGBFloat, .BC7RGBAUnorm, .BC7RGBAUnormSrgb,
		 .ETC2RGB8Unorm, .ETC2RGB8UnormSrgb, .ETC2RGB8A1Unorm, .ETC2RGB8A1UnormSrgb,
		 .ETC2RGBA8Unorm, .ETC2RGBA8UnormSrgb, .EACR11Unorm, .EACR11Snorm, .EACRG11Unorm,
		 .EACRG11Snorm:
		return 4, 4

	case .ASTC4x4Unorm, .ASTC4x4UnormSrgb: return 4, 4
	case .ASTC5x4Unorm, .ASTC5x4UnormSrgb: return 5, 5
	case .ASTC5x5Unorm, .ASTC5x5UnormSrgb: return 5, 5
	case .ASTC6x5Unorm, .ASTC6x5UnormSrgb: return 6, 5
	case .ASTC6x6Unorm, .ASTC6x6UnormSrgb: return 6, 6
	case .ASTC8x5Unorm, .ASTC8x5UnormSrgb: return 8, 5
	case .ASTC8x6Unorm, .ASTC8x6UnormSrgb: return  8, 6
	case .ASTC8x8Unorm, .ASTC8x8UnormSrgb: return 8, 8
	case .ASTC10x5Unorm, .ASTC10x5UnormSrgb: return 10, 5
	case .ASTC10x6Unorm, .ASTC10x6UnormSrgb: return 10, 6
	case .ASTC10x8Unorm, .ASTC10x8UnormSrgb: return 10, 8
	case .ASTC10x10Unorm, .ASTC10x10UnormSrgb: return 10, 10
	case .ASTC12x10Unorm, .ASTC12x10UnormSrgb: return 12, 10
	case .ASTC12x12Unorm, .ASTC12x12UnormSrgb: return 12, 12
	}

	return 1, 1
}

/*
The number of bytes one [texel block](https://gpuweb.github.io/gpuweb/#texel-block) occupies
during an image copy, if applicable.

Known as the [texel block copy footprint](https://gpuweb.github.io/gpuweb/#texel-block-copy-footprint).

Note that for uncompressed formats this is the same as the size of a single texel,
since uncompressed formats have a block size of 1x1.

Returns `0` if any of the following are true:
 - the format is a combined depth-stencil and no `aspect` was provided
 - the format is a multi-planar format and no `aspect` was provided
 - the format is `Depth24Plus`
 - the format is `Depth24PlusStencil8` and `aspect` is depth.
*/
texture_format_block_size :: proc "contextless" (
	self: wgpu.TextureFormat,
	aspect: Maybe(wgpu.TextureAspect) = nil,
) -> u32 {
	_aspect, aspect_ok := aspect.?

	#partial switch self {
	case .R8Unorm, .R8Snorm, .R8Uint, .R8Sint:
		return 1

	case .RG8Unorm, .RG8Snorm, .RG8Uint, .RG8Sint:
		return 2

	case .R16Uint, .R16Sint, .R16Float, .RG16Uint, .RG16Sint, .RG16Float,
	     .RGB10A2Uint, .RG11B10Ufloat, .RGB9E5Ufloat:
		return 4

	case .RGBA8Unorm, .RGBA8UnormSrgb, .RGBA8Snorm, .RGBA8Uint, .RGBA8Sint,
	     .BGRA8Unorm, .BGRA8UnormSrgb:
		return 4

	case .R32Uint, .R32Sint, .R32Float, .RG32Uint, .RG32Sint, .RG32Float:
		return 8

	case .RGBA16Uint, .RGBA16Sint, .RGBA16Float, .RGBA32Uint, .RGBA32Sint, .RGBA32Float,
	     .BC1RGBAUnorm, .BC1RGBAUnormSrgb, .BC4RUnorm, .BC4RSnorm, .BC5RGUnorm,
	     .BC5RGSnorm, .BC6HRGBUfloat, .BC6HRGBFloat, .BC7RGBAUnorm, .BC7RGBAUnormSrgb,
		 .ETC2RGB8Unorm, .ETC2RGB8UnormSrgb, .ETC2RGB8A1Unorm, .ETC2RGB8A1UnormSrgb,
		 .ETC2RGBA8Unorm, .ETC2RGBA8UnormSrgb, .EACR11Unorm, .EACR11Snorm, .EACRG11Unorm,
	     .EACRG11Snorm, .ASTC4x4Unorm, .ASTC4x4UnormSrgb, .ASTC5x4Unorm, .ASTC5x4UnormSrgb,
	     .ASTC5x5Unorm, .ASTC5x5UnormSrgb, .ASTC6x5Unorm, .ASTC6x5UnormSrgb, .ASTC6x6Unorm,
	     .ASTC6x6UnormSrgb, .ASTC8x5Unorm, .ASTC8x5UnormSrgb, .ASTC8x6Unorm,
		 .ASTC8x6UnormSrgb, .ASTC8x8Unorm, .ASTC8x8UnormSrgb, .ASTC10x5Unorm,
	     .ASTC10x5UnormSrgb, .ASTC10x6Unorm, .ASTC10x6UnormSrgb, .ASTC10x8Unorm,
	     .ASTC10x8UnormSrgb, .ASTC10x10Unorm, .ASTC10x10UnormSrgb, .ASTC12x10Unorm,
	     .ASTC12x10UnormSrgb, .ASTC12x12Unorm, .ASTC12x12UnormSrgb:
		return 16

	case .Stencil8:
		return 1

	case .Depth16Unorm:
		return 2

	case .Depth32Float:
		return 4

	case .Depth24Plus:
		return 0

	case .Depth24PlusStencil8:
		if aspect_ok {
			#partial switch _aspect {
			case .DepthOnly:
				return 0
			case .StencilOnly:
				return 1
			}
		}
		return 0

	case .Depth32FloatStencil8:
		if aspect_ok {
			#partial switch _aspect {
			case .DepthOnly:
				return 4
			case .StencilOnly:
				return 1
			}
		}
		return 0
	}

	return 0
}

/* Calculate bytes per row from the given row width. */
texture_format_bytes_per_row :: proc "contextless" (
	format: wgpu.TextureFormat,
	width: u32,
) -> (
	bytes_per_row: u32,
) {
	block_width, _ := texture_format_block_dimensions(format)
	block_size := texture_format_block_size(format)

	// Calculate the number of blocks for the given width
	blocks_in_width := (width + block_width - 1) / block_width

	// Calculate unaligned bytes per row
	unaligned_bytes_per_row := blocks_in_width * block_size

	// Align to COPY_BYTES_PER_ROW_ALIGNMENT
	bytes_per_row =
		(unaligned_bytes_per_row + COPY_BYTES_PER_ROW_ALIGNMENT - 1) &
		~(COPY_BYTES_PER_ROW_ALIGNMENT - 1)

	return
}