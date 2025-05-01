package test

import "vendor:wgpu"

texture_from_colour_f32 :: proc(colour: [4]f32) -> wgpu.Texture {
    return texture_from_colour_u8({u8(colour.r * 255), u8(colour.g * 255), u8(colour.b * 255), u8(colour.a * 255)})
}

texture_from_colour_u8 :: proc(colour: [4]u8) -> wgpu.Texture {
    colour := colour
    newTexture := wgpu.DeviceCreateTexture(state.device, &wgpu.TextureDescriptor{
        label = "Colour Texture",
        size = wgpu.Extent3D{1, 1, 1},
        usage = {.TextureBinding, .CopyDst},
        mipLevelCount = 1,
        sampleCount = 1,
        dimension = ._2D,
        format = .RGBA8Unorm,
    })
    wgpu.QueueWriteTexture(
        state.queue,
        &wgpu.TexelCopyTextureInfo{
            texture = newTexture,
            mipLevel = 0,
            origin = {},
            aspect = .All,
        },
        &colour,
        len(colour),
        &wgpu.TexelCopyBufferLayout{
            offset       = 0,
            bytesPerRow  = texture_format_bytes_per_row(.RGBA8Unorm, 1),
            rowsPerImage = 1,
        },
        &wgpu.Extent3D{
            width = 1,
            height = 1,
            depthOrArrayLayers = 1,
        },
    )

    return newTexture
}