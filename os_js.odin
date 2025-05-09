#+feature dynamic-literals
package test

import "core:sys/wasm/js"
import "core:unicode"
import "core:unicode/utf8"

import "vendor:wgpu"
import mu "vendor:microui"

OS :: struct {
	initialized: bool,
    clipboard: [dynamic]byte,
}

os : OS
mu_state := struct {
	cursor: [2]i32,
}{}

os_init :: proc() {
	assert(js.add_window_event_listener(.Key_Down, nil, key_down_callback))
	assert(js.add_window_event_listener(.Key_Up, nil, key_up_callback))
	assert(js.add_window_event_listener(.Mouse_Down, nil, mouse_down_callback))
	assert(js.add_window_event_listener(.Mouse_Up, nil, mouse_up_callback))
	assert(js.add_event_listener("wgpu-canvas", .Mouse_Move, nil, mouse_move_callback))
	assert(js.add_window_event_listener(.Wheel, nil, scroll_callback))
	assert(js.add_window_event_listener(.Resize, nil, size_callback))
}

// NOTE: frame loop is done by the runtime.js repeatedly calling `step`.
os_run :: proc() {
	os.initialized = true
}

@(private="file", export)
step :: proc(dt: f32) -> bool {
	context = state.ctx

    if !os.initialized {
		return true
	}
    
	frame(dt)
	return true
}

os_get_render_bounds :: proc() -> (width, height: u32) {
	rect := js.get_bounding_client_rect("body")
	dpi := os_get_dpi()
	return u32(f32(rect.width) * dpi), u32(f32(rect.height) * dpi)
}

os_get_dpi :: proc() -> f32 {
	ratio := f32(js.device_pixel_ratio())
	return ratio
}

os_get_framebuffer_size :: proc() -> (width, height: u32) {
	rect := js.get_bounding_client_rect("body")
	dpi := js.device_pixel_ratio()
	return u32(f64(rect.width) * dpi), u32(f64(rect.height) * dpi)
}

os_get_surface :: proc(instance: wgpu.Instance) -> wgpu.Surface {
	return wgpu.InstanceCreateSurface(
		instance,
		&wgpu.SurfaceDescriptor{
			nextInChain = &wgpu.SurfaceSourceCanvasHTMLSelector{
				sType = .SurfaceSourceCanvasHTMLSelector,
				selector = "#wgpu-canvas",
			},
		},
	)
}

os_set_clipboard :: proc(_: rawptr, text: string) -> bool {
	// TODO: Use browser APIs
	clear(&os.clipboard)
	append(&os.clipboard, text)
	return true
}

os_get_clipboard :: proc(_: rawptr) -> (string, bool) {
	// TODO: Use browser APIs
	return string(os.clipboard[:]), true
}

@(private="file", fini)
os_fini :: proc() {
	js.remove_window_event_listener(.Key_Down, nil, key_down_callback)
	js.remove_window_event_listener(.Key_Up, nil, key_up_callback)
	js.remove_window_event_listener(.Mouse_Down, nil, mouse_down_callback)
	js.remove_window_event_listener(.Mouse_Up, nil, mouse_up_callback)
	js.remove_event_listener("wgpu-canvas", .Mouse_Move, nil, mouse_move_callback)
	js.remove_window_event_listener(.Wheel, nil, scroll_callback)
	js.remove_window_event_listener(.Resize,   nil, size_callback)

    finish()
}

@(private="file")
size_callback :: proc(e: js.Event) {
    context = state.ctx
	resize()
}

@(private="file")
KEY_MAP := map[string]mu.Key{
	"ShiftLeft"    = .SHIFT,
	"ShiftRight"   = .SHIFT,
	"ControlLeft"  = .CTRL,
	"ControlRight" = .CTRL,
	"MetaLeft"     = .CTRL,
	"MetaRight"    = .CTRL,
	"AltLeft"      = .ALT,
	"AltRight"     = .ALT,
	"Backspace"    = .BACKSPACE,
	"Delete"       = .DELETE,
	"Enter"        = .RETURN,
	"ArrowLeft"    = .LEFT,
	"ArrowRight"   = .RIGHT,
	"Home"         = .HOME,
	"End"          = .END,
	"KeyA"         = .A,
	"KeyX"         = .X,
	"KeyC"         = .C,
	"KeyV"         = .V,
}

@(private="file")
key_down_callback :: proc(e: js.Event) {
	context = state.ctx

	js.event_prevent_default()

	if k, ok := KEY_MAP[e.data.key.code]; ok {
		mu.input_key_down(&mu_ctx, k)
	}

	if .CTRL in mu_ctx.key_down_bits {
		return
	}

	ch, size := utf8.decode_rune(e.data.key.key)
	if len(e.data.key.key) == size && unicode.is_print(ch) {
		mu.input_text(&mu_ctx, e.data.key.key)
	}
}

@(private="file")
key_up_callback :: proc(e: js.Event) {
	context = state.ctx

	if k, ok := KEY_MAP[e.data.key.code]; ok {
		mu.input_key_up(&mu_ctx, k)
	}

	js.event_prevent_default()
}

@(private="file")
mouse_down_callback :: proc(e: js.Event) {
	context = state.ctx

	switch e.data.mouse.button {
	case 0: mu.input_mouse_down(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .LEFT)
	case 1: mu.input_mouse_down(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .MIDDLE)
	case 2: mu.input_mouse_down(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .RIGHT)
	}

	js.event_prevent_default()
}

@(private="file")
mouse_up_callback :: proc(e: js.Event) {
	context = state.ctx

	switch e.data.mouse.button {
	case 0: mu.input_mouse_up(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .LEFT)
	case 1: mu.input_mouse_up(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .MIDDLE)
	case 2: mu.input_mouse_up(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .RIGHT)
	}

	js.event_prevent_default()
}

@(private="file")
mouse_move_callback :: proc(e: js.Event) {
	context = state.ctx
	mu_state.cursor = {i32(e.data.mouse.offset.x), i32(e.data.mouse.offset.y)}
	mu.input_mouse_move(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y)
}

@(private="file")
scroll_callback :: proc(e: js.Event) {
	context = state.ctx
	mu.input_scroll(&mu_ctx, i32(e.data.wheel.delta.x), i32(e.data.wheel.delta.y))
}