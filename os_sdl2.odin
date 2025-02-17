package test

import "core:c"
import "core:fmt"
import "core:strings"

import mu "vendor:microui"
import "vendor:sdl2"
import "vendor:wgpu"
import "vendor:wgpu/sdl2glue"

OS :: struct {
    window: ^sdl2.Window,
}

os_init :: proc(os: ^OS) {
    sdl_flags := sdl2.InitFlags{.VIDEO, .JOYSTICK, .GAMECONTROLLER, .EVENTS}
    if res := sdl2.Init(sdl_flags); res != 0 {
		fmt.eprintfln("ERROR: Failed to initialize SDL: [%s]", sdl2.GetError())
		return
	}

	window_flags: sdl2.WindowFlags = {.SHOWN, .ALLOW_HIGHDPI, .RESIZABLE}
	os.window = sdl2.CreateWindow(
		"Test",
		sdl2.WINDOWPOS_CENTERED,
		sdl2.WINDOWPOS_CENTERED,
		960,
		540,
		window_flags,
	)
	if os.window == nil {
		fmt.eprintfln("ERROR: Failed to create the SDL Window: [%s]", sdl2.GetError())
		return
	}

	sdl2.AddEventWatch(event_callback, nil)
}

os_run :: proc(os: ^OS) {
	now := sdl2.GetPerformanceCounter()
	last : u64
	dt: f32
	main_loop: for {
		last = now
		now = sdl2.GetPerformanceCounter()
		dt = f32((now - last) * 1000) / f32(sdl2.GetPerformanceFrequency())

		e: sdl2.Event

		for sdl2.PollEvent(&e) {
			#partial switch (e.type) {
			case .QUIT:
				break main_loop
			}
		}

		frame(dt)
	}

	sdl2.DestroyWindow(os.window)
	sdl2.Quit()

	finish()
}


os_get_render_bounds :: proc(os: ^OS) -> (width, height: u32) {
	iw, ih: c.int
	sdl2.GetWindowSize(os.window, &iw, &ih)
	return u32(iw), u32(ih)
}

os_get_dpi :: proc() -> f32 {
	return 1
}

os_get_surface :: proc(os: ^OS, instance: wgpu.Instance) -> wgpu.Surface {
	return sdl2glue.GetSurface(instance, os.window)
}

@(private="file")
event_callback :: proc "c" (userdata: rawptr, event: ^sdl2.Event) -> c.int {
	context = state.ctx
	#partial switch event.type {
		case .WINDOWEVENT:
			if event.window.event == .SIZE_CHANGED || event.window.event == .RESIZED {
				resize()
			}
		case .MOUSEBUTTONDOWN:
			switch event.button.button {
			case sdl2.BUTTON_LEFT:
				mu.input_mouse_down(&mu_ctx, event.button.x, event.button.y, .LEFT)
			case sdl2.BUTTON_MIDDLE:
				mu.input_mouse_down(&mu_ctx, event.button.x, event.button.y, .MIDDLE)
			case sdl2.BUTTON_RIGHT:
				mu.input_mouse_down(&mu_ctx, event.button.x, event.button.y, .RIGHT)
			}
		case .MOUSEBUTTONUP:
			switch event.button.button {
			case sdl2.BUTTON_LEFT:
				mu.input_mouse_up(&mu_ctx, event.button.x, event.button.y, .LEFT)
			case sdl2.BUTTON_MIDDLE:
				mu.input_mouse_up(&mu_ctx, event.button.x, event.button.y, .MIDDLE)
			case sdl2.BUTTON_RIGHT:
				mu.input_mouse_up(&mu_ctx, event.button.x, event.button.y, .RIGHT)
			}
		case .MOUSEMOTION:
			mu.input_mouse_move(&mu_ctx, event.motion.x, event.motion.y)
		case .MOUSEWHEEL:
			mu.input_scroll(&mu_ctx, event.wheel.x * 30, event.wheel.y * -30)

		case .TEXTINPUT:
			mu.input_text(&mu_ctx, string(cstring(&event.text.text[0])))

		case .KEYDOWN, .KEYUP:
			if event.type == .KEYUP && event.key.keysym.sym == .ESCAPE {
				sdl2.PushEvent(&sdl2.Event{type = .QUIT})
			}

			fn := mu.input_key_down if event.type == .KEYDOWN else mu.input_key_up

			#partial switch event.key.keysym.sym {
			case .LSHIFT:    fn(&mu_ctx, .SHIFT)
			case .RSHIFT:    fn(&mu_ctx, .SHIFT)
			case .LCTRL:     fn(&mu_ctx, .CTRL)
			case .RCTRL:     fn(&mu_ctx, .CTRL)
			case .LALT:      fn(&mu_ctx, .ALT)
			case .RALT:      fn(&mu_ctx, .ALT)
			case .RETURN:    fn(&mu_ctx, .RETURN)
			case .KP_ENTER:  fn(&mu_ctx, .RETURN)
			case .BACKSPACE: fn(&mu_ctx, .BACKSPACE)

			case .LEFT:  fn(&mu_ctx, .LEFT)
			case .RIGHT: fn(&mu_ctx, .RIGHT)
			case .HOME:  fn(&mu_ctx, .HOME)
			case .END:   fn(&mu_ctx, .END)
			case .A:     fn(&mu_ctx, .A)
			case .X:     fn(&mu_ctx, .X)
			case .C:     fn(&mu_ctx, .C)
			case .V:     fn(&mu_ctx, .V)
			}
		}
	return 0
}

os_set_clipboard :: proc(user_data: rawptr, text: string) -> (ok: bool) {
	cstr := strings.clone_to_cstring(text)
	sdl2.SetClipboardText(cstr)
	delete(cstr)
	return true
}

os_get_clipboard :: proc(user_data: rawptr) -> (text: string, ok: bool) {
	if sdl2.HasClipboardText() {
		text = string(sdl2.GetClipboardText())
		ok = true
	}
	return
}