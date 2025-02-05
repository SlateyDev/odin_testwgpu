package test

import "core:fmt"

GameState :: struct {
    alive : bool,
}

game_state : ^GameState

init_game_state :: proc() {
    game_state = new(GameState)
}

destroy_game_state :: proc() {
    free(game_state)
}

display_game_state :: proc() {
    fmt.println("Game State:", game_state)
}