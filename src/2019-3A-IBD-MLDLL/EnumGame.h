#pragma once

typedef enum {
	FPS,
	RTS,
	MOBA,
	FPS_TEST,
	RTS_TEST,
	MOBA_TEST
} GAME;

char* getGame(GAME g)
{
	switch (g)
	{
	case FPS:
		return "FPS";
	case RTS:
		return "RTS";
	case MOBA:
		return "MOBA";
	default:
		return "NO GAME SELECTED";
	}
}