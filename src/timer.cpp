#include "../include/timer.h"

timer::timer::timer() : start(std::chrono::high_resolution_clock::now()) {};

void timer::reset() {
	start = std::chrono::high_resolution_clock::now();
}

double timer::elapsed() const {
	auto duration = std::chrono::high_resolution_clock::now() - start;
	return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0;
}

